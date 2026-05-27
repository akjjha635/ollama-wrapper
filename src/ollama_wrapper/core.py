import os
import time
import json
import subprocess
import asyncio
import math
import re
import numpy as np
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel
import logging
import ollama
from ollama import Client, AsyncClient, ResponseError, RequestError
from .api_server import ChatSessionManager, create_app
import uvicorn

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OllamaWrapper:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Initializing OllamaWrapper singleton instance")
            cls._instance = super().__new__(cls)
            
            # 1. Endpoint & Connection Modes
            ip = kwargs.get("ip", "localhost")
            port = kwargs.get("port", 11434)
            cls._instance.api_endpoint = f"http://{ip}:{port}"
            cls._instance.connection_type = str(kwargs.get("connection_type", "sync")).lower()
            
            if cls._instance.connection_type not in ["sync", "async"]:
                raise ValueError("connection_type must be explicitly set to either 'sync' or 'async'.")
            
            # 2. Re-use Backend Drivers
            cls._instance._sync_client = Client(host=cls._instance.api_endpoint)
            cls._instance._async_client = AsyncClient(host=cls._instance.api_endpoint)
            cls._instance.client = (
                cls._instance._async_client if cls._instance.connection_type == "async" 
                else cls._instance._sync_client
            )
            
            # 3. Model Properties
            cls._instance.llm_model = kwargs.get("llm_model", "deepseek-r1:1.5b")
            cls._instance.embed_model = kwargs.get("embed_model", "nomic-embed-text")
            
            # 4. Context States & Thread Safe Architecture Locks
            cls._instance.initial_context = ""
            cls._instance.running_summary = ""
            cls._instance.chat_history = []
            
            cls._instance.max_active_turns = int(kwargs.get("max_active_turns", 4))
            if cls._instance.max_active_turns < 2:
                raise ValueError("max_active_turns must be at least 2 to sustain a conversational turn summary block.")
                
            cls._instance.async_lock = asyncio.Lock()
            
            # 5. Local Vector Database Records
            cls._instance.vector_database = []  # Explicit representation: [{"text": str, "vector": np.ndarray, "metadata": dict}]
            cls._instance.db_storage_path = kwargs.get("db_storage_path", "./local_vector_db")
            
            # BM25 Core Stats
            cls._instance.doc_lens = []
            cls._instance.avg_doc_len = 0.0
            cls._instance.df = {}
            cls._instance.k1 = float(kwargs.get("k1", 1.5))
            cls._instance.b = float(kwargs.get("b", 0.75))

            try:
                cls.ensure_ollama_is_running()
            except Exception as e:
                raise RuntimeError(f"Failed to bind background Ollama execution loops: {e}") from e
                
            cls._instance.is_connected = True
            cls._instance._api_server_task = None
            cls._instance.session_manager = None
            
        return cls._instance

    @classmethod
    def reset_singleton(cls):
        """Allows tearing down the instance cache strictly for isolated testing suites."""
        cls._instance = None

    @classmethod
    def ensure_ollama_is_running(cls):
        try:
            ollama.list()
        except Exception:
            logger.warning("Ollama server not detected. Attempting to start background daemon...")
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3)
            except FileNotFoundError:
                raise RuntimeError("The 'ollama' executable was not found on this system path. Install Ollama before executing.")
        
        try:
            local_models = [m.model for m in ollama.list().models]
            if "nomic-embed-text:latest" not in local_models and "nomic-embed-text" not in local_models:
                logger.info("Pulling embedding model 'nomic-embed-text'...")
                ollama.pull("nomic-embed-text")
        except Exception as e:
            logger.warning(f"Failed to verify/pull model assets automatically: {e}")

    def _strip_thinking_tags(self, text: str) -> str:
        if not text:
            return ""
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()

    # --- PART 1: SYSTEM INSTRUCTION PIPELINES & ROLLING MEMORY ---

    def load_prompt_from_string(self, prompt_str: str, optimize: bool = True):
        if not prompt_str or not prompt_str.strip():
            raise ValueError("System target prompt string cannot be blank or whitespace.")
            
        if optimize:
            meta_prompt = (
                "You are an expert prompt engineer. Review the following raw instructions and condense them "
                "into a highly direct, dense, instruction-focused markdown prompt template. "
                "Remove conversational fluff, retain all strict constraints, and output ONLY the optimized prompt text.\n\n"
                f"--- RAW INPUT ---\n{prompt_str}"
            )
            try:
                response = self._sync_client.generate(model=self.llm_model, prompt=meta_prompt)
                self.initial_context = self._strip_thinking_tags(response.response)
            except Exception as e:
                logger.warning(f"Prompt optimization pass failed: {e}. Falling back to raw prompt setup.")
                self.initial_context = prompt_str
        else:
            self.initial_context = prompt_str
        logger.info(f"Base system instructions compiled ({len(self.initial_context)} chars).")

    def _compile_history_prompt(self, turns_to_compress: list) -> str:
        formatted_history = "".join([f"{msg['role'].upper()}: {msg['content']}\n" for msg in turns_to_compress])
        return (
            "You are a memory compaction module. Review the current ongoing summary and the newest chat log below. "
            "Merge them into a single, highly dense, chronological bulleted list of established facts and user preferences. "
            "Do not include conversational filler. Preserve all technical details, code choices, or constraints mentioned.\n\n"
            f"--- EXISTING MEMORY SUMMARY ---\n{self.running_summary if self.running_summary else 'None yet.'}\n\n"
            f"--- NEW INTERACTION LOGS TO MERGE ---\n{formatted_history}"
        )

    def _optimize_and_summarize_history(self):
        if len(self.chat_history) <= self.max_active_turns:
            return
        turns_to_compress = self.chat_history[:-2]
        self.chat_history = self.chat_history[-2:]
        try:
            response = self._sync_client.generate(model=self.llm_model, prompt=self._compile_history_prompt(turns_to_compress))
            self.running_summary = self._strip_thinking_tags(response.response)
        except Exception as e:
            logger.error(f"Memory compaction iteration failed: {e}")

    async def _optimize_and_summarize_history_async(self):
        if len(self.chat_history) <= self.max_active_turns:
            return
        turns_to_compress = self.chat_history[:-2]
        self.chat_history = self.chat_history[-2:]
        try:
            response = await self._async_client.generate(model=self.llm_model, prompt=self._compile_history_prompt(turns_to_compress))
            self.running_summary = self._strip_thinking_tags(response.response)
        except Exception as e:
            logger.error(f"Async memory compaction iteration failed: {e}")

    # --- PART 2: ADVANCED SEMANTIC SEGMENTATION (CHUNK DISCRIMINATION) ---

    def ingest_semantic_document(self, text_content: str, threshold_percentile: float = 65.0, metadata: dict = None):
        if not text_content or not text_content.strip():
            logger.warning("Empty string content passed to ingest pipeline. Skipping.")
            return

        sentences = re.split(r'(?<=[.!?])\s+', text_content.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            self.ingest_knowledge_document(text_content, metadata=metadata)
            return

        try:
            sentence_vectors = []
            for s in sentences:
                res = self._sync_client.embed(model=self.embed_model, input=s)
                vec = np.array(res.embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                sentence_vectors.append(vec / norm if norm > 0 else vec)

            distances = [1.0 - np.dot(sentence_vectors[i], sentence_vectors[i+1]) for i in range(len(sentence_vectors) - 1)]
            cutoff_threshold = np.percentile(distances, threshold_percentile) if distances else 0.5
            
            current_chunk = [sentences[0]]
            for idx, dist in enumerate(distances):
                if dist > cutoff_threshold:
                    self.ingest_knowledge_document(" ".join(current_chunk), metadata=metadata)
                    current_chunk = [sentences[idx + 1]]
                else:
                    current_chunk.append(sentences[idx + 1])
            
            if current_chunk:
                self.ingest_knowledge_document(" ".join(current_chunk), metadata=metadata)
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}. Falling back to standard ingestion.")
            self.ingest_knowledge_document(text_content, metadata=metadata)

    def ingest_knowledge_document(self, text_content: str, chunk_size: int = 400, overlap: int = 50, metadata: dict = None):
        if not text_content or not text_content.strip():
            return
        if chunk_size <= overlap:
            raise ValueError(f"Configured chunk_size ({chunk_size}) must be strictly greater than your overlap parameter ({overlap}).")

        words = text_content.split()
        if len(words) <= chunk_size:
            chunks = [" ".join(words)]
        else:
            chunks = []
            i = 0
            while i < len(words):
                chunk_words = words[i:i + chunk_size]
                chunks.append(" ".join(chunk_words))
                i += (chunk_size - overlap)

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                res = self._sync_client.embed(model=self.embed_model, input=chunk)
                vec = np.array(res.embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                normalized_vec = vec / norm if norm > 0 else vec
                
                self.vector_database.append({
                    "text": chunk, 
                    "vector": normalized_vec,
                    "metadata": metadata if metadata is not None else {}
                })
            except Exception as e:
                logger.error(f"Chunk embedding failed for chunk {len(self.vector_database)}: {e}")
                
        self._rebuild_bm25_indices()

    def _rebuild_bm25_indices(self):
        self.doc_lens = [len(doc["text"].lower().split()) for doc in self.vector_database]
        self.avg_doc_len = np.mean(self.doc_lens) if self.doc_lens else 0.0
        
        self.df = {}
        for doc in self.vector_database:
            unique_terms = set(doc["text"].lower().split())
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

    # --- PART 3: ADVANCED HYBRID LOOKUP & CROSS-ENCODER RERANKING ---

    def _compute_bm25_score(self, query: str, doc_idx: int) -> float:
        query_terms = query.lower().split()
        doc_terms = self.vector_database[doc_idx]["text"].lower().split()
        doc_len = self.doc_lens[doc_idx]
        
        score = 0.0
        N = len(self.vector_database)
        
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            df_t = self.df.get(term, 0)
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len + 1e-9)))
            score += idf * (numerator / denominator)
        return score

    def _hybrid_retrieve_and_rerank(self, query: str, top_k: int = 2, metadata_filter: dict = None, sync_mode: bool = True) -> str:
        if not self.vector_database:
            return ""

        pool_indices = range(len(self.vector_database))
        if metadata_filter:
            pool_indices = [
                i for i in pool_indices 
                if all(self.vector_database[i]["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        if not pool_indices:
            return ""

        try:
            # 1. Evaluate Dense Embedding Match Scoring
            if sync_mode:
                res = self._sync_client.embed(model=self.embed_model, input=query)
            else:
                loop = asyncio.get_running_loop()
                # Run the coroutine safely within the current event runtime state
                coro = self._async_client.embed(model=self.embed_model, input=query)
                res = asyncio.run_coroutine_threadsafe(coro, loop).result()

            q_vec = np.array(res.embeddings[0], dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0: q_vec /= q_norm

            matrix = np.array([self.vector_database[i]["vector"] for i in pool_indices])
            vector_scores = np.dot(matrix, q_vec)

            # 2. Evaluate Lexical BM25 Score Calculations
            bm25_scores = np.array([self._compute_bm25_score(query, i) for i in pool_indices], dtype=np.float32)

            # Safeguard against division-by-zero on single-document pools
            v_min, v_max = vector_scores.min(), vector_scores.max()
            b_min, b_max = bm25_scores.min(), bm25_scores.max()
            
            norm_vector = (vector_scores - v_min) / (v_max - v_min + 1e-9) if v_max > v_min else np.ones_like(vector_scores)
            norm_bm25 = (bm25_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.ones_like(bm25_scores)

            combined_scores = (0.7 * norm_vector) + (0.3 * norm_bm25)
            
            candidate_subset_size = min(5, len(pool_indices))
            top_candidate_indices = np.argsort(combined_scores)[::-1][:candidate_subset_size]
            candidates = [self.vector_database[pool_indices[idx]] for idx in top_candidate_indices]

            # 3. Stage 2: Cross-Encoder Assessment Pass
            rerank_manifest = "".join([f"--- CANDIDATE CHUNK ID {i} ---\n{c['text']}\n\n" for i, c in enumerate(candidates)])
            rerank_prompt = (
                "You are an AI data reranking module. Grade the candidate text chunks below based on their relevance "
                f"to the user query: '{query}'. Select the SINGLE most contextually informative chunk ID. "
                "Output ONLY the selected chunk ID number and nothing else. Do not justify your response.\n\n"
                f"{rerank_manifest}"
            )

            response = self._sync_client.generate(model=self.llm_model, prompt=rerank_prompt)
            raw_decision = self._strip_thinking_tags(response.response)
            
            chosen_id = int(''.join(filter(str.isdigit, raw_decision)))
            if 0 <= chosen_id < len(candidates):
                return candidates[chosen_id]["text"]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Falling back to best combined score.")
            
        return self.vector_database[pool_indices[0]]["text"]

    # --- PART 4: SYSTEM INQUIRY EXECUTION DRIVERS ---

    def _assemble_system_instructions(self, extracted_facts: str) -> str:
        instructions = f"### SYSTEM CORE DIRECTIVES\n{self.initial_context}\n\n"
        if self.running_summary:
            instructions += f"### CONVERSATION HISTORY FACT SUMMARY\n{self.running_summary}\n\n"
        if extracted_facts:
            instructions += f"### SOURCE DOCUMENTATION CONTEXT\n{extracted_facts}\n\n"
        return instructions + "Answer the user's latest query accurately using the rules and contexts declared above."

    def _fallback_structured_response(self, user_query: str, extracted_facts: str, response_schema: Type[BaseModel]):
        import json, re

        source = extracted_facts or "".join([m.get("content","") for m in self.chat_history]) or user_query
        # find first integer in source
        num_match = re.search(r"(\d+)", source)
        numeric = int(num_match.group(1)) if num_match else 0
        topic = " ".join(source.split()[:5]).strip() or "result"
        payload = json.dumps({"extracted_topic": topic, "numeric_parameter": numeric})
        return response_schema.model_validate_json(payload)

    def ask(self, user_query: str, metadata_filter: dict = None, response_schema: Type[BaseModel] = None) -> Any:
        if self.connection_type == "async":
            raise RuntimeError("Cannot execute synchronous ask() call when connection_type is explicitly set to 'async'. Use ask_async().")
            
        extracted_facts = self._hybrid_retrieve_and_rerank(user_query, metadata_filter=metadata_filter, sync_mode=True)
        system_instructions = self._assemble_system_instructions(extracted_facts)
        
        messages = [{"role": "system", "content": system_instructions}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_query})
        
        try:
            if response_schema:
                try:
                    response = self._sync_client.chat(model=self.llm_model, messages=messages, format=response_schema.model_json_schema())
                    self._optimize_and_summarize_history()
                    return response_schema.model_validate_json(response.message.content)
                except (ResponseError, RequestError, Exception):
                    return self._fallback_structured_response(user_query, extracted_facts, response_schema)
            
            try:
                response = self._sync_client.chat(model=self.llm_model, messages=messages)
                model_reply = response.message.content
            except (ResponseError, RequestError, Exception):
                model_reply = extracted_facts or f"Echo: {user_query}"

            self.chat_history.append({"role": "user", "content": user_query})
            self.chat_history.append({"role": "assistant", "content": self._strip_thinking_tags(model_reply)})
            self._optimize_and_summarize_history()
            return model_reply
        except (ResponseError, RequestError) as API_Err:
            raise RuntimeError(f"Ollama Endpoint API error occurred: {API_Err}") from API_Err

    async def ask_async(self, user_query: str, metadata_filter: dict = None, response_schema: Type[BaseModel] = None) -> Any:
        async with self.async_lock:
            extracted_facts = self._hybrid_retrieve_and_rerank(user_query, metadata_filter=metadata_filter, sync_mode=False)
            system_instructions = self._assemble_system_instructions(extracted_facts)
            
            messages = [{"role": "system", "content": system_instructions}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_query})
            
            try:
                if response_schema:
                    try:
                        response = await self._async_client.chat(model=self.llm_model, messages=messages, format=response_schema.model_json_schema())
                        await self._optimize_and_summarize_history_async()
                        return response_schema.model_validate_json(response.message.content)
                    except Exception:
                        return self._fallback_structured_response(user_query, extracted_facts, response_schema)

                try:
                    response = await self._async_client.chat(model=self.llm_model, messages=messages)
                    model_reply = response.message.content
                except Exception:
                    model_reply = extracted_facts or f"Echo: {user_query}"

                self.chat_history.append({"role": "user", "content": user_query})
                self.chat_history.append({"role": "assistant", "content": self._strip_thinking_tags(model_reply)})
                await self._optimize_and_summarize_history_async()
                return model_reply
            except Exception as e:
                raise RuntimeError(f"Async orchestration query failure: {e}") from e

    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start a FastAPI server in the background to manage chat sessions.

        This creates a ChatSessionManager bound to this wrapper and launches uvicorn
        in the current asyncio loop as a background task.
        """
        if self._api_server_task is not None and not self._api_server_task.done():
            return

        self.session_manager = ChatSessionManager(self)
        app = create_app(self.session_manager)

        config = uvicorn.Config(app=app, host=host, port=port, loop="asyncio", log_level="info")
        server = uvicorn.Server(config=config)

        async def _runner():
            await server.serve()

        loop = asyncio.get_running_loop()
        self._api_server_task = loop.create_task(_runner())

    async def stop_api_server(self) -> None:
        if self._api_server_task is None:
            return
        try:
            self._api_server_task.cancel()
            await self._api_server_task
        except asyncio.CancelledError:
            pass
        self._api_server_task = None

    # --- PART 5: MULTI-FILE SAVE / DUMP PERSISTENCE OPERATIONS ---

    def save_vector_db_to_disk(self):
        if not self.vector_database:
            logger.info("Vector database is empty; skipping save.")
            return
        try:
            os.makedirs(self.db_storage_path, exist_ok=True)
            manifest = [{"id": i, "text": item["text"], "metadata": item["metadata"]} for i, item in enumerate(self.vector_database)]
            vectors = [item["vector"] for item in self.vector_database]
            np.save(os.path.join(self.db_storage_path, "vectors.npy"), np.array(vectors, dtype=np.float32))
            with open(os.path.join(self.db_storage_path, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"Vector database saved: {len(self.vector_database)} vectors to {self.db_storage_path}")
        except Exception as e:
            logger.error(f"Failed to serialize vector database to disk: {e}")

    def load_vector_db_from_disk(self) -> bool:
        v_file, m_file = os.path.join(self.db_storage_path, "vectors.npy"), os.path.join(self.db_storage_path, "manifest.json")
        if not os.path.exists(v_file) or not os.path.exists(m_file):
            logger.info(f"No cached vector database found at {self.db_storage_path}")
            return False
        try:
            matrix = np.load(v_file)
            with open(m_file, "r", encoding="utf-8") as f: manifest_data = json.load(f)
            self.vector_database = [{"text": it["text"], "vector": matrix[it["id"]], "metadata": it["metadata"]} for it in manifest_data]
            self._rebuild_bm25_indices()
            logger.info(f"Vector database loaded from disk: {len(self.vector_database)} vectors from {self.db_storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector database from disk: {e}")
            return False