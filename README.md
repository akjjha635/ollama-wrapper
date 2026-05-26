# Ollama-Wrapper 🚀

An ultra-lightweight local RAG (Retrieval-Augmented Generation) framework and memory management engine with a minimal Python dependency footprint, optimized explicitly for consumer-grade hardware (like NVIDIA RTX 2060 6GB). 

Built entirely as a Python Singleton, this library replaces heavy enterprise framework overhead (LangChain, LlamaIndex) and external vector database infrastructure with raw, C-accelerated NumPy matrix math, smart contextual pruning, and local multi-file persistence.

---

## 🌟 Core Value Proposition & Architecture

Most local RAG setups fall off a performance cliff on consumer hardware because they lack context boundaries and run heavy Docker-based database containers. This framework shifts all the structural heavy lifting onto CPU system memory while preserving precious GPU VRAM for raw model execution.



### 🔑 Key Engineering Features
* **Hybrid Dual-Engine Search:** Combines dense semantic vector embeddings (`nomic-embed-text`) with standard lexical keyword matching (**BM25**) through weighted score fusion.
* **Two-Stage Reranking:** Implements an internal Cross-Encoder grading pass. The system filters candidate chunks using the LLM to eliminate context noise before passing the final data to the prompt window.
* **Rolling Memory Compaction:** Automatically intercepts conversation history. When your active turn limits are crossed, it uses a background loop to compress older dialogue sequences into a dense bulleted list of facts, maintaining an immutable VRAM footprint.
* **Semantic Distance Chunking:** Splits incoming research documents dynamically based on semantic distance shifts between sentence transitions, ensuring text fragments remain contextually whole.
* **Zero-Infra Disk Persistence:** Serializes multi-dimensional matrices natively into binary (`.npy`) format and structures metadata in flat `.json` layouts for instant rehydration upon boot.
* **Strict Schema Guardrails:** Built-in validation allows you to pass Pydantic `BaseModel` schemas directly into inference targets, guaranteeing strict, parseable JSON payloads.

---

## 📁 Repository Structure

```text
ollama-wrapper/
├── src/
│   └── ollama_wrapper/
│       ├── __init__.py      # Exposes the main Singleton Wrapper
│       └── core.py          # Complete optimized core engine class
├── tests/
│   └── test_core.py         # Concurrent async and sync verification tests
├── pyproject.toml           # Modern PEP 517 build configuration
└── README.md                # Project documentation
```
## ⚙️ Installation & Prerequisites
Ensure you have Ollama installed and running locally on your machine.
1. Clone & Install Dependencies
```
git clone [https://github.com/akjjha635/ollama-wrapper.git](https://github.com/akjjha635/ollama-wrapper.git)
cd ollama-wrapper
pip install numpy pydantic ollama
```
2. Pull Required Model Weights

The framework can auto-pull your embedding model on initialization for local hosts (`auto_pull_embed_model=True`, default), but you can also download models manually via terminal for a seamless setup:
```
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
```

##🚀 Quick Start Guide

This single script demonstrates document ingestion with semantic topic-splitting, metadata filtering, a hybrid search query, and a strictly structured Pydantic JSON extraction pass.
```
import asyncio
from pydantic import BaseModel, Field
from typing import List
from ollama-wrapper import OllamaWrapper

# 1. Define your target structured output schema
class TechnicalExtraction(BaseModel):
    subject_matter: str = Field(description="The primary engine architecture or project name.")
    key_metrics: List[str] = Field(description="List of precise hardware parameters or numeric configurations.")
    confidence_rating: float = Field(description="Calculated extraction metric between 0.0 and 1.0.")

async def main():
    # 2. Instantiate the Singleton Wrapper (Optimized for 6GB VRAM)
    ai_system = OllamaWrapper(
        connection_type="sync",
        llm_model="deepseek-r1:1.5b",
        embed_model="nomic-embed-text",
        max_active_turns=2,             # Aggressive memory compaction trigger
        db_storage_path="./my_vault"
    )
    
    # Configure base global rules
    ai_system.load_prompt_from_string("You are an analytical data extractor.", optimize=False)

    # 3. Ingest documents (The chunker splits these contextually on topic shifts)
    raw_document = (
        "Project Orion hardware validation complete. The internal hyper-frequency engine array "
        "stabilized at exactly 440.2 Megahertz with zero terminal plasma leaks detected. "
        "On an unrelated administrative note, the corporate office confirmed that financial budget audits "
        "must be completed before September 15th to prevent processing penalties."
    )
    ai_system.ingest_semantic_document(raw_document, threshold_percentile=50.0, metadata={"vault": "secure_notes"})
    
    # Optional: Save compiled matrix vectors directly to disk
    ai_system.save_vector_db_to_disk()

    # 4. Dispatch query using Hybrid Search, Cross-Encoder Reranking, and Schema Enforcement
    query = "What are the configuration constraints and exact frequency metrics for Project Orion?"
    
    structured_json = ai_system.ask(
        user_query=query,
        metadata_filter={"vault": "secure_notes"},
        response_schema=TechnicalExtraction
    )

    # 5. Review cleanly parsed native Python object properties
    print("\n[Result] Output successfully extracted:")
    print(f"Subject: {structured_json.subject_matter}")
    print(f"Metrics: {structured_json.key_metrics}")
    print(f"Confidence: {structured_json.confidence_rating}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
