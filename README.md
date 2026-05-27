# Ollama-Wrapper 🚀

An ultra-lightweight local RAG (Retrieval-Augmented Generation) framework and memory management engine with a minimal Python dependency footprint, optimized explicitly for consumer-grade hardware (like NVIDIA RTX 2060 6GB). 

Built as a lightweight instance-based Python engine, this library replaces heavy enterprise framework overhead (LangChain, LlamaIndex) and external vector database infrastructure with raw, C-accelerated NumPy matrix math, smart contextual pruning, and local multi-file persistence.

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
│       ├── __init__.py      # Exposes the main Wrapper class
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

## Development Chat Server

Use this script when you want a local chat session API for testing integrations from other libraries/services.

```bash
python scripts/start_chat_server.py --host 127.0.0.1 --port 8000
```

For fast integration tests when Ollama is not available, run a local echo backend:

```bash
python scripts/start_chat_server.py --host 127.0.0.1 --port 8000 --dummy-wrapper
```

Example API flow after startup:

```bash
curl -X POST http://127.0.0.1:8000/session -H "Content-Type: application/json" -d '{"system_prompt":"You are helpful."}'
```

```bash
curl -X POST http://127.0.0.1:8000/session/<SESSION_ID>/message -H "Content-Type: application/json" -d '{"message":"hello"}'
```

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
    # 2. Instantiate a Wrapper instance (Optimized for 6GB VRAM)
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

Lifecycle note: each `OllamaWrapper(...)` call creates an independent instance. If you started the API server on an instance, call `close(stop_server=True)` during teardown.

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

## Project Roadmap

See [ROADMAP.md](ROADMAP.md) for the v0.2 layered architecture plan, migration phases, and success metrics.

## Benchmark Runner

After starting the chat server, run a quick latency/token benchmark:

```bash
python scripts/run_benchmark.py --base-url http://127.0.0.1:8000 --iterations 20 --message "hello"
```

Optional: include message options and save the JSON report.

```bash
python scripts/run_benchmark.py --base-url http://127.0.0.1:8000 --iterations 20 --options-json '{"budget_mode":"warn"}' --output benchmark_report.json
```

Comparative benchmark mode (for example, linear vs faiss):

```bash
# Terminal 1
python scripts/start_chat_server.py --port 8000 --retrieval-backend linear

# Terminal 2
python scripts/start_chat_server.py --port 8001 --retrieval-backend faiss

# Terminal 3
python scripts/run_benchmark.py --compare --base-url http://127.0.0.1:8000 --candidate-base-url http://127.0.0.1:8001 --baseline-label linear --candidate-label faiss --iterations 20
```

Generate a markdown release-note table from comparison JSON:

```bash
python scripts/run_benchmark.py --compare --base-url http://127.0.0.1:8000 --candidate-base-url http://127.0.0.1:8001 --output comparison.json
python scripts/benchmark_report_to_markdown.py --input comparison.json --output benchmark_summary.md
```

Run retrieval quality regression gates from a JSON dataset:

```bash
python scripts/run_quality_regression.py --dataset data/quality_samples.json --min-query-hit 0.40 --min-groundedness 0.50 --min-coverage 0.05
```
