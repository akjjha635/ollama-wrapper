# ollama-wrapper

A lightweight local RAG and orchestration framework with:

- Hybrid retrieval (dense + BM25)
- Optional async query path
- Structured output validation via Pydantic schemas
- Local vector persistence using NumPy + JSON
- Optional FastAPI chat session server

The package is designed to run locally with Ollama, while still supporting CI/dev workflows where Ollama is unavailable.

## Installation

```bash
git clone https://github.com/akjjha635/ollama-wrapper.git
cd ollama-wrapper
pip install -e .
```

Optional dev dependencies:

```bash
pip install -e .[dev]
```

## Ollama Runtime Behavior

- By default, wrapper initialization tries to auto-check/start Ollama.
- If Ollama is missing/unreachable, initialization logs a warning and continues.
- To enforce fail-fast startup (useful in strict local environments), set:

```bash
export OLLAMA_WRAPPER_STRICT_STARTUP=true
```

You can always disable auto-start checks per instance:

```python
from ollama_wrapper import OllamaWrapper

wrapper = OllamaWrapper(auto_ensure_ollama=False)
```

## Quick Start (Structured Output)

```python
from typing import List
from pydantic import BaseModel, Field
from ollama_wrapper import OllamaWrapper


class TechnicalExtraction(BaseModel):
    subject_matter: str = Field(description="Primary topic")
    key_metrics: List[str] = Field(description="Extracted metrics")
    confidence_rating: float = Field(description="0.0-1.0 confidence")


wrapper = OllamaWrapper(
    connection_type="sync",
    llm_model="deepseek-r1:1.5b",
    embed_model="nomic-embed-text",
    db_storage_path="./my_vault",
)

wrapper.load_prompt_from_string("You are an analytical extractor.", optimize=False)
wrapper.ingest_semantic_document(
    "Project Orion stabilized at 440.2 MHz with zero leaks.",
    metadata={"vault": "secure_notes"},
)

result = wrapper.ask(
    user_query="Extract the technical signal.",
    metadata_filter={"vault": "secure_notes"},
    response_schema=TechnicalExtraction,
)

print(result.model_dump())
```

## Development Chat Server

Start local API server (real Ollama-backed wrapper):

```bash
python scripts/start_chat_server.py --host 127.0.0.1 --port 8000
```

Start local API server with dummy echo backend (no Ollama needed):

```bash
python scripts/start_chat_server.py --host 127.0.0.1 --port 8000 --dummy-wrapper
```

Example API flow:

```bash
curl -X POST http://127.0.0.1:8000/session \
    -H "Content-Type: application/json" \
    -d '{"system_prompt":"You are helpful."}'
```

```bash
curl -X POST http://127.0.0.1:8000/session/<SESSION_ID>/message \
    -H "Content-Type: application/json" \
    -d '{"message":"hello"}'
```

## Benchmark Runner

```bash
python scripts/run_benchmark.py \
    --base-url http://127.0.0.1:8000 \
    --iterations 20 \
    --message "hello"
```

Compare two deployments (example: linear vs faiss):

```bash
# Terminal 1
python scripts/start_chat_server.py --port 8000 --retrieval-backend linear

# Terminal 2
python scripts/start_chat_server.py --port 8001 --retrieval-backend faiss

# Terminal 3
python scripts/run_benchmark.py \
    --compare \
    --base-url http://127.0.0.1:8000 \
    --candidate-base-url http://127.0.0.1:8001 \
    --baseline-label linear \
    --candidate-label faiss \
    --iterations 20
```

Create markdown summary from comparison output:

```bash
python scripts/run_benchmark.py --compare --base-url http://127.0.0.1:8000 --candidate-base-url http://127.0.0.1:8001 --output comparison.json
python scripts/benchmark_report_to_markdown.py --input comparison.json --output benchmark_summary.md
```

## Retrieval Quality Regression

```bash
python scripts/run_quality_regression.py \
    --dataset data/quality_samples.json \
    --min-query-hit 0.40 \
    --min-groundedness 0.50 \
    --min-coverage 0.05
```

## Repository Layout

```text
src/ollama_wrapper/
    core.py
    api_server.py
    control/
    eval/
    llm/
    observability/
    optimization/
    orchestration/
    retrieval/
tests/
scripts/
```

## Additional Documentation

- Architecture: ARCHITECTURE.md
- Troubleshooting: TROUBLESHOOTING.md
- Public usage review: PUBLIC_USAGE_REVIEW.md

## License

MIT. See LICENSE.
