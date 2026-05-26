# Troubleshooting Guide

## Installation & Setup

### `ModuleNotFoundError: No module named 'ollama'`
```bash
# Solution: Install the package with dependencies
pip install -e .
# or explicitly:
pip install ollama pydantic numpy
```

### `[System] Ollama server not detected. Instantiating background daemon...` hangs indefinitely
```python
# Problem: Ollama process spawn fails silently, system waits 3s then retries
# Solution: Start Ollama manually FIRST

# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run your script
python your_script.py
```

### `RuntimeError: Ollama endpoint is unreachable and auto_start_local=False`
```python
# Problem: You set a remote endpoint but auto_start_local=False
# Solution: Either start Ollama on remote host, or enable auto_start:

# Option 1: Remote host already running
wrapper = OllamaWrapper(
    ip="192.168.1.100",
    port=11434,
    auto_start_local=False  # ✓ Remote is already up
)

# Option 2: Local host, allow auto-start
wrapper = OllamaWrapper(
    ip="localhost",
    auto_start_local=True  # ✓ Subprocess will spawn if needed
)
```

### `[System] 'nomic-embed-text' missing. Pulling dependencies...` (then hangs)
```bash
# Problem: Model pull is slow or network is poor
# Solution: Pull model manually FIRST (once)

ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b  # or your chosen LLM

# Then disable auto-pull in code:
wrapper = OllamaWrapper(
    auto_pull_embed_model=False  # ✓ Skip redundant pulls
)
```

---

## Common Runtime Issues

### `Warning: Reranker or search exception caught (...). Falling back to primary index match.`
```python
# Problem: LLM reranking failed (timeout, API error, malformed response)
# Impact: Uses highest-scored chunk anyway (graceful degradation)
# Solution: 

# 1. Check Ollama logs:
ollama logs

# 2. Verify LLM is responsive:
from ollama import Client
client = Client()
response = client.generate(model="deepseek-r1:1.5b", prompt="Say hello")
print(response)

# 3. If timeouts, increase patience:
wrapper._sync_client.timeout = 60  # default is ~30s
```

### `ValueError: Configured chunk_size (100) must be strictly greater than your overlap parameter (100).`
```python
# Problem: chunk_size == overlap (or chunk_size < overlap)
# Solution: overlap must be < chunk_size for sliding window

wrapper.ingest_knowledge_document(
    text,
    chunk_size=400,  # ✓ Size of each chunk
    overlap=50       # ✓ Words overlapped between chunks (< chunk_size)
)
```

### `Empty string content passed to ingest pipeline. Skipping iteration.`
```python
# Problem: Whitespace-only or empty text was ingested
# Impact: Chunk is silently skipped (no error)
# Solution: Pre-filter input:

if text.strip():
    wrapper.ingest_knowledge_document(text)
else:
    print("Skipping empty document")
```

### Chat history grows indefinitely; model responses become slow
```python
# Problem: Memory compaction isn't triggering
# Cause: max_active_turns threshold not exceeded
# Solution:

wrapper = OllamaWrapper(
    max_active_turns=2  # Trigger compaction after 2 turns (4 messages)
)

# Monitor compaction:
print(f"History length: {len(wrapper.chat_history)}")
print(f"Summary: {wrapper.running_summary[:100]}...")
```

### `AssertionError: response.model_validate_json() failed`
```python
# Problem: LLM returned non-JSON or invalid schema
# Solution:

from pydantic import BaseModel, Field

class MySchema(BaseModel):
    name: str = Field(description="Required")
    count: int = Field(description="Required integer")

# Make prompt very explicit:
wrapper.load_prompt_from_string(
    "You MUST respond ONLY with valid JSON matching this schema: "
    '{"name": "...", "count": 123}',
    optimize=False
)

# Add pre-validation:
result = wrapper.ask(query, response_schema=MySchema)
assert isinstance(result, MySchema)
assert isinstance(result.count, int)
```

---

## Performance Issues

### Ingestion is slow (5+ seconds per document)
```python
# Problem: Embedding API calls are sequential
# Root cause: Model inference latency (~50-100ms per embedding)
# Solution:

# 1. Pre-batch sentences (manual):
sentences = [s1, s2, s3, s4, s5]  # 5 sentences = 5 * 100ms = ~500ms

# 2. Use smaller chunk_size:
wrapper.ingest_knowledge_document(text, chunk_size=100)  # More smaller chunks

# 3. Use faster embedding model (if available):
wrapper = OllamaWrapper(embed_model="all-minilm:22m")  # Smaller model
```

### Queries return results in 5+ seconds
```python
# Problem: Three sequential operations: embed query, score, rerank
# Root causes:
# 1. Embedding latency: ~100ms
# 2. Vector scoring: ~10ms (but fast)
# 3. LLM rerank: ~1-2s (slow)
# 4. LLM chat: ~3-5s (slowest)

# Solutions:
# 1. Use faster embedding model:
wrapper = OllamaWrapper(embed_model="all-minilm:22m")  # 20ms embedding

# 2. Skip reranking (use top-1 directly):
# [Not currently supported; would require code change]

# 3. Use faster LLM model:
wrapper = OllamaWrapper(llm_model="neural-chat:7b")  # Faster than deepseek

# 4. Reduce document pool (metadata filter):
wrapper.ask(query, metadata_filter={"year": 2026})  # Fewer candidates to rerank
```

### Memory/VRAM usage grows after many ingestions
```python
# Problem: vector_database list keeps growing
# Solution:

# Check size:
print(f"Vector count: {len(wrapper.vector_database)}")
print(f"Approx VRAM: {len(wrapper.vector_database) * 384 * 4 / 1e9:.2f}GB")

# Option 1: Reset and reload from disk
wrapper = OllamaWrapper()
wrapper.load_vector_db_from_disk()  # Start fresh

# Option 2: Clear old vectors (no built-in method yet)
wrapper.vector_database = []
wrapper._rebuild_bm25_indices()
```

### Top-K retrieval is returning identical chunks
```python
# Problem: top_k parameter isn't respected as expected
# Note: top_k affects candidate_subset_size before rerank
# Final result is SINGLE best chunk (after LLM rerank)

# If you need multiple results:
# [Not currently supported; would require API change]

# Workaround: Call ask() multiple times with metadata filters
results = []
for source in sources:
    result = wrapper.ask(query, metadata_filter={"source": source})
    results.append(result)
```

---

## Debugging & Diagnostics

### Enable verbose output
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ollama-wrapper")

wrapper = OllamaWrapper()
# Now you'll see detailed logs (once logging is integrated)
```

### Inspect retrieval behavior
```python
from ollama-wrapper import OllamaWrapper

wrapper = OllamaWrapper(connection_type="sync")
wrapper.ingest_knowledge_document("Document A about topic X.")
wrapper.ingest_knowledge_document("Document B about topic Y.")

query = "Tell me about topic X"

# See what gets retrieved:
retrieved = wrapper._hybrid_retrieve_and_rerank(query)
print(f"Retrieved chunk:\n{retrieved}")

# Manually score to debug:
from numpy.linalg import norm
import numpy as np

res = wrapper._sync_client.embed(model=wrapper.embed_model, input=query)
q_vec = np.array(res.embeddings[0], dtype=np.float32)
q_norm = norm(q_vec)
if q_norm > 0:
    q_vec /= q_norm

for i, doc in enumerate(wrapper.vector_database):
    score = np.dot(doc["vector"], q_vec)
    print(f"Chunk {i}: {score:.3f} → {doc['text'][:50]}...")
```

### Check embeddings quality
```python
# Verify embedding model is working:
from ollama import Client

client = Client()
res = client.embed(model="nomic-embed-text", input="Hello world")

print(f"Embedding shape: {len(res.embeddings[0])}")
print(f"Embedding sample: {res.embeddings[0][:5]}")
print(f"Norm: {np.linalg.norm(res.embeddings[0]):.3f}")
```

---

## VRAM Optimization

### Current system setup (RTX 2060 6GB target)
```
Embedding model (nomic-embed-text):     ~500MB
LLM model (deepseek-r1:1.5b):           ~3GB
Vector database (1000 docs * 384-dim):  ~1.5MB
Chat history (50 turns, avg 200 chars): ~50KB
Running summary:                        ~10KB

Total: ~3.5GB (comfortable on 6GB)
```

### Reduce VRAM usage
```python
# 1. Use smaller LLM:
wrapper = OllamaWrapper(llm_model="neural-chat:7b")  # ~2GB

# 2. Use smaller embedding model:
wrapper = OllamaWrapper(embed_model="all-minilm:22m")  # ~100MB

# 3. Limit vector database size:
# (No built-in method; prune manually)
wrapper.vector_database = wrapper.vector_database[:500]  # Keep first 500 only

# 4. Reduce max_active_turns (triggers compaction sooner):
wrapper = OllamaWrapper(max_active_turns=2)  # Compress after 2 turns instead of 4
```

---

## Integration Testing

### Test without a live Ollama instance
```python
# Mock Ollama responses for CI/CD:
from unittest.mock import MagicMock, patch

with patch('ollama.Client') as MockClient:
    mock_client = MagicMock()
    mock_client.embed.return_value = MagicMock(
        embeddings=[[0.1] * 384]
    )
    mock_client.chat.return_value = MagicMock(
        message=MagicMock(content='{"key": "value"}')
    )
    MockClient.return_value = mock_client
    
    # Now test works without Ollama running
    wrapper = OllamaWrapper()
    # ... test code ...
```

---

## Getting Help

1. **Check Ollama logs**: `ollama logs` (macOS/Linux) or Event Viewer (Windows)
2. **Verify endpoint**: `curl http://localhost:11434/api/tags`
3. **Test embedding**: Manual embed call (see Diagnostics section)
4. **Check repo issues**: GitHub Issues (if available)
5. **Community**: Local LLM forums (Discord, Reddit r/LocalLLMs)
