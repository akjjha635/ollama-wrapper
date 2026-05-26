# Architecture Overview

## System Design

OllamaWrapper is a lightweight, single-process RAG (Retrieval-Augmented Generation) engine optimized for consumer GPUs. It combines semantic search, lexical ranking, and LLM-based reranking into a unified query pipeline.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    OllamaWrapper (Singleton)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. INGESTION LAYER                                         │
│     ├─ ingest_semantic_document()  [Topic-aware chunking]   │
│     └─ ingest_knowledge_document() [Sliding window chunks]  │
│           ↓                                                  │
│     [Embedding Client] → [NumPy Vector Store]              │
│           ↓                                                  │
│     [BM25 Index Rebuild]                                    │
│                                                               │
│  2. RETRIEVAL LAYER                                         │
│     ├─ _hybrid_retrieve_and_rerank() [Sync path]           │
│     └─ _hybrid_retrieve_and_rerank_async() [Async path]    │
│           ↓                                                  │
│     ┌─────────────────────────────────────────┐             │
│     │  Dense Embedding Score (Cosine)        │             │
│     │  + BM25 Lexical Score (Term Frequency) │ → Combined  │
│     │                                          │   Score    │
│     └─────────────────────────────────────────┘             │
│           ↓                                                  │
│     [Top-K Candidates Selected]                             │
│           ↓                                                  │
│  3. RERANKING LAYER                                         │
│     └─ Cross-Encoder LLM Pass                               │
│           ↓                                                  │
│     [Single Best Chunk Selected]                            │
│                                                               │
│  4. QUERY EXECUTION LAYER                                   │
│     ├─ ask() [Synchronous]                                  │
│     └─ ask_async() [Asynchronous]                           │
│           ↓                                                  │
│     [System Prompt + History + Retrieved Context]           │
│           ↓                                                  │
│     [LLM Response] → [Validate Against Schema] (optional)   │
│           ↓                                                  │
│     [Append to Chat History] → [Trigger Memory Compaction]  │
│                                                               │
│  5. MEMORY MANAGEMENT                                       │
│     └─ _optimize_and_summarize_history() [Auto-trigger]    │
│           ↓                                                  │
│     [Compress N oldest turns into bullet summary]           │
│           ↓                                                  │
│     [Keep last 2 turns verbatim + rolling summary]          │
│                                                               │
│  6. PERSISTENCE LAYER                                       │
│     ├─ save_vector_db_to_disk() [Numpy .npy + JSON]        │
│     └─ load_vector_db_from_disk() [Instant rehydration]    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Query Execution

### Synchronous Path (ask)
```
user_query
    ↓
[_hybrid_retrieve_and_rerank]
    ├─ Pool indices by metadata filter
    ├─ Embed query (sync client)
    ├─ Compute vector scores (cosine similarity)
    ├─ Compute BM25 scores (term frequency-inverse document frequency)
    ├─ Normalize both to [0,1] range
    ├─ Blend: 0.7 * vector + 0.3 * BM25
    ├─ Select top-K candidates
    ├─ LLM rerank: "Pick the best chunk for this query"
    └─ Return selected chunk text
    ↓
[_assemble_system_instructions]
    ├─ Base system prompt
    ├─ Conversation history summary (if exists)
    └─ Retrieved context
    ↓
[LLM chat call]
    ├─ Build message array: [system, history..., user]
    ├─ If response_schema: enforce JSON schema validation
    └─ Return response
    ↓
[Append to chat_history]
    ├─ Add user message
    ├─ Add assistant response
    ├─ Trigger memory compaction if len(history) > max_active_turns
    └─ Return to user
```

### Asynchronous Path (ask_async)
- Same logic as sync path
- Uses `_hybrid_retrieve_and_rerank_async()` instead (awaitable embedding + rerank)
- Protected by `asyncio.Lock` to prevent concurrent history mutations
- Allows multiple queries to run concurrently without blocking

---

## Semantic Chunking Algorithm

**Goal**: Split documents at topic boundaries, not token limits.

```
Input: raw_document (e.g., 500 words)
    ↓
[Regex split on sentence boundaries]
    → Sentences: ["Sent1", "Sent2", ..., "SentN"]
    ↓
[Embed each sentence with embed_model]
    → Vectors: [v1, v2, ..., vN]
    ↓
[Compute pairwise distances (1 - dot_product)]
    → Distances: [d1, d2, ..., d(N-1)]
    ↓
[Find threshold via percentile]
    → threshold = np.percentile(distances, threshold_percentile=65.0)
    ↓
[Iterate through distances]
    IF distance > threshold:
        → End current chunk, start new chunk
    ELSE:
        → Append sentence to current chunk
    ↓
Output: Chunks: [Chunk1, Chunk2, ..., ChunkK]
```

**Why This Works**: Topics have consistent semantic distance; abrupt topic shifts show high distance spikes.

---

## Hybrid Search Scoring

The system combines two complementary signals:

### 1. Dense Semantic Score (Vector Similarity)
```
Query: "What is the learning rate?"
    ↓
[Embed query] → query_vector (384-dim)
[Embed each document chunk] → chunk_vectors
    ↓
[Cosine similarity] → scores in [-1, 1]
Normalize to [0, 1]
```
**Strength**: Captures meaning, robust to paraphrasing  
**Weakness**: Slow (embedding cost), insensitive to exact keywords

### 2. Lexical Score (BM25)
```
Query: "What is the learning rate?"
Query terms: ["what", "is", "the", "learning", "rate"]
    ↓
For each chunk:
    TF = term frequency in chunk
    IDF = log((N - df + 0.5) / (df + 0.5))
    BM25 = Σ(IDF * (TF * (k1+1)) / (TF + k1*(1 - b + b*doc_len/avg_doc_len)))
Normalize to [0, 1]
```
**Strength**: Fast, exact keyword matches, industry standard  
**Weakness**: Blind to synonyms, fails on paraphrasing

### 3. Blended Score
```
final_score = 0.7 * semantic_score + 0.3 * bm25_score
```
**Why 70/30?** Semantic is more informative; BM25 is a tiebreaker for exact matches.

---

## Memory Compaction Strategy

**Problem**: Long conversations blow up VRAM and hurt LLM latency.

**Solution**: Auto-compress old turns into a bullet summary.

```
Chat history: [
  {"role": "user", "content": "Turn 1..."},
  {"role": "assistant", "content": "Response 1..."},
  {"role": "user", "content": "Turn 2..."},
  {"role": "assistant", "content": "Response 2..."},
  {"role": "user", "content": "Turn 3..."},
  {"role": "assistant", "content": "Response 3..."},
]

max_active_turns = 2  (keep last 2 turns verbatim)

[Trigger]
    ↓
[Compress turns 1-2 using LLM]
    "Merge this history into a bullet summary of established facts..."
    ↓
running_summary = "• User asked about X. System explained Y. • User clarified Z..."

[Keep turns 3 (still live)]

Result: messages = [
  {"role": "system", "content": "### CONVERSATION HISTORY FACT SUMMARY\n• User asked..."},
  {"role": "user", "content": "Turn 3..."},
  {"role": "assistant", "content": "Response 3..."},
]
```

**Why This Works**: Preserves critical context (facts, constraints, preferences) without token bloat.

---

## Singleton Pattern & State Management

```python
OllamaWrapper._instance = None

wrapper1 = OllamaWrapper(connection_type="sync")  # Initialize
wrapper2 = OllamaWrapper(connection_type="async")  # Returns wrapper1 (same object)

# All configuration is set on first instantiation; subsequent calls are no-ops
wrapper1.vector_database  # Shared across all references
wrapper1.chat_history     # Shared state
```

**Trade-off**:
- ✅ Simple: Users don't manage connection lifecycle
- ❌ Limited: Can't run multiple independent instances (single-process constraint)

---

## Persistence Format

### Vector Storage (`vectors.npy`)
- NumPy binary format (C-contiguous float32)
- One row per chunk embedding (384-dim for nomic-embed-text)
- Fast load time (milliseconds for 1000+ vectors)

### Metadata (`manifest.json`)
```json
[
  {"id": 0, "text": "Chunk 1...", "metadata": {"source": "doc_a", "year": 2026}},
  {"id": 1, "text": "Chunk 2...", "metadata": {"source": "doc_b", "year": 2026}},
]
```

**Why this split?**
- Binary vectors are efficient to load/compute
- JSON metadata is human-readable and queryable
- On reload: `manifest.json[i]["id"]` indexes into `vectors.npy[i]`

---

## Startup & Connection Flow

```
OllamaWrapper(
    connection_type="sync",
    ip="localhost",
    port=11434,
    auto_start_local=True,        # NEW: allow auto-spawn
    auto_pull_embed_model=True    # NEW: allow model pull
)
    ↓
[ensure_ollama_is_running()]
    ├─ Is endpoint reachable? (sync_client.list())
    ├─ If NO:
    │   ├─ Is auto_start_local=True? If False → raise RuntimeError
    │   ├─ Is local host? If remote → raise RuntimeError
    │   └─ Spawn: subprocess.Popen(["ollama", "serve"])
    │       Sleep 3s → retry list()
    ├─ Is embedding model present?
    └─ If auto_pull_embed_model=True → ollama.pull("nomic-embed-text")
    ↓
Connected!
```

**Why these flags matter:**
- Remote deployments (cloud Ollama) don't want subprocess spawning
- CI/CD environments may have Ollama mocked; no auto-pull needed
- Users can now control startup behavior explicitly

---

## Exception Handling & Fallback

```
ask(query)
    ↓
[_hybrid_retrieve_and_rerank]
    ├─ TRY: Embed → Score → Rerank → LLM decide
    └─ EXCEPT: 
        ├─ Log warning
        └─ Return best_fallback_idx (highest combined score before rerank failed)
    ↓
[LLM chat call]
    ├─ TRY: Call LLM with retrieved context
    └─ EXCEPT: Raise RuntimeError (schema validation, API failure)
```

**Resilience**: Retrieval failures gracefully degrade to highest-ranked candidate. LLM failures propagate (user handles retries).

---

## Thread Safety & Async

- **Sync mode**: Direct client calls, no threading overhead
- **Async mode**: AsyncClient + asyncio.Lock on history mutations
  - Multiple queries can execute concurrently
  - Lock prevents race conditions on chat_history append
  - Each query waits its turn to mutate state

---

## Performance Characteristics

| Operation | Latency | Depends On |
|-----------|---------|-----------|
| Ingest 1 document (1KB) | 50-200ms | Embed model latency |
| Embed query | 50-100ms | Embed model, query length |
| Vector similarity (1000 docs) | 5-10ms | NumPy (C-accelerated) |
| BM25 scoring (1000 docs) | 10-50ms | Query term count |
| LLM rerank (top-5 candidates) | 500ms-2s | LLM model, candidate lengths |
| LLM chat response | 1s-10s+ | LLM model, context length |
| Memory compaction | 500ms-3s | History size, LLM model |
| Save vectors to disk (1000 docs) | 50-200ms | Disk I/O, vector count |
| Load vectors from disk (1000 docs) | 10-100ms | Disk I/O, vector count |

**Bottleneck**: LLM inference (rerank + chat). Embedding model is 2nd bottleneck.

---

## Future Extensibility

Current architecture supports:
- ✅ Different embedding models (switch `embed_model` kwarg)
- ✅ Different LLM models (switch `llm_model` kwarg)
- ✅ Custom metadata filters
- ✅ Custom BM25 parameters (k1, b)

Would require refactoring:
- ❌ Multiple independent instances (singleton constraint)
- ❌ Distributed deployments (no multi-process support)
- ❌ Streaming responses (all-or-nothing response model)
- ❌ Custom chunking strategies (hardcoded semantic + sliding window)
