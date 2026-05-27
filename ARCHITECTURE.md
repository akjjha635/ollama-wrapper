# Architecture Overview

## System Design

OllamaWrapper is a lightweight, single-process RAG (Retrieval-Augmented Generation) engine optimized for consumer GPUs. It combines semantic search, lexical ranking, and LLM-based reranking into a unified query pipeline.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    OllamaWrapper (Instance)                 │
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

## Layered API Flow (Session Endpoints)

For `POST /session/{session_id}/message`, the execution flow is now policy-driven:

```
Incoming request
        ↓
[Rate limiter]
        ├─ allow → continue
        └─ reject → HTTP 429
        ↓
[Session-aware planner]
        └─ infers query_type and summary inclusion hints
        ↓
[Routing policy]
        └─ selects provider/model + route_reason
        ↓
[Governance guardrails]
        ├─ allow → continue
        └─ reject → HTTP 400
        ↓
[Budget policy]
        ├─ warn      → continue
        ├─ truncate  → continue with adjusted query
        └─ reject    → HTTP 400
        ↓
[Context policy + optimization]
        └─ diversity-aware selection, query-type calibration, confidence scoring
        ↓
[Provider execution]
        └─ sync/async chat (streaming contract supported)
        ↓
[Observability]
        └─ trace + metrics recording
        ↓
Structured response payload
```

For `POST /session/{session_id}/dry-run`, the same route/planner/guardrails/budget/optimization decisions are computed without invoking provider generation.

---

## API Response Schema Highlights

### Message Response (`POST /session/{session_id}/message`)

Primary fields:
- `reply`: assistant text
- `provider`, `model`: selected backend route
- `usage`: normalized token usage payload (`input_tokens`, `output_tokens`, `total_tokens`, `raw`)
- `route_reason`: short route reason string
- `route_explanation`: standardized route details
    - `provider`
    - `model`
    - `reason`
    - `strategy` (currently `policy-routing-v1`)
    - `query_type`
    - `planning_reason`
- `budget_decision`: budget decision object
    - `status`, `action`, `reason`
    - `max_input_tokens`, `max_output_tokens`
    - `estimated_input_tokens`, `effective_input_tokens`
- `guardrails`: governance decision object
    - `status`, `action`, `reason`, `details`
- `session_plan`: planner output
    - `query_type`, `session_turn_count`, `include_summary`, `planning_reason`

### Dry-Run Response (`POST /session/{session_id}/dry-run`)

Primary fields:
- `dry_run`: always `true`
- `provider`, `model`, `route_reason`, `route_explanation`
- `session_plan`, `guardrails`, `budget`, `context_plan`
- `optimization`
    - `candidate_count`, `selected_count`
    - `token_estimate`, `token_budget`
    - `score_weights`, `selected_indices`
    - `confidence_scores`, `avg_confidence`, `max_confidence`

This structure makes dry-run output directly actionable for routing/cost debugging and release validation.

### Minimal OpenAPI-Style Schemas

```yaml
openapi: 3.0.3
paths:
    /session/{session_id}/message:
        post:
            summary: Send a chat message in an existing session
            responses:
                "200":
                    description: Successful assistant response
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/MessageResponse"
                "400":
                    description: Budget/guardrails rejection
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/ErrorResponse"
                "404":
                    description: Session not found
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/ErrorResponse"
                "429":
                    description: Rate limit exceeded
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/ErrorResponse"

    /session/{session_id}/dry-run:
        post:
            summary: Preview route/policy/optimization decisions without generation
            responses:
                "200":
                    description: Successful dry-run preview
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/DryRunResponse"
                "404":
                    description: Session not found
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/ErrorResponse"
                "429":
                    description: Rate limit exceeded
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/ErrorResponse"

components:
    schemas:
        MessageResponse:
            type: object
            required: [reply, provider, model]
            properties:
                reply: {type: string}
                provider: {type: string}
                model: {type: string}
                usage:
                    type: object
                    properties:
                        input_tokens: {type: integer, nullable: true}
                        output_tokens: {type: integer, nullable: true}
                        total_tokens: {type: integer, nullable: true}
                        raw: {type: object, additionalProperties: true}
                route_reason: {type: string}
                route_explanation:
                    type: object
                    properties:
                        provider: {type: string}
                        model: {type: string}
                        reason: {type: string}
                        strategy: {type: string}
                        query_type: {type: string}
                        planning_reason: {type: string}
                budget_decision: {$ref: "#/components/schemas/BudgetDecision"}
                guardrails: {$ref: "#/components/schemas/GuardrailsDecision"}
                session_plan: {$ref: "#/components/schemas/SessionPlan"}

        DryRunResponse:
            type: object
            required: [dry_run, provider, model, route_reason]
            properties:
                dry_run: {type: boolean, enum: [true]}
                provider: {type: string}
                model: {type: string}
                route_reason: {type: string}
                route_explanation: {type: object, additionalProperties: true}
                session_plan: {$ref: "#/components/schemas/SessionPlan"}
                guardrails: {$ref: "#/components/schemas/GuardrailsDecision"}
                budget: {$ref: "#/components/schemas/BudgetDecision"}
                context_plan: {type: object, additionalProperties: true}
                optimization:
                    type: object
                    properties:
                        candidate_count: {type: integer}
                        selected_count: {type: integer}
                        token_estimate: {type: integer}
                        token_budget: {type: integer}
                        score_weights: {type: object, additionalProperties: {type: number}}
                        selected_indices:
                            type: array
                            items: {type: integer}
                        confidence_scores:
                            type: array
                            items: {type: number}
                        avg_confidence: {type: number}
                        max_confidence: {type: number}
                options: {type: object, additionalProperties: true}

        BudgetDecision:
            type: object
            properties:
                max_input_tokens: {type: integer, nullable: true}
                max_output_tokens: {type: integer, nullable: true}
                mode: {type: string}
                status: {type: string}
                action: {type: string}
                estimated_input_tokens: {type: integer, nullable: true}
                effective_input_tokens: {type: integer, nullable: true}
                reason: {type: string}

        GuardrailsDecision:
            type: object
            properties:
                status: {type: string}
                action: {type: string}
                reason: {type: string}
                details: {type: object, additionalProperties: true}

        SessionPlan:
            type: object
            properties:
                query_type: {type: string}
                session_turn_count: {type: integer}
                include_summary: {type: boolean}
                planning_reason: {type: string}

        ErrorResponse:
            type: object
            properties:
                detail:
                    oneOf:
                        - type: string
                        - type: object
                            additionalProperties: true
```

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

## Instance Pattern & State Management

```python
wrapper1 = OllamaWrapper(connection_type="sync")
wrapper2 = OllamaWrapper(connection_type="async")

# Independent runtime state per instance
assert wrapper1 is not wrapper2

# Explicit teardown when API task is running
wrapper1.close(stop_server=True)
```

**Trade-off**:
- ✅ Flexible: Multiple independent instances can run in one process
- ❌ Lifecycle is explicit: callers should close instances that own background API tasks

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
- ❌ Distributed deployments (no multi-process support)
- ❌ Full HTTP chunked/SSE stream endpoint at API layer (provider streaming contract exists)
- ❌ Custom chunking strategies (hardcoded semantic + sliding window)
