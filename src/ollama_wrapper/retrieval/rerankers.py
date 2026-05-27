from __future__ import annotations

from typing import Any

from .contracts import Reranker


class LLMChunkIDReranker(Reranker):
    """Default reranker that asks the model to choose the best chunk id."""

    def choose(self, query: str, candidates: list[dict[str, Any]], wrapper: Any) -> int:
        if not candidates:
            return 0

        rerank_manifest = "".join([f"--- CANDIDATE CHUNK ID {i} ---\n{c['text']}\n\n" for i, c in enumerate(candidates)])
        rerank_prompt = (
            "You are an AI data reranking module. Grade the candidate text chunks below based on their relevance "
            f"to the user query: '{query}'. Select the SINGLE most contextually informative chunk ID. "
            "Output ONLY the selected chunk ID number and nothing else. Do not justify your response.\n\n"
            f"{rerank_manifest}"
        )

        response = wrapper._sync_client.generate(model=wrapper.llm_model, prompt=rerank_prompt)
        raw_decision = wrapper._strip_thinking_tags(response.response)
        chosen_id = int("".join(filter(str.isdigit, raw_decision)))
        if 0 <= chosen_id < len(candidates):
            return chosen_id
        return 0


class FirstCandidateReranker(Reranker):
    """Deterministic reranker primarily for tests and baseline comparisons."""

    def choose(self, query: str, candidates: list[dict[str, Any]], wrapper: Any) -> int:
        _ = query
        _ = wrapper
        return 0
