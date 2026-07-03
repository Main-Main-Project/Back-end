"""Cross-Encoder 기반 검색 결과 재정렬 서비스."""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings


# Cross-Encoder 모델을 최초 한 번만 로드한다.
@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
        settings.RERANKER_MODEL,
        max_length=1024,
        device=settings.RERANKER_DEVICE,
    )


# 검색 결과를 Cross-Encoder 입력 문서로 변환한다.
def _build_candidate_text(result: dict) -> str:
    metadata = result.get("metadata") or {}
    content = (
        metadata.get("text")
        or metadata.get("source_text")
        or metadata.get("qa_text")
        or ""
    )
    content = str(content).strip()

    if not content:
        return ""

    law_name = (
        metadata.get("law_name")
        or metadata.get("category")
        or ""
    )
    article = metadata.get("article") or ""

    return (
        f"법령명: {law_name}\n"
        f"조문: {article}\n"
        f"본문: {content}"
    )


# 질문과 각 후보 원문의 관련도를 계산해 내림차순으로 정렬한다.
def rerank_search_results(
    question: str,
    search_results: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    candidates: list[tuple[dict, str]] = []

    for result in search_results:
        candidate_text = _build_candidate_text(result)

        if candidate_text:
            candidates.append((result, candidate_text))

    if not candidates:
        return []

    pairs = [
        [question, candidate_text]
        for _, candidate_text in candidates
    ]
    scores = get_reranker().predict(
        pairs,
        batch_size=4,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    reranked_results = []

    for (result, _), score in zip(candidates, scores):
        reranked_results.append(
            {
                **result,
                "metadata": dict(result.get("metadata") or {}),
                "pinecone_score": float(result.get("score") or 0.0),
                "rerank_score": float(score),
            }
        )

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return (
        reranked_results[:top_k]
        if top_k is not None
        else reranked_results
    )
