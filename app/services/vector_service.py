"""Pinecone 검색 결과와 RDB 원문을 연결한다."""

import uuid

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

import app.db.vector.client as pinecone_client
from app.db.models.chunk import Chunk
from app.db.vector.embedding import EmbeddingResult, embed_query


# 문자열을 UUID로 안전하게 변환한다.
def _to_uuid(value) -> uuid.UUID | None:
    if not value:
        return None

    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


# Pinecone 검색 점수를 float로 반환한다.
def _get_score(result: dict) -> float:
    try:
        return float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# 검색 결과에 모델이 사용할 본문이 있는지 확인한다.
def _has_usable_text(result: dict) -> bool:
    metadata = result.get("metadata") or {}
    content = (
        metadata.get("text")
        or metadata.get("source_text")
        or metadata.get("qa_text")
        or ""
    )
    return bool(str(content).strip())


# RDB 조회 후에도 본문이 없는 후보를 제거한다.
def _remove_empty_text_results(results: list[dict]) -> list[dict]:
    return [result for result in results if _has_usable_text(result)]


# dict 또는 객체에서 임베딩 값을 가져온다.
def _get_embedding_value(embedding, *names):
    if isinstance(embedding, dict):
        for name in names:
            value = embedding.get(name)
            if value is not None:
                return value

    for name in names:
        value = getattr(embedding, name, None)
        if value is not None:
            return value

    return None


# namespace를 기준으로 공용 법률과 사용자 문서를 구분한다.
def _get_source_type(namespace: str, result: dict) -> str:
    metadata = result.get("metadata") or {}

    if metadata.get("source_type"):
        return metadata["source_type"]

    if namespace == "public":
        return "legal_vector"

    return "document"


# Pinecone 결과를 서비스 공통 구조로 변환한다.
def _normalize_result(result: dict, namespace: str) -> dict:
    metadata = dict(result.get("metadata") or {})
    vector_id = metadata.get("vector_id") or result.get("id")

    if vector_id and not metadata.get("vector_id"):
        metadata["vector_id"] = vector_id

    return {
        "id": result.get("id"),
        "score": _get_score(result),
        "source_type": _get_source_type(namespace, result),
        "metadata": metadata,
    }


# Pinecone 후보의 vector_id로 RDB 본문과 조문을 결합한다.
def _attach_chunk_text(
    db: Session | None,
    results: list[dict],
    exclude_empty: bool = False,
) -> list[dict]:
    for result in results:
        metadata = result.get("metadata") or {}

        if not metadata.get("text") and metadata.get("source_text"):
            metadata["text"] = metadata["source_text"]

        result["metadata"] = metadata

    if db is None or not results:
        return (
            _remove_empty_text_results(results)
            if exclude_empty
            else results
        )

    public_ids: set[uuid.UUID] = set()
    document_ids: set[uuid.UUID] = set()

    for result in results:
        metadata = result.get("metadata") or {}
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        if vector_id is None:
            continue

        if result.get("source_type") == "legal_vector":
            public_ids.add(vector_id)
        else:
            document_ids.add(vector_id)

    public_texts: dict[str, str] = {}
    public_articles: dict[str, str] = {}

    if public_ids:
        statement = text(
            """
            SELECT vector_id, chunk_text, article
            FROM chunk_dataset
            WHERE vector_id::text IN :vector_ids
            """
        ).bindparams(bindparam("vector_ids", expanding=True))

        rows = db.execute(
            statement,
            {"vector_ids": [str(value) for value in public_ids]},
        ).mappings().all()

        public_texts = {
            str(row["vector_id"]): row["chunk_text"]
            for row in rows
            if row.get("chunk_text")
        }
        public_articles = {
            str(row["vector_id"]): row["article"]
            for row in rows
            if row.get("article")
        }

    document_texts: dict[str, str] = {}
    document_articles: dict[str, str] = {}

    if document_ids:
        rows = (
            db.query(Chunk)
            .filter(Chunk.vector_id.in_(document_ids))
            .all()
        )

        document_texts = {
            str(row.vector_id): row.chunk_text
            for row in rows
            if getattr(row, "chunk_text", None)
        }
        document_articles = {
            str(row.vector_id): row.article
            for row in rows
            if getattr(row, "article", None)
        }

    for result in results:
        metadata = result.get("metadata") or {}
        raw_id = metadata.get("vector_id") or result.get("id")
        vector_id = _to_uuid(raw_id)

        if vector_id is None:
            continue

        key = str(vector_id)

        if result.get("source_type") == "legal_vector":
            chunk_text = public_texts.get(key)
            article = public_articles.get(key)
        else:
            chunk_text = document_texts.get(key)
            article = document_articles.get(key)

        if chunk_text and not metadata.get("text"):
            metadata["text"] = chunk_text

        if article and not metadata.get("article"):
            metadata["article"] = article

        result["metadata"] = metadata

    return (
        _remove_empty_text_results(results)
        if exclude_empty
        else results
    )


# 선택된 Pinecone 후보에 RDB 원문을 붙이고 빈 후보를 제거한다.
def attach_chunk_text(
    db: Session | None,
    results: list[dict],
) -> list[dict]:
    return _attach_chunk_text(
        db=db,
        results=results,
        exclude_empty=True,
    )


# 질문 임베딩으로 Pinecone 후보를 검색한다.
def search_pinecone(
    question: str,
    context_mode: str = "general",
    user_id: str | None = None,
    top_k: int = 5,
    db: Session | None = None,
    load_rdb_text: bool = True,
    embedding: EmbeddingResult | dict | None = None,
) -> list[dict]:
    if context_mode not in ("general", "document", "hybrid"):
        raise ValueError(
            f"지원하지 않는 context_mode입니다: {context_mode}"
        )

    query_embedding = embedding or embed_query(question)
    dense_vector = _get_embedding_value(
        query_embedding,
        "dense_vecs",
        "dense",
        "dense_vector",
    )
    sparse_vector = _get_embedding_value(
        query_embedding,
        "lexical_weights",
        "sparse",
        "sparse_vector",
    )

    if dense_vector is None:
        raise ValueError(
            "임베딩 결과에서 dense vector를 찾을 수 없습니다."
        )

    if sparse_vector is None:
        raise ValueError(
            "임베딩 결과에서 sparse vector를 찾을 수 없습니다."
        )

    results: list[dict] = []

    if context_mode in ("general", "hybrid"):
        public_matches = pinecone_client.query(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            namespace=pinecone_client.public_namespace(),
            top_k=top_k,
            filter=None,
        )
        results.extend(
            _normalize_result(match, namespace="public")
            for match in public_matches
        )

    if context_mode in ("document", "hybrid") and user_id:
        user_namespace = pinecone_client.user_namespace(str(user_id))
        user_matches = pinecone_client.query(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            namespace=user_namespace,
            top_k=top_k,
            filter=None,
        )
        results.extend(
            _normalize_result(match, namespace=user_namespace)
            for match in user_matches
        )

    results.sort(key=_get_score, reverse=True)

    if context_mode != "hybrid":
        results = results[:top_k]
    if not load_rdb_text:
        return results

    return _attach_chunk_text(db=db, results=results)
