"""Multi-Centroid 기반 법률 질문 분류 서비스."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from app.core.config import settings
from app.db.vector.embedding import EmbeddingResult, embed_query


DomainLabel = Literal["legal", "nonlegal", "ambiguous"]


@dataclass(frozen=True)
class CentroidModel:
    """법률 및 비법률 중심 벡터를 보관한다."""

    legal_centroids: np.ndarray
    nonlegal_centroids: np.ndarray


@dataclass(frozen=True)
class DomainDecision:
    """질문 분류 결과와 판정 점수를 보관한다."""

    label: DomainLabel
    legal_score: float
    nonlegal_score: float
    best_score: float
    margin: float


# 벡터의 길이를 1로 정규화한다.
def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)

    return (vector / norm).astype(np.float32)


# 중심 벡터들을 행 단위로 정규화한다.
def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms == 0.0)] = 1.0
    return (vectors / norms).astype(np.float32)


# npz 파일에서 Multi-Centroid 모델을 한 번만 로드한다.
@lru_cache(maxsize=1)
def get_centroid_model() -> CentroidModel:
    model_path = Path(settings.CENTROID_MODEL_PATH).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Centroid 모델 파일을 찾을 수 없습니다: {model_path}"
        )

    with np.load(model_path, allow_pickle=False) as model_file:
        legal_centroids = np.asarray(
            model_file["legal_centroids"],
            dtype=np.float32,
        )
        nonlegal_centroids = np.asarray(
            model_file["nonlegal_centroids"],
            dtype=np.float32,
        )

    if legal_centroids.ndim != 2 or nonlegal_centroids.ndim != 2:
        raise ValueError("Centroid 배열은 2차원이어야 합니다.")

    if legal_centroids.shape[1] != nonlegal_centroids.shape[1]:
        raise ValueError("법률과 비법률 Centroid 차원이 다릅니다.")

    if not np.all(np.isfinite(legal_centroids)):
        raise ValueError("법률 Centroid에 비정상 값이 있습니다.")

    if not np.all(np.isfinite(nonlegal_centroids)):
        raise ValueError("비법률 Centroid에 비정상 값이 있습니다.")

    return CentroidModel(
        legal_centroids=_normalize_rows(legal_centroids),
        nonlegal_centroids=_normalize_rows(nonlegal_centroids),
    )


# Dense 벡터와 각 Centroid의 코사인 유사도로 질문을 분류한다.
def classify_dense_vector(
    dense_vector: list[float] | np.ndarray,
    model: CentroidModel | None = None,
    min_score: float | None = None,
    min_margin: float | None = None,
) -> DomainDecision:
    centroid_model = model or get_centroid_model()
    vector = _normalize_vector(
        np.asarray(dense_vector, dtype=np.float32)
    )

    expected_dimension = centroid_model.legal_centroids.shape[1]

    if vector.ndim != 1 or vector.shape[0] != expected_dimension:
        raise ValueError(
            "질문 임베딩과 Centroid의 차원이 일치하지 않습니다."
        )

    legal_score = float(
        np.max(centroid_model.legal_centroids @ vector)
    )
    nonlegal_score = float(
        np.max(centroid_model.nonlegal_centroids @ vector)
    )
    best_score = max(legal_score, nonlegal_score)
    margin = abs(legal_score - nonlegal_score)

    score_threshold = (
        settings.CENTROID_MIN_SCORE
        if min_score is None
        else min_score
    )
    margin_threshold = (
        settings.CENTROID_MIN_MARGIN
        if min_margin is None
        else min_margin
    )

    if best_score < score_threshold or margin < margin_threshold:
        label: DomainLabel = "ambiguous"
    elif legal_score >= nonlegal_score:
        label = "legal"
    else:
        label = "nonlegal"

    return DomainDecision(
        label=label,
        legal_score=legal_score,
        nonlegal_score=nonlegal_score,
        best_score=best_score,
        margin=margin,
    )


# 이미 생성된 임베딩으로 질문을 분류한다.
def classify_embedding(
    embedding: EmbeddingResult,
) -> DomainDecision:
    return classify_dense_vector(embedding.dense)


# 질문을 한 번 임베딩하고 분류 결과와 임베딩을 함께 반환한다.
def classify_question(
    question: str,
) -> tuple[DomainDecision, EmbeddingResult]:
    embedding = embed_query(question)
    decision = classify_embedding(embedding)
    return decision, embedding
