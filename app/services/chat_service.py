"""RAG 검색부터 답변 생성까지 전체 흐름을 관리한다."""

from sqlalchemy.orm import Session

from app.crud.message_crud import get_recent_messages
from app.db.models.document import Document, Status
from app.schemas.chat.response import ChatAnswerResponse
from app.services.centroid_service import classify_question
from app.services.hallucination_service import build_response
from app.services.prompt_service import assemble_messages
from app.services.reranker_service import rerank_search_results
from app.services.service import generate_answer
from app.services.vector_service import attach_chunk_text, search_pinecone




# 직전 질문 하나와 현재 질문을 결합한다.
def build_contextual_query(
    question: str,
    history: list[dict],
) -> str:
    if not history:
        return question

    previous_question = (history[0].get("question") or "").strip()

    if previous_question and previous_question != question:
        return f"{previous_question} {question}"

    return question


# Cross-Encoder 점수를 우선하여 결과 점수를 반환한다.
def get_score(result: dict) -> float:
    try:
        score = result.get("rerank_score")

        if score is None:
            score = result.get("score")

        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


# 일반 또는 Hybrid 모드에 맞춰 최종 참고자료를 선택한다.
def select_llm_search_results(
    search_results: list[dict],
    context_mode: str,
    document_k: int = 2,
    public_k: int = 3,
) -> list[dict]:
    total_k = document_k + public_k

    if context_mode != "hybrid":
        return search_results[:total_k]

    document_results = sorted(
        (
            result
            for result in search_results
            if result.get("source_type") == "document"
        ),
        key=get_score,
        reverse=True,
    )
    public_results = sorted(
        (
            result
            for result in search_results
            if result.get("source_type") == "legal_vector"
        ),
        key=get_score,
        reverse=True,
    )

    selected = (
        document_results[:document_k]
        + public_results[:public_k]
    )
    selected.sort(key=get_score, reverse=True)
    return selected


# LLM을 호출하지 않는 차단 응답을 생성한다.
def build_reject_response(
    message: str = "죄송합니다. 법률 관련 질문에만 답변할 수 있습니다.",
) -> ChatAnswerResponse:
    return ChatAnswerResponse(
        answer=message,
        verified=True,
        warnings=[],
        unverified_refs=[],
        sources=[],
    )


# 질문을 받아 RAG 파이프라인 전체를 실행한다.
async def process_chat(
    db: Session,
    user_id: str,
    session_id: str,
    question: str,
    context_mode: str = "general",
    top_k: int = 20,
    history_limit: int = 10,
    history: list[dict] | None = None,
) -> ChatAnswerResponse:
    # 파일이 있는 세션은 공용 자료와 사용자 문서를 함께 검색한다.
    if db is not None and context_mode == "general":
        has_file = (
            db.query(Document)
            .filter(
                Document.session_id == session_id,
                Document.status.in_([Status.READY, Status.EMBEDDED]),
            )
            .first()
            is not None
        )

        if has_file:
            context_mode = "hybrid"

    # 이전 대화는 애매한 후속 질문의 맥락 확인에만 사용한다.
    if history is None:
        history = get_recent_messages(
            db=db,
            session_id=session_id,
            limit=history_limit,
        )
    # 현재 질문만으로 법률 영역을 먼저 판별한다.
    current_decision, current_embedding = classify_question(question)

    if current_decision.label == "nonlegal":
        return build_reject_response()

    ranking_query = question
    search_embedding = current_embedding

    # 현재 질문이 애매할 때만 직전 질문과 결합해 다시 판별한다.
    if current_decision.label == "ambiguous":
        if not history:
            return build_reject_response(
                "법률 질문인지 판단하기 어렵습니다. 질문을 조금 더 구체적으로 입력해주세요."
            )

        contextual_query = build_contextual_query(question, history)

        if contextual_query == question:
            return build_reject_response(
                "법률 질문인지 판단하기 어렵습니다. 질문을 조금 더 구체적으로 입력해주세요."
            )

        contextual_decision, contextual_embedding = classify_question(
            contextual_query
        )

        if contextual_decision.label != "legal":
            return build_reject_response()

        ranking_query = contextual_query
        search_embedding = contextual_embedding

    # Centroid 판별에 사용한 임베딩을 Pinecone 검색에 재사용한다.
    search_results = search_pinecone(
        question=ranking_query,
        context_mode=context_mode,
        user_id=user_id,
        top_k=top_k,
        db=db,
        load_rdb_text=False,
        embedding=search_embedding,
    )

    if not search_results:
        return build_reject_response(
            "질문과 관련된 법률 자료를 찾을 수 없습니다."
        )

    # Top-K의 vector_id로 RDB 원문을 붙이고 빈 본문을 제거한다.
    search_results = attach_chunk_text(
        db=db,
        results=search_results,
    )

    if not search_results:
        return build_reject_response(
            "관련 법률 자료의 본문을 확인할 수 없습니다."
        )

    # 질문과 RDB 원문을 Cross-Encoder로 재정렬한다.
    search_results = rerank_search_results(
        question=ranking_query,
        search_results=search_results,
    )

    if not search_results:
        return build_reject_response(
            "답변에 사용할 수 있는 관련 법률 자료가 없습니다."
        )

    # 일반은 상위 5개, Hybrid는 사용자 2개와 공용 3개를 선택한다.
    llm_search_results = select_llm_search_results(
        search_results=search_results,
        context_mode=context_mode,
        document_k=2,
        public_k=3,
    )

    if not llm_search_results:
        return build_reject_response(
            "답변에 사용할 수 있는 관련 법률 자료가 없습니다."
        )

    # 현재 질문, 대화 맥락, 최종 참고자료를 프롬프트로 조립한다.
    messages = assemble_messages(
        question=question,
        search_results=llm_search_results,
        history=history,
    )
    answer = await generate_answer(messages)

    # 답변 조문을 검증하고 출처를 포함한 응답을 반환한다.
    return build_response(answer, llm_search_results)
