"""
message_crud: 대화 이력 조회 담당 (CRUD 계층)

역할:
- 특정 채팅방(session_id)의 이전 대화를 DB에서 꺼내옴
- chat_service가 호출 → 꺼낸 이력을 prompt_service.format_history에 넘김
  → 멀티턴(이전 대화 기억) 구현

정렬:
- question_at DESC(최신순)로 최근 N개를 가져옴
- 오래된 순 정렬은 format_history에서 reversed로 처리하므로 여기선 안 함
"""

from sqlalchemy.orm import Session

from app.db.models.chat import Message


def get_recent_messages(
    db: Session,
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    특정 채팅방의 최근 대화 이력을 최신순으로 반환
    (format_history가 reversed로 오래된 순으로 바꿔서 사용)

    입력:
        db         : DB 세션
        session_id : 채팅방 id
        limit      : 가져올 최근 대화 개수 (기본 10턴)

    반환:
        [{"question": "...", "answer": "..."}, ...]
    """
    rows = (
        db.query(Message)
        .filter(
            Message.session_id == session_id,
            Message.answer.isnot(None),
            Message.answer != "",
        )
        .order_by(Message.question_at.desc())
        .limit(limit)
        .all()
    )

    history = [
        {"question": r.question, "answer": r.answer}
        for r in rows
    ]

    return history