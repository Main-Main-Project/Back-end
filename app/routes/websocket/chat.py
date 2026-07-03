import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
import traceback

from app.services.logger_service import create_log
from app.db.models.logger import Level
from app.services.chat_service import process_chat
from app.db.db import get_db
from app.db.models.chat import Chat
from app.core.security import verify_ws_token, decode_token
from app.core.token_store import token_store
from app.db.models.chat import Message 

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str = Query(...),
    session_id: Optional[str] = Query(None),  # 있으면 기존, 없으면 새 세션
    db: Session = Depends(get_db),
):
    trace_id = uuid.uuid4().hex

    user = await verify_ws_token(token, db)

    if not user:
        create_log(
            db,
            level=Level.WARN,
            endpoint="/ws/chat",
            method="WS",
            status_code=401,
            error_code="WS_AUTH_001",
            message="WebSocket 연결 실패 - 토큰 검증 실패",
            trace_id=trace_id,
        )
        await websocket.close(code=4001)
        return

    await websocket.accept()

    chat_session = None

    # 기존 세션으로 입장하는 경우
    if session_id:
        chat_session = db.query(Chat).filter(
            Chat.session_id == session_id,
            Chat.user_id == user.user_id,
        ).first()

        if not chat_session:
            create_log(
                db,
                level=Level.WARN,
                endpoint="/ws/chat",
                method="WS",
                status_code=404,
                error_code="WS_SESSION_001",
                message=f"WebSocket 세션 조회 실패 - 채팅방 없음: {session_id}",
                user_id=user.user_id,
                session_id=session_id,
                trace_id=trace_id,
            )
            await websocket.send_json({
                "type": "error",
                "code": "SESSION_NOT_FOUND",
                "message": "채팅방을 찾을 수 없습니다."
            })
            await websocket.close(code=4004)
            return

    try:
        while True:
            data = await websocket.receive_json()

            # 토큰 재검증
            try:
                payload = decode_token(token)
                jti = payload.get("jti")
                if token_store.is_blacklisted(jti):
                    create_log(
                        db,
                        level=Level.WARN,
                        endpoint="/ws/chat",
                        method="WS",
                        status_code=401,
                        error_code="WS_AUTH_002",
                        message="WebSocket 처리 중단 - 블랙리스트 토큰",
                        user_id=user.user_id,
                        session_id=chat_session.session_id if chat_session else None,
                        trace_id=trace_id,
                    )

                    await websocket.send_json({
                        "type": "error",
                        "code": "TOKEN_BLACKLISTED",
                        "message": "로그아웃된 토큰입니다."
                    })
                    await websocket.close(code=4001)
                    break
            except ValueError:
                create_log(
                    db,
                    level=Level.WARN,
                    endpoint="/ws/chat",
                    method="WS",
                    status_code=401,
                    error_code="WS_AUTH_003",
                    message="WebSocket 처리 중단 - 토큰 만료",
                    user_id=user.user_id,
                    session_id=chat_session.session_id if chat_session else None,
                    trace_id=trace_id,
                )

                await websocket.send_json({
                    "type": "error",
                    "code": "TOKEN_EXPIRED",
                    "message": "토큰이 만료되었습니다."
                })
                await websocket.close(code=4001)
                break

            message_text = (data.get("message") or "").strip()

            if not message_text:
                create_log(
                    db,
                    level=Level.WARN,
                    endpoint="/ws/chat",
                    method="WS",
                    status_code=400,
                    error_code="WS_MSG_001",
                    message="빈 메시지 전송",
                    user_id=user.user_id,
                    session_id=chat_session.session_id if chat_session else None,
                    trace_id=trace_id,
                )

                await websocket.send_json({
                    "type": "error",
                    "code": "EMPTY_MESSAGE",
                    "message": "메시지가 비어 있습니다.",
                })
                continue

            now = datetime.now(timezone.utc)

            # 첫 메시지일 때 세션 자동 생성
            if chat_session is None:
                chat_session = Chat(
                    user_id=user.user_id,
                    title=message_text[:20],
                )
                

                db.add(chat_session)
                db.commit()
                db.refresh(chat_session)

                create_log(
                    db,
                    level=Level.INFO,
                    endpoint="/ws/chat",
                    method="WS",
                    status_code=201,
                    message=f"WebSocket 새 채팅방 생성: {chat_session.title}",
                    user_id=user.user_id,
                    session_id=chat_session.session_id,
                    trace_id=trace_id,
                )

                await websocket.send_json({
                    "type": "session_created",
                    "session_id": str(chat_session.session_id),
                    "title": chat_session.title,
                })

            # 메시지 저장
            msg = Message(
                session_id=chat_session.session_id,
                user_id=user.user_id,
                question=message_text,
                question_at=datetime.now(timezone.utc),
                answer = "",
                answer_at=datetime.now(timezone.utc)
            )

            db.add(msg)
            db.commit()
            db.refresh(msg)            

            # 웹소켓 파이프라인 연결
            response = await process_chat(
                db=db,
                user_id=str(user.user_id),
                session_id=str(chat_session.session_id),
                question=message_text,
            )
            ai_answer = response.answer

            msg.answer = ai_answer
            msg.answer_at = datetime.now(timezone.utc)

            # 채팅방 updated_at 갱신
            if hasattr(chat_session, "updated_at"):
                chat_session.updated_at = datetime.now(timezone.utc)

            db.add(msg)
            db.commit()
            db.refresh(msg)

            create_log(
                db,
                level=Level.INFO,
                endpoint="/ws/chat",
                method="WS",
                status_code=200,
                message=f"AI 답변 생성 성공 - message_id={msg.message_id}",
                user_id=user.user_id,
                session_id=chat_session.session_id,
                trace_id=trace_id,
            )

            await websocket.send_json({
                "type": "message",
                "session_id": str(chat_session.session_id),
                "message_id": str(getattr(msg, "message_id", "")),
                "question": msg.question,
                "answer": msg.answer,
                "sources": response.sources,
                "question_at": msg.question_at.isoformat()
                if msg.question_at
                else None,
                "answer_at": msg.answer_at.isoformat()
                if msg.answer_at
                else None,
            })
            """
            # 프론트에 필요한 정보(가져다 쓰시오)
            await websocket.send_json({
                "type": "message",
                "session_id": str(chat_session.session_id),
                "message_id": str(getattr(msg, "message_id", "")),
                "question": msg.question,
                "answer": msg.answer,
                "verified": response.verified,
                "warnings": response.warnings,
                "sources": response.sources,
                "question_at": msg.question_at.isoformat() if msg.question_at else None,
                "answer_at": msg.answer_at.isoformat() if msg.answer_at else None,
            })
            """

    except WebSocketDisconnect:
        create_log(
            db,
            level=Level.INFO,
            endpoint="/ws/chat",
            method="WS",
            status_code=1000,
            message="WebSocket 연결 종료",
            user_id=user.user_id,
            session_id=chat_session.session_id if chat_session else None,
            trace_id=trace_id,
        )

    except Exception as e:
        db.rollback()

        create_log(
            db,
            level=Level.ERROR,
            endpoint="/ws/chat",
            method="WS",
            status_code=500,
            error_code="WS_999",
            message=f"WebSocket 서버 오류: {str(e)}",
            user_id=user.user_id if user else None,
            session_id=chat_session.session_id if chat_session else None,
            trace_id=trace_id,
        )
        
        print("[WebSocket ERROR]", repr(e), flush=True)
        traceback.print_exc()

        try:
            await websocket.send_json({
                "type": "error",
                "code": "SERVER_ERROR",
                "message": "서버 처리 중 오류가 발생했습니다.",
            })
            await websocket.close(code=1011)
        except Exception:
            pass