"""
message를 입력 받아 ollama에 보내서 답변을 받아온 뒤 최종적으로 문자열만 반환
"""

# HTTP 요청을 보내는 라이브러리
import httpx
# settings값 가져오기
from app.core.config import settings

# 비동기 함수를 사용하여 효율 높임
async def generate_answer(messages : list[dict]):
    # 함수의 역할 설명
    """
    Ollama에 메시지 리스트를 보내고 답변 문자열을 반환
    messages 형식 : [{"role" : "system", "content":"..."}, {"role":"user", "content":"..."}]
    """
    # Ollama에게 보낼 요청
    payload = {
        # config에 세팅해놓은 모델 선택 (학습시킨 모델 가져옴)
        "model" : settings.RAG_MODEL,
        # 실제로 모델에게 보낼 대화 내용
        "messages" : messages,
        # 답변을 한번에 받을지, 조금씩 받을지 정하는 곳 False > 한 번에
        "stream" : False,
        # 모델 답변 스타일 설정
        "options" : {
            # 모델의 창의성, 랜덤성 조정 0.3으로 보수적으로 답변(법률 도메인이라 선택)
            "temperature" : 0,
            # 모델이 다음 단어를 고를 때 후보 범위 설정 
            # 너무 낮으면 단조롭고, 너무 높으면 답변이 흔들릴 수 있다
            "top_p" : 0.85,
            # 모델이 반복되는 말을 억제하는 설정
            "repeat_penalty" : 1.15,
            "num_thread" : 4,
            "num_ctx" : 16384,      # 그릇 크게 → 10턴 안 짤림
            "num_predict" : 300,   
        }
    } 
    # Ollama와 통신하기 위한 
    # async with : 작업이 끝나면 자동으로 정리
    # httpx.AsyncClient : 비동기 방식으로 HTTP 요청을 보냄
    # timeout=300.0 : 요청을 최대 300초까지 기다리겠다
    async with httpx.AsyncClient(timeout=600.0) as client:
        # await : 비동기 함수 결과를 기다림
        # client.post : POST 요청을 보내는 코드 
        response = await client.post(
            # Ollama의 채팅 API로 요청을 보냄]
            # /api/chat → 채팅 형식으로 답변 생성
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            # payload를 json형태로 보내겠다
            json = payload
        )
        # HTTP 요청이 실패했는지 확인
        response.raise_for_status()
        # Ollama가 보내준 응답을 파이썬 딕셔너리로 바꿈
        data = response.json()
        # 답변 내용만 반환
        return data["message"]["content"]