from app.services.prompts import SYSTEM_PROMPT

# history 받아서 LLM한테 보여줄 하나의 문자열 만듬
def format_history_context(history: list[dict]) -> str:
    """
    이전 대화는 현재 질문의 맥락 파악용으로만 전달한다.
    법적 근거로 쓰지 못하게 프롬프트에서 명확히 제한한다.
    """

    if not history:
        return ""

    lines = [
        "[이전 대화 맥락]",
        "아래 이전 대화는 사용자의 이어지는 질문을 이해하기 위한 참고용입니다.",
        "이전 대화의 답변 내용은 법적 근거로 사용하지 마세요.",
        "법령명, 조문번호, 수치, 처벌 내용, 판례번호는 반드시 [참고자료]에 있는 내용만 근거로 사용하세요.",
        "",
    ]

    # history는 최신순이라고 가정하므로 오래된 것부터 보여준다.
    for i, turn in enumerate(reversed(history), start=1):
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()

        if question:
            lines.append(f"{i}. 이전 질문: {question}")

        # 이전 답변은 맥락용으로만 짧게 전달한다.
        if answer:
            answer_summary = answer[:120].replace("\n", " ").strip()
            lines.append(f"   이전 답변 참고: {answer_summary}")

    return "\n".join(lines)


# select_llm_search_results 거친 후 모델에 넣을 형식만들기
def format_search_results(search_results: list[dict]) -> str:
    if not search_results:
        return "[참고자료]\n관련 자료를 찾을 수 없습니다."

    lines = [
        "[참고자료]",
        "아래 참고자료만 근거로 답변하세요.",
        "법령명이나 조문번호를 새로 생성하지 마세요.",
        "답변 마지막 줄에 사용한 조문을 '근거: 법령명 제○조' 형식으로 표시하세요.",
        "근거에는 반드시 참고자료의 [조문] 필드에 있는 값만 사용하세요.",
        "",
    ]

    for i, result in enumerate(search_results, start=1):
        metadata = result.get("metadata") or {}

        content = (
            metadata.get("text")
            or metadata.get("source_text")
            or metadata.get("qa_text")
            or ""
        )

        article = (metadata.get("article") or "").strip()
        law_name = (
            metadata.get("law_name")
            or metadata.get("category")
            or ""
        ).strip()

        content = str(content).strip() or "본문 내용 없음"

        display_article = article
        if article and law_name and law_name not in article:
            display_article = f"{law_name} {article}"

        if display_article:
            lines.append(f"{i}. [조문: {display_article}] [본문]\n{content}")
        elif law_name:
            lines.append(f"{i}. [법령명: {law_name}] [조문: 확인 불가] [본문]\n{content}")
        else:
            lines.append(f"{i}. [조문: 확인 불가] [본문]\n{content}")

    return "\n".join(lines)


# 두 개(format_history_context, format_search_results)를 하나로 합쳐서 최종 형태로 포장
def assemble_messages(
    question: str,
    search_results: list[dict],
    history: list[dict],
) -> list[dict]:

    history_context = format_history_context(history)
    search_str = format_search_results(search_results)

    user_parts = []

    if history_context:
        user_parts.append(history_context)

    user_parts.append(search_str)
    user_parts.append(f"[현재 질문]\n{question}")

    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return messages