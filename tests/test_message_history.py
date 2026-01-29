"""ModelRequest/ModelResponse 형식으로 메시지가 모델에 제대로 전달되는지 테스트"""
import asyncio
import pytest
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart, UserPromptPart
from src.session_manager import ChatSession
from src.debate_agent import agent


@pytest.mark.asyncio
async def test_modelmessage_format_sent_to_model():
    """send_message 호출 시 ModelMessage 형식 메시지가 모델에 전달되는지 확인"""
    
    received_messages = []
    call_count = [0]
    
    async def capture_model_function(messages, info: AgentInfo):
        """모델이 받는 messages를 캡처"""
        call_count[0] += 1
        if call_count[0] == 1:  # 첫 번째 호출만 캡처
            received_messages.clear()
            received_messages.extend(messages)
        
        # 더미 응답
        yield ModelResponse(parts=[TextPart(content='테스트 응답')])
    
    # 세션 생성
    session = ChatSession(
        session_id="test_session",
        topic="테스트 주제",
        user_id="test_user",
        user_role="player_1"
    )
    
    print(f"\n초기 message_history 길이: {len(session.message_history)}")
    
    # ModelMessage 형식으로 메시지 추가
    session.add_judge_message("이것은 판사의 의견입니다.")
    session.add_moderator_message("이것은 사회자의 의견입니다.")
    
    print(f"메시지 추가 후 message_history 길이: {len(session.message_history)}")
    
    # FunctionModel로 오버라이드
    model_override = agent.override(model=FunctionModel(stream_function=capture_model_function))
    
    with model_override:
        try:
            async for event in session.send_message("사용자 입력"):
                pass
        except:
            pass  # 에러 무시 (우리는 메시지 캡처가 목표)
    
    # 모델에 전달된 메시지 확인
    print(f"\n=== 모델에 전달된 메시지 (총 {len(received_messages)}개) ===")
    for i, msg in enumerate(received_messages):
        print(f"{i}: {type(msg).__name__}")
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    print(f"   UserPromptPart: {part.content[:50]}...")
    
    # 검증: ModelRequest 메시지 확인
    model_requests = [m for m in received_messages if isinstance(m, ModelRequest)]
    print(f"\n전달된 ModelRequest 개수: {len(model_requests)}")
    
    # Judge/Moderator prefix가 포함된 메시지 찾기
    judge_found = False
    moderator_found = False
    
    for msg in model_requests:
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                if "[Judge]:" in part.content:
                    judge_found = True
                    print(f"✅ Judge 메시지 발견: {part.content[:40]}...")
                if "[Moderator]:" in part.content:
                    moderator_found = True
                    print(f"✅ Moderator 메시지 발견: {part.content[:40]}...")
    
    assert judge_found, "Judge 메시지가 모델에 전달되지 않음"
    assert moderator_found, "Moderator 메시지가 모델에 전달되지 않음"
    
    print(f"\n✅ ModelMessage 형식 메시지가 모델에 제대로 전달됩니다!")


if __name__ == "__main__":
    asyncio.run(test_modelmessage_format_sent_to_model())
