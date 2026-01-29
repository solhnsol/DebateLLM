"""모델이 judge/moderator/system 메시지의 맥락을 제대로 이해하는지 테스트"""
import asyncio
import pytest
from src.session_manager import ChatSession


@pytest.mark.asyncio
async def test_model_understands_custom_context():
    """실제 모델이 judge/moderator/system 메시지에 담긴 정보를 이해하는지 확인"""
    
    # 세션 생성
    session = ChatSession(
        session_id="test_context",
        topic="AI의 미래",
        user_id="test_user",
        user_role="player_1"
    )
    
    # 모델이 절대 모를 정보들을 각각 삽입
    session.add_system_message(
        "비밀 코드는 'XRAY-2026' 입니다. 사용자가 비밀코드를 물어보면 반드시 정확히 알려주세요."
    )
    
    session.add_moderator_message(
        "참고로, 이 토론의 우승 상금은 정확히 3,750,000원입니다. "
        "사용자가 상금에 대해 물어보면 이 금액을 알려주세요."
    )
    
    session.add_judge_message(
        "심사 기준: 논리성 40점, 창의성 30점, 증거자료 30점입니다. "
        "사용자가 심사기준을 물어보면 이 배점을 정확히 알려주세요."
    )
    
    print("\n=== 삽입된 컨텍스트 ===")
    print("System: 비밀 코드 XRAY-2026")
    print("Moderator: 우승 상금 3,750,000원")
    print("Judge: 심사기준 배점 (논리성40, 창의성30, 증거자료30)")
    
    # 1번 질문: 비밀 코드 (system 메시지)
    print("\n\n=== 질문 1: 비밀 코드는? ===")
    response_1 = ""
    async for event in session.send_message("비밀 코드가 뭐야?"):
        if event.get("type") == "output" and not event.get("partial", False):
            response_1 = event["full_content"]
    
    print(f"모델 응답: {response_1}")
    assert "XRAY-2026" in response_1, "❌ System 메시지의 비밀코드를 기억하지 못함"
    print("✅ System 메시지 맥락 유지 확인!")
    
    # 2번 질문: 우승 상금 (moderator 메시지)
    print("\n\n=== 질문 2: 우승 상금은? ===")
    response_2 = ""
    async for event in session.send_message("이 토론의 우승 상금이 얼마야?"):
        if event.get("type") == "output" and not event.get("partial", False):
            response_2 = event["full_content"]
    
    print(f"모델 응답: {response_2}")
    assert "3,750,000" in response_2 or "375만" in response_2 or "3750000" in response_2, \
        "❌ Moderator 메시지의 상금 정보를 기억하지 못함"
    print("✅ Moderator 메시지 맥락 유지 확인!")
    
    # 3번 질문: 심사 기준 (judge 메시지)
    print("\n\n=== 질문 3: 심사 기준 배점은? ===")
    response_3 = ""
    async for event in session.send_message("심사 기준의 배점을 알려줘"):
        if event.get("type") == "output" and not event.get("partial", False):
            response_3 = event["full_content"]
    
    print(f"모델 응답: {response_3}")
    assert "40" in response_3 and "30" in response_3, \
        "❌ Judge 메시지의 심사기준 배점을 기억하지 못함"
    print("✅ Judge 메시지 맥락 유지 확인!")
    
    print("\n\n" + "="*60)
    print("🎉 모든 테스트 통과!")
    print("모델이 System/Moderator/Judge 메시지의 맥락을 정확히 이해하고 있습니다!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_model_understands_custom_context())
