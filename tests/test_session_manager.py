import asyncio
import time
import pytest
from src.session_manager import SessionManager


@pytest.mark.asyncio
async def test_session_manager():
    """SessionManager 테스트"""
    manager = SessionManager()
    
    session1_id = "12345"
    session2_id = "67890"

    session1_userid = "user_1"
    session2_userid = "user_2"

    session1_role = "player 1"
    session2_role = "player 2"

    # 1. 세션 생성
    print("\n=== 1. 세션 생성 ===")
    session1 = manager.create_session(topic="재생 에너지 vs 원자력", user_id=session1_userid)
    session2 = manager.create_session(topic="AI 윤리", user_id=session2_userid)
    
    print(f"세션 1 생성: {session1.session_id}")
    print(f"세션 2 생성: {session2.session_id}")
    
    # 2. 세션 목록 조회
    print("\n=== 2. 세션 목록 ===")
    sessions = manager.list_sessions()
    for s in sessions:
        print(f"  - {s['session_id'][:8]}... | {s['topic']} | 메시지: {s['message_count']}")
    
    # 3. 첫 번째 세션에 메시지 전송 (스트리밍)
    print(f"\n=== 3. 세션 1에 메시지 전송 (스트리밍) ===")
    user_input = "재생 에너지가 답이다."
    print(f"사용자: {user_input}\n")
    
    start_time = time.time()
    final_response = None
    
    async for event in manager.send_message(session1.session_id, user_input, test_mode=True):
        elapsed = f"{time.time() - start_time:.2f}s"
        event_type = event.get("type")
        
        if event_type == "thinking":
            # print(f"[{elapsed}] [Thinking] {event['content']}")
            pass
        elif event_type == "output":
            if not event.get("partial", False):
                final_response = event["content"]
                print(f"\n[{elapsed}] [최종 응답]")
                print(f"  - 내적독백: {final_response.get('internal_monologue', 'N/A')}")
                print(f"  - 연설문: {final_response.get('argument_speech', 'N/A')}")
                print(f"  - 증거: {final_response.get('key_evidence', 'N/A')}")
    
    print(f"\n총 소요시간: {time.time() - start_time:.2f}초")
    
    # 4. 채팅 히스토리 확인
    print("\n=== 4. 세션 1 채팅 히스토리 ===")
    history = manager.get_chat_history(session1.session_id)
    for msg in history:
        print(f"  [{msg.role}] {msg.content[:100]}...")
    
    # 5. 같은 세션에 두 번째 메시지 (컨텍스트 유지)
    print("\n=== 5. 세션 1에 두 번째 메시지 (컨텍스트 유지) ===")
    user_input2 = "그러면 저장 기술은 어떻게 되는가?"
    print(f"사용자: {user_input2}\n")
    
    async for event in manager.send_message(session1.session_id, user_input2, test_mode=True):
        if event.get("type") == "output" and not event.get("partial", False):
            response = event["content"]
            print(f"[AI 응답] {response.get('argument_speech', 'N/A')[:200]}...")
    
    # 6. 최종 히스토리 확인
    print("\n=== 6. 세션 1 최종 히스토리 ===")
    history = manager.get_chat_history(session1.session_id)
    print(f"총 메시지 수: {len(history)}")
    for i, msg in enumerate(history, 1):
        print(f"  {i}. [{msg.role}] {msg.content[:80]}...")
    
    # 7. 세션 삭제
    print("\n=== 7. 세션 2 삭제 ===")
    deleted = manager.delete_session(session2.session_id)
    print(f"세션 2 삭제 성공: {deleted}")
    
    print("\n=== 최종 세션 목록 ===")
    sessions = manager.list_sessions()
    for s in sessions:
        print(f"  - {s['session_id'][:8]}... | {s['topic']} | 메시지: {s['message_count']}")


if __name__ == "__main__":
    asyncio.run(test_session_manager())
