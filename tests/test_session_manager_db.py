"""SessionManager DB 메서드 테스트"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session_manager import ChatSession, SessionManager
from db.models import init_db


async def test_session_db_save_and_load():
    """세션 저장 및 로드 테스트"""
    print("\n" + "="*60)
    print("SessionManager DB 메서드 테스트")
    print("="*60)
    
    await init_db()
    
    # 1. 세션 생성 및 메시지 추가
    print("\n[1/5] 세션 생성 및 메시지 추가")
    session = ChatSession(
        session_id="db_test_001",
        topic="DB 테스트 토론",
        user_id="test_user",
        user_role="player_1"
    )
    
    print(f"✅ 세션 생성: {session.session_id}")
    print(f"   메모리 메시지 수: {len(session.message_history)}")
    
    # 추가 메시지 삽입
    session.add_judge_message("이 토론은 공정하게 진행되어야 합니다.")
    session.add_moderator_message("시간은 5분씩 할당됩니다.")
    
    print(f"   메시지 추가 후: {len(session.message_history)}개")
    
    # 2. 세션을 DB에 저장
    print("\n[2/5] 세션을 DB에 저장")
    await session.save_to_db()
    print(f"✅ DB 저장 완료")
    
    # 3. 동기화 확인
    print("\n[3/5] 메모리-DB 동기화 확인")
    is_synced = await session.check_history_sync()
    assert is_synced, "동기화 실패"
    print(f"✅ 동기화 확인 완료")
    
    # 4. 메모리 클리어 후 DB에서 로드
    print("\n[4/5] 메모리 클리어 후 DB에서 로드")
    original_count = len(session.message_history)
    session.message_history.clear()
    print(f"   메모리 클리어 후: {len(session.message_history)}개")
    
    loaded = await session.load_history_from_db()
    assert loaded, "로드 실패"
    assert len(session.message_history) == original_count, "로드된 메시지 수가 일치하지 않음"
    print(f"✅ DB에서 로드 완료: {len(session.message_history)}개 메시지")
    
    # 5. 로드된 데이터 검증
    print("\n[5/5] 로드된 데이터 검증")
    chat_history = session.get_chat_history()
    print(f"✅ Chat History (총 {len(chat_history)}개):")
    for i, msg in enumerate(chat_history[-5:]):  # 마지막 5개만 출력
        print(f"   {i}: [{msg.role}] {msg.content[:50]}...")
    
    print("\n" + "="*60)
    print("✅ 모든 DB 메서드 테스트 통과!")
    print("="*60)


async def test_session_manager_db_methods():
    """SessionManager DB 메서드 테스트"""
    print("\n" + "="*60)
    print("SessionManager 통합 DB 메서드 테스트")
    print("="*60)
    
    manager = SessionManager()
    
    # 1. 세션 생성
    print("\n[1/3] 세션 생성")
    session = manager.create_session(user_id="test_user", topic="Manager 테스트")
    session_id = session.session_id
    print(f"✅ 세션 생성: {session_id}")
    
    # 2. 메시지 추가 및 DB 저장
    print("\n[2/3] 메시지 추가 및 저장")
    session.add_judge_message("심사 기준을 명확히 합시다.")
    saved = await manager.save_session_to_db(session_id)
    assert saved, "저장 실패"
    print(f"✅ 저장 완료")
    
    # 3. 동기화 확인
    print("\n[3/3] 동기화 확인")
    is_synced = await manager.check_session_sync(session_id)
    assert is_synced, "동기화 확인 실패"
    print(f"✅ 동기화 확인 완료")
    
    print("\n" + "="*60)
    print("✅ SessionManager DB 메서드 테스트 통과!")
    print("="*60)


async def cleanup():
    """테스트 DB 삭제"""
    import os
    db_file = "./debate.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print("✅ 테스트 DB 삭제")


async def main():
    try:
        await test_session_db_save_and_load()
        await test_session_manager_db_methods()
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    # finally:
    #     await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
