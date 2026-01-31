"""Async DB 모델 동작 테스트"""
import sys
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import SessionDB, MessageDB, AsyncSessionLocal, init_db


async def test_async_session_creation():
    """비동기 세션 생성 및 저장 테스트"""
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # 세션 생성
        db_session = SessionDB(
            session_id="async_test_001",
            user_id="user123",
            user_role="player_1",
            agent_role="player_2",
            topic="AI의 미래"
        )
        
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        
        print(f"✅ 비동기 세션 생성 성공: {db_session}")
        
        # 세션 조회
        from sqlalchemy import select
        stmt = select(SessionDB).where(SessionDB.session_id == "async_test_001")
        result = await session.execute(stmt)
        retrieved = result.scalar_one_or_none()
        
        assert retrieved is not None, "세션 조회 실패"
        assert retrieved.topic == "AI의 미래", "토픽이 일치하지 않음"
        
        print(f"✅ 비동기 세션 조회 성공: {retrieved}")


async def test_async_message_creation():
    """비동기 메시지 생성 및 저장 테스트"""
    async with AsyncSessionLocal() as session:
        # 메시지 생성
        message = MessageDB(
            session_id="async_test_001",
            message_json=json.dumps({"type": "test", "content": "비동기 테스트 메시지"}),
            chat_display="[Judge]: 비동기 판사 의견",
            role="judge"
        )
        
        session.add(message)
        await session.commit()
        await session.refresh(message)
        
        print(f"✅ 비동기 메시지 생성 성공: {message}")
        
        # 메시지 조회
        from sqlalchemy import select
        stmt = select(MessageDB).where(MessageDB.session_id == "async_test_001")
        result = await session.execute(stmt)
        messages = result.scalars().all()
        
        assert len(messages) > 0, "메시지 조회 실패"
        
        print(f"✅ 비동기 메시지 조회 성공: {len(messages)}개 메시지 발견")
        for msg in messages:
            print(f"   - {msg.role}: {msg.chat_display[:40]}...")


async def test_async_message_order():
    """비동기 메시지 순서 테스트"""
    async with AsyncSessionLocal() as session:
        # 여러 메시지 추가
        for i in range(3):
            message = MessageDB(
                session_id="async_test_001",
                message_json=json.dumps({"seq": i}),
                chat_display=f"비동기 메시지 {i}",
                role="user"
            )
            session.add(message)
        
        await session.commit()
        
        # 순서대로 조회
        from sqlalchemy import select
        stmt = select(MessageDB).where(
            MessageDB.session_id == "async_test_001"
        ).order_by(MessageDB.id)
        result = await session.execute(stmt)
        all_messages = result.scalars().all()
        
        print(f"✅ 비동기 메시지 순서 확인: 총 {len(all_messages)}개")
        for i, msg in enumerate(all_messages):
            print(f"   {i}: id={msg.id}, role={msg.role}, display={msg.chat_display[:30]}...")


async def cleanup():
    """테스트 DB 삭제"""
    import os
    db_file = "./debate.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print("✅ 테스트 DB 삭제 완료")


async def main():
    print("\n" + "="*60)
    print("Async DB 모델 테스트 시작")
    print("="*60)
    
    print("\n[1/3] 비동기 세션 생성 테스트")
    await test_async_session_creation()
    
    print("\n[2/3] 비동기 메시지 생성 테스트")
    await test_async_message_creation()
    
    print("\n[3/3] 비동기 메시지 순서 테스트")
    await test_async_message_order()
    
    print("\n" + "="*60)
    print("✅ 모든 비동기 DB 모델 테스트 통과!")
    print("="*60)
    
    # await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
