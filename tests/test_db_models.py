"""DB 모델 기본 동작 테스트"""
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import SessionDB, MessageDB, SessionLocal, init_db


def test_session_creation():
    """세션 생성 및 저장 테스트"""
    init_db()
    
    db = SessionLocal()
    
    try:
        # 세션 생성
        session = SessionDB(
            session_id="test_session_001",
            user_id="user123",
            user_role="player_1",
            agent_role="player_2",
            topic="AI의 미래"
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        print(f"✅ 세션 생성 성공: {session}")
        
        # 세션 조회
        retrieved = db.query(SessionDB).filter(SessionDB.session_id == "test_session_001").first()
        assert retrieved is not None, "세션 조회 실패"
        assert retrieved.topic == "AI의 미래", "토픽이 일치하지 않음"
        
        print(f"✅ 세션 조회 성공: {retrieved}")
        
        return session
        
    finally:
        db.close()


def test_message_creation(session_id="test_session_001"):
    """메시지 생성 및 저장 테스트"""
    db = SessionLocal()
    
    try:
        # 메시지 생성
        message = MessageDB(
            session_id=session_id,
            message_json=json.dumps({"type": "test", "content": "테스트 메시지"}),
            chat_display="[Judge]: 이것은 판사의 의견입니다",
            role="judge"
        )
        
        db.add(message)
        db.commit()
        db.refresh(message)
        
        print(f"✅ 메시지 생성 성공: {message}")
        
        # 메시지 조회
        retrieved = db.query(MessageDB).filter(
            MessageDB.session_id == session_id
        ).first()
        
        assert retrieved is not None, "메시지 조회 실패"
        assert retrieved.role == "judge", "역할이 일치하지 않음"
        assert retrieved.chat_display == "[Judge]: 이것은 판사의 의견입니다", "display 내용이 일치하지 않음"
        
        print(f"✅ 메시지 조회 성공: {retrieved}")
        
        return message
        
    finally:
        db.close()


def test_session_message_relationship():
    """세션-메시지 관계 테스트"""
    db = SessionLocal()
    
    try:
        # 세션 조회 및 메시지 확인
        session = db.query(SessionDB).filter(
            SessionDB.session_id == "test_session_001"
        ).first()
        
        assert session is not None, "세션을 찾을 수 없음"
        assert len(session.messages) > 0, "메시지가 없음"
        
        print(f"✅ 관계 확인: 세션 {session.session_id}에 {len(session.messages)}개의 메시지 있음")
        
        for msg in session.messages:
            print(f"   - {msg.role}: {msg.chat_display[:40]}...")
        
    finally:
        db.close()


def test_message_order():
    """메시지 순서 테스트 (ID 기반)"""
    db = SessionLocal()
    
    try:
        # 여러 메시지 추가
        for i in range(3):
            message = MessageDB(
                session_id="test_session_001",
                message_json=json.dumps({"seq": i}),
                chat_display=f"메시지 {i}",
                role="user"
            )
            db.add(message)
        
        db.commit()
        
        # 순서대로 조회
        messages = db.query(MessageDB).filter(
            MessageDB.session_id == "test_session_001"
        ).order_by(MessageDB.id).all()
        
        print(f"✅ 메시지 순서 확인: 총 {len(messages)}개")
        for i, msg in enumerate(messages):
            print(f"   {i}: id={msg.id}, role={msg.role}, display={msg.chat_display[:30]}...")
        
    finally:
        db.close()


def cleanup():
    """테스트 데이터 삭제"""
    import os
    db_file = "./debate.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        print("✅ 테스트 DB 삭제 완료")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DB 모델 테스트 시작")
    print("="*60)
    
    print("\n[1/4] 세션 생성 테스트")
    test_session_creation()
    
    print("\n[2/4] 메시지 생성 테스트")
    test_message_creation()
    
    print("\n[3/4] 세션-메시지 관계 테스트")
    test_session_message_relationship()
    
    print("\n[4/4] 메시지 순서 테스트")
    test_message_order()
    
    print("\n" + "="*60)
    print("✅ 모든 DB 모델 테스트 통과!")
    print("="*60)
    
    # cleanup()
