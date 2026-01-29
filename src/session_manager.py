import uuid
from typing import Dict, AsyncIterator, Optional, List
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from src.variables import UserRole
import random

from . import debate_agent


class ChatMessage(BaseModel):
    """개별 메시지"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ChatSession:
    """개별 채팅 세션"""
    
    def __init__(self, session_id: str, topic: str , user_id: str, user_role: str):
        self.session_id = session_id or str(uuid.uuid4())
        self.topic = topic
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # 메시지 히스토리 (표시용)
        self.chat_history: List[ChatMessage] = []
        # 모델에 전달할 히스토리 (role/content dict 또는 ModelMessage)
        self.message_history: List[dict | ModelMessage] = []
        self.user_id = user_id
        self.user_role = user_role
    
    def add_message(self, content: str, role: str):
        """사용자 메시지 추가"""
        self.chat_history.append(ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        ))
        self.updated_at = datetime.now()
    
    
    async def send_message(self, user_input: str, test_mode: bool = False) -> AsyncIterator[dict]:
        """
        메시지를 보내고 스트리밍 응답을 받음
        
        Yields:
            dict: 스트리밍 이벤트
                - type: "thinking" | "output"
                - content: 델타 콘텐츠
                - full_content: 누적된 전체 콘텐츠
                - partial: bool (output일 때만)
        """
        # 사용자 메시지를 히스토리에 추가
        self.add_message(content=user_input, role=self.user_role)
        
        final_output = None
        
        async for event in debate_agent.invoke(
            user_input,
            history=self.message_history,
            test_mode=test_mode
        ):
            yield event
            
            # 최종 output 저장
            if event.get("type") == "output" and not event.get("partial", False):
                final_output = event["content"]

            if event.get("type") == "history":
                self.message_history.extend(event["data"])
        
        # 최종 응답을 히스토리에 추가
        if final_output:
            self.add_message(final_output, role=UserRole.USER_ROLES[1] if self.user_role == UserRole.USER_ROLES[0] else UserRole.USER_ROLES[0])
    
    def get_chat_history(self) -> List[ChatMessage]:
        """채팅 히스토리 반환"""
        return self.chat_history
    
    def to_dict(self) -> dict:
        """세션 정보를 dict로 변환"""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.chat_history)
        }


class SessionManager:
    """채팅 세션 관리자"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def create_session(self, user_id: str, topic: str = None) -> ChatSession:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        user_role = random.choice(UserRole.USER_ROLES)
        session = ChatSession(session_id=session_id, topic=topic, user_id=user_id, user_role=user_role)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 조회"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[dict]:
        """모든 세션 목록 반환"""
        return [session.to_dict() for session in self.sessions.values()]
    
    async def send_message(
        self, 
        session_id: str, 
        user_input: str,
        test_mode: bool = False
    ) -> AsyncIterator[dict]:
        """
        특정 세션에 메시지 전송
        
        Args:
            session_id: 세션 ID
            user_input: 사용자 입력
            test_mode: 테스트 모드 여부
            
        Yields:
            dict: 스트리밍 이벤트
            
        Raises:
            ValueError: 세션이 존재하지 않을 때
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        async for event in session.send_message(user_input, test_mode=test_mode):
            yield event
    
    def get_chat_history(self, session_id: str) -> Optional[List[ChatMessage]]:
        """특정 세션의 채팅 히스토리 반환"""
        session = self.get_session(session_id)
        if session:
            return session.get_chat_history()
        return None
