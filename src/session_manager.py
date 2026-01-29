import uuid
from typing import Dict, AsyncIterator, Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import ModelRequest, TextPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, SystemPromptPart
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
    
    def __init__(self, session_id: str, topic: str , user_id: str, user_role: Literal[UserRole.USER_ROLES]):
        self.session_id = session_id or str(uuid.uuid4())
        self.topic = topic
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # 모델에 전달할 히스토리
        self.message_history: List[dict | ModelMessage] = []
        self.user_id = user_id
        self.user_role = user_role
        self.agent_role = UserRole.USER_ROLES[1] if user_role == UserRole.USER_ROLES[0] else UserRole.USER_ROLES[0]
        
        self.add_moderator_message(f"토론이 시작되었습니다. 토론 주제는 {topic}입니다.")
        self.add_moderator_message(f"먼저 찬성 측 토론자부터 발언해 주세요.")
        self.add_system_message(f"[시스템 정보] 당신의 역할은 {self.agent_role}입니다.")
    
    def add_judge_message(self, content: str):
        """Judge 메시지 추가 - UserPromptPart로 prefix 포함"""
        self.message_history.append(
            ModelRequest(parts=[UserPromptPart(content=f"[Judge]: {content}")])
        )
    
    def add_moderator_message(self, content: str):
        """Moderator 메시지 추가 - UserPromptPart로 prefix 포함"""
        self.message_history.append(
            ModelRequest(parts=[UserPromptPart(content=f"[Moderator]: {content}")])
        )

    def add_system_message(self, content: str):
        """System 메시지 추가 - SystemPromptPart 사용"""
        self.message_history.append(
            ModelRequest(parts=[UserPromptPart(content=f"[System]: {content}")])
        )
    
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
        
        async for event in debate_agent.invoke(
            user_input,
            history=self.message_history,
            test_mode=test_mode
        ):
            yield event

            if event.get("type") == "history":
                self.message_history.extend(event["data"])
    
    def get_chat_history(self) -> List[ChatMessage]:
        """message_history에서 role과 text content만 필터링해서 반환 (UI용)"""
        result = []
        for msg in self.message_history:
            text_content = ""
            role = "unknown"
            
            # ModelRequest 또는 ModelResponse
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_content += part.content
                    elif isinstance(part, UserPromptPart):
                        # [Judge], [Moderator] 같은 prefix 파싱
                        content = part.content
                        if content.startswith("[Judge]:"):
                            role = "judge"
                            text_content += content.replace("[Judge]:", "").strip()
                        elif content.startswith("[Moderator]:"):
                            role = "moderator"
                            text_content += content.replace("[Moderator]:", "").strip()
                        else:
                            role = "user"
                            text_content += content
                
                # ModelRequest vs ModelResponse 구분
                if isinstance(msg, ModelRequest) and role == "unknown":
                    role = "user"
                elif isinstance(msg, ModelResponse) and role == "unknown":
                    role = "assistant"
            
            if text_content:
                result.append(ChatMessage(
                    role=role,
                    content=text_content,
                    timestamp=datetime.now()
                ))
        return result
    
    def to_dict(self) -> dict:
        """세션 정보를 dict로 변환"""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
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
