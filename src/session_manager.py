import uuid
import json
import asyncio
from typing import Dict, AsyncIterator, Optional, List
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import TextPart
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, SystemPromptPart, ModelMessagesTypeAdapter
from src.variables import UserRole, USER_ROLES
from db.models import SessionDB, MessageDB, AsyncSessionLocal
from sqlalchemy import select, delete
import os

from . import debate_agent

def _read_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing prompt file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to read prompt file {path}: {exc}") from exc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "data", "system_prompts", "debate.txt")
SYSTEM_PROMPT = _read_text(SYSTEM_PROMPT_PATH)


class ChatMessage(BaseModel):
    """개별 메시지"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ChatSession:
    """개별 채팅 세션"""
    
    def __init__(self, session_id: str, topic: str , user_id: str, user_role: bool, init_messages: bool = True):
        self.session_id = session_id or str(uuid.uuid4())
        self.topic = topic
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # 모델에 전달할 히스토리
        self.message_history: List[dict | ModelMessage] = []
        self.user_id = user_id
        self.user_role = USER_ROLES[0] if user_role else USER_ROLES[1]
        self.agent_role = USER_ROLES[1] if user_role else USER_ROLES[0]

    async def async_init(self, system_prompt: str, init_messages: bool = True):
        """비동기 초기화 메서드"""
        await self.aadd_system_prompt(system_prompt)

        if init_messages:
            await self.aadd_moderator_message(f"토론이 시작되었습니다. 토론 주제는 [{self.topic}]입니다.")
            
            # AI가 찬성 측이면 AI가 먼저 발언
            if self.agent_role == USER_ROLES[0]:  # '찬성 측'
                await self.aadd_moderator_message(f"먼저 찬성 측 토론자부터 발언해 주세요.")
                await self.aadd_system_message(f"당신의 역할은 [{self.agent_role}]입니다. 토론 주제에 대한 찬성 입장으로 첫 발언을 시작하세요.")
                
                # AI의 첫 발화 생성 (유도 메시지는 히스토리에 저장 안 함)
                await self.generate_first_speech()
                
            else:  # AI가 반대 측이면 사용자(찬성 측)가 먼저
                await self.aadd_moderator_message(f"먼저 찬성 측 토론자부터 발언해 주세요.")
                await self.aadd_system_message(f"당신의 역할은 [{self.agent_role}]입니다.")
   
    async def aadd_system_prompt(self, content: str):
        """시스템 프롬프트 추가"""
        message = [ModelRequest(parts=[SystemPromptPart(content=content)])]
        self.message_history.extend(message)
        # system_prompt는 저장하지 않음 (모델에만 사용)

    async def aadd_judge_message(self, content: str):
        """Judge 메시지 추가 - UserPromptPart로 prefix 포함"""
        message = [ModelRequest(parts=[UserPromptPart(content=f"[Judge]: {content}")])]
        self.message_history.extend(message)
        async with AsyncSessionLocal() as db:
            await self.save_message_to_db(db, message)
            await db.commit()
    
    async def aadd_moderator_message(self, content: str):
        """Moderator 메시지 추가 - UserPromptPart로 prefix 포함"""
        message = [ModelRequest(parts=[UserPromptPart(content=f"[Moderator]: {content}")])]
        self.message_history.extend(message)
        async with AsyncSessionLocal() as db:
            await self.save_message_to_db(db, message)
            await db.commit()

    async def aadd_system_message(self, content: str):
        """System 메시지 추가"""
        message = [ModelRequest(parts=[UserPromptPart(content=f"[System]: {content}")])]
        self.message_history.extend(message)
        async with AsyncSessionLocal() as db:
            await self.save_message_to_db(db, message)
            await db.commit()
    
    async def send_message(self, user_input: str, test_mode: bool = False) -> AsyncIterator[dict]:
        """
        메시지를 보내고 스트리밍 응답을 받음
        
        Yields:
            dict: 스트리밍 이벤트
        """
        tagged_input = f"[상대 토론자]: {user_input}"
        async for event in debate_agent.invoke(
            tagged_input,
            history=self.message_history,
            test_mode=test_mode
        ):
            yield event

            if event.get("type") == "history":
                new_messages = event["data"]
                self.message_history.extend(new_messages)
                async with AsyncSessionLocal() as db:
                    await self.save_message_to_db(db, new_messages)
                    await db.commit()
    
    async def generate_first_speech(self) -> str:
        """
        AI의 첫 발언 생성 (유도 메시지는 히스토리에 저장하지 않음)
        
        Returns:
            str: AI의 첫 발언 텍스트
        """
        # 임시 히스토리로 응답 생성 (현재 히스토리 + 유도 메시지)
        first_speech = ""
        new_messages = []
        
        async for event in debate_agent.invoke(
            "[System]: 토론 주제에 대한 찬성 입장을 밝히며 첫 발언을 해주세요.",
            history=self.message_history,
            test_mode=False
        ):
            if event.get("type") == "output" and event.get("full_content"):
                first_speech = event.get("full_content")
            elif event.get("type") == "history":
                new_messages = event["data"]
        
        # 실제 히스토리에는 전체 메시지 저장 (API 히스토리와의 동기화를 위해)
        if new_messages:
            self.message_history.extend(new_messages)
            
            async with AsyncSessionLocal() as db:
                await self.save_message_to_db(db, new_messages)
                await db.commit()
        
        return first_speech
    
    def convert_chat_history(self, history: List[ModelMessage]) -> List[ChatMessage]:
        """message_history에서 role과 text content만 필터링해서 반환 (UI용)"""
        result = []
        for msg in history:
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
                        elif content.startswith("[System]:"):
                            role = "system"
                            text_content += content.replace("[System]:", "").strip()
                        else:
                            role = "user"
                            text_content += content.replace("[상대 토론자]:", "").strip()
                
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
    
    async def save_to_db(self) -> None:
        """세션 정보를 DB에 저장"""
        async with AsyncSessionLocal() as db:
            # 세션 정보 저장/업데이트
            stmt = select(SessionDB).where(SessionDB.session_id == self.session_id)
            result = await db.execute(stmt)
            db_session = result.scalar_one_or_none()
            
            if not db_session:
                db_session = SessionDB(
                    session_id=self.session_id,
                    topic=self.topic,
                    user_id=self.user_id,
                    user_role=self.user_role,
                    agent_role=self.agent_role,
                    created_at=self.created_at,
                    updated_at=datetime.now()
                )
                db.add(db_session)
            await db.commit()
   
    async def save_message_to_db(self, db, messages: List[ModelMessage]) -> None:
        """메시지들을 한 row에 저장"""
        if not messages:
            return
        
        # ModelMessage 배열을 JSON으로 저장
        try:
            message_json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
            message_json = message_json_bytes.decode('utf-8')
        except Exception as e:
            print(f"⚠️ 메시지 직렬화 실패: {e}")
            return
        
        # chat_history에서 텍스트 추출
        chat_history = self.convert_chat_history(messages)
        chat_display = json.dumps(
            [{"role": msg.role, "content": msg.content} for msg in chat_history],
            ensure_ascii=False
        )
        
        db_message = MessageDB(
            session_id=self.session_id,
            message_json=message_json,
            chat_display=chat_display
        )
        db.add(db_message)
    
    async def load_history_from_db(self) -> bool:
        """DB에서 세션 히스토리를 로드"""
        async with AsyncSessionLocal() as db:
            stmt = select(MessageDB).where(MessageDB.session_id == self.session_id).order_by(MessageDB.id)
            result = await db.execute(stmt)
            db_messages = result.scalars().all()
            
            if not db_messages:
                return False
            
            for db_msg in db_messages:
                try:
                    messages = ModelMessagesTypeAdapter.validate_json(db_msg.message_json)
                    self.message_history.extend(messages)
                except Exception as e:
                    print(f"⚠️ 메시지 역직렬화 실패: {e}")
                    continue
            return True

class SessionManager:
    """채팅 세션 관리자"""
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.system_prompt = SYSTEM_PROMPT
    
    async def create_session(self, user_id: str, user_role: bool, topic: str = None) -> ChatSession:
        """새 세션 생성 (DB 저장 포함)"""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id, 
            topic=topic, 
            user_id=user_id, 
            user_role=user_role
        )
        self.sessions[session.session_id] = session
        await session.async_init(system_prompt=self.system_prompt, init_messages=True)
        await session.save_to_db()
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 조회 (메모리 우선, 없으면 DB에서 로드)"""
        session = self.sessions.get(session_id)
        if session:
            return session
        return await self._load_session_from_db(session_id)
    
    async def _load_session_from_db(self, session_id: str) -> Optional[ChatSession]:
        """DB에서 세션과 히스토리를 로드"""
        async with AsyncSessionLocal() as db:
            stmt = select(SessionDB).where(SessionDB.session_id == session_id)
            result = await db.execute(stmt)
            db_session = result.scalar_one_or_none()
            if not db_session:
                return None
            
            # user_role을 bool로 변환
            user_role_bool = db_session.user_role == USER_ROLES[0]
            
            session = ChatSession(
                session_id=db_session.session_id,
                topic=db_session.topic,
                user_id=db_session.user_id,
                user_role=user_role_bool,  # bool로 변환해서 전달
                init_messages=False
            )
            session.created_at = db_session.created_at
            session.updated_at = db_session.updated_at
            
            # System prompt 먼저 추가
            await session.aadd_system_prompt(self.system_prompt)
            # 그 다음 DB에서 히스토리를 로드
            await session.load_history_from_db()
            self.sessions[session_id] = session
            return session
    
    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (DB + 메모리)"""
        async with AsyncSessionLocal() as db:
            try:
                stmt = delete(SessionDB).where(SessionDB.session_id == session_id)
                await db.execute(stmt)
                await db.commit()
            except Exception as e:
                await db.rollback()
                print(f"❌ 세션 삭제 실패: {e}")
                return False
        
        if session_id in self.sessions:
            del self.sessions[session_id]
        return True
    
    async def list_sessions(self) -> List[dict]:
        """모든 세션 목록 반환 (DB 기준)"""
        async with AsyncSessionLocal() as db:
            stmt = select(SessionDB)
            result = await db.execute(stmt)
            sessions = result.scalars().all()
            return [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "user_role": s.user_role,
                    "topic": s.topic,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sessions
            ]
    
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
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        async for event in session.send_message(user_input, test_mode=test_mode):
            yield event
    
    async def get_chat_history(self, session_id: str) -> Optional[List[ChatMessage]]:
        """특정 세션의 채팅 히스토리 반환"""
        session = await self.get_session(session_id)
        if session:
            return session.convert_chat_history(session.message_history)
        return None
    
    # ============ DB 관련 메서드 ============
    
    async def save_session_to_db(self, session_id: str) -> bool:
        """특정 세션을 DB에 저장"""
        session = await self.get_session(session_id)
        if not session:
            print(f"❌ 세션을 찾을 수 없음: {session_id}")
            return False
        
        try:
            await session.save_to_db()
            print(f"✅ 세션 저장 완료: {session_id}")
            return True
        except Exception as e:
            print(f"❌ 세션 저장 실패: {e}")
            return False
    
    async def load_session_from_db(self, session_id: str) -> bool:
        """DB에서 세션 히스토리를 로드"""
        session = await self.get_session(session_id)
        if not session:
            print(f"❌ 세션을 찾을 수 없음: {session_id}")
            return False
        
        return await session.load_history_from_db()