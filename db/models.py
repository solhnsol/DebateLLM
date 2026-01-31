"""SQLAlchemy ORM models for debate sessions and messages"""
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./debate.db")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False
)

# Create AsyncSessionLocal for DB sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base class for models
Base = declarative_base()


class SessionDB(Base):
    """Database model for debate sessions"""
    __tablename__ = "sessions"
    
    # Primary key
    session_id = Column(String, primary_key=True, index=True)
    
    # Session metadata
    user_id = Column(String, index=True, nullable=False)
    user_role = Column(String, nullable=False)  # player_1 or player_2
    agent_role = Column(String, nullable=False)  # player_1 or player_2
    topic = Column(String, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    # Relationship
    messages = relationship("MessageDB", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SessionDB(session_id={self.session_id}, user_id={self.user_id}, topic={self.topic})>"


class MessageDB(Base):
    """Database model for messages in debate sessions"""
    __tablename__ = "messages"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    
    # Foreign key
    session_id = Column(String, ForeignKey("sessions.session_id"), index=True, nullable=False)
    
    # Message content
    # JSON 형식의 전체 ModelMessage 배열 (여러 메시지를 한 row에)
    message_json = Column(Text, nullable=False)  # JSON array string
    
    # Chat display용 텍스트 배열 (JSON 형식)
    chat_display = Column(Text, nullable=False)  # JSON array string
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)
    
    # Relationship
    session = relationship("SessionDB", back_populates="messages")
    
    def __repr__(self):
        return f"<MessageDB(id={self.id}, session_id={self.session_id}, role={self.role})>"


# Create tables
async def init_db():
    """Create database tables asynchronously"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency for getting async DB session"""
    async with AsyncSessionLocal() as session:
        yield session


if __name__ == "__main__":
    import asyncio
    asyncio.run(init_db())
    print("Database tables created successfully!")
