import os
from enum import Enum
from typing import List
import yaml


def _load_roles_config() -> dict:
    """roles.yaml 설정 파일 로드"""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        os.pardir, 
        "config", 
        "roles.yaml"
    )
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 설정 로드
_config = _load_roles_config()

# 역할 설정
USER_ROLES: List[str] = _config["roles"]["user_roles"]
JUDGE_ROLE: str = _config["roles"]["judge_role"]
MODERATOR_ROLE: str = _config["roles"]["moderator_role"]

# 기본값
DEFAULT_USER_ROLE: str = _config["defaults"]["user_role"]
DEFAULT_OPPONENT_ROLE: str = _config["defaults"]["opponent_role"]

# 메시지 역할
MESSAGE_ROLE_USER: str = _config["message_roles"]["user"]
MESSAGE_ROLE_ASSISTANT: str = _config["message_roles"]["assistant"]
MESSAGE_ROLE_SYSTEM: str = _config["message_roles"]["system"]


class UserRole(str, Enum):
    USER_ROLES = USER_ROLES
    JUDGE = JUDGE_ROLE
    MODERATOR = MODERATOR_ROLE


class MessageRole(str, Enum):
    USER = MESSAGE_ROLE_USER
    ASSISTANT = MESSAGE_ROLE_ASSISTANT
    SYSTEM = MESSAGE_ROLE_SYSTEM