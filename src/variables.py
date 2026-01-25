from enum import Enum

class UserRole(str, Enum):
    PANELIST_1 = "panelist_1"
    PANELIST_2 = "panelist_2"
    JUDGE = "judge"
    MODERATOR = "moderator"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"