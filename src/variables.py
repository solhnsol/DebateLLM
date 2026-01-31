import os
from enum import Enum
from typing import List
import yaml


# 역할 설정
USER_ROLES: List[str] = ['찬성 측', '반대 측']
PRO: str = USER_ROLES[0]
CON: str = USER_ROLES[1]
JUDGE_ROLE: str = '심사위원'
MODERATOR_ROLE: str = '사회자'


class UserRole(str, Enum):
    PRO = PRO
    CON = CON
    JUDGE = JUDGE_ROLE
    MODERATOR = MODERATOR_ROLE