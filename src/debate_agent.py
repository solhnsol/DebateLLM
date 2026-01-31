import os
from contextlib import nullcontext
from pydantic_ai import (
    Agent,
    PartStartEvent,
    PartDeltaEvent,
)

from pydantic_ai.models.function import FunctionModel, AgentInfo, DeltaToolCall, DeltaThinkingPart
from pydantic_ai.messages import (
    ModelMessage, 
    TextPart, TextPartDelta,
    ToolCallPart
)
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient

from typing import Literal, Sequence
from dotenv import load_dotenv

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
from xai_sdk import AsyncClient
from pydantic_ai.providers.xai import XaiProvider

load_dotenv()



# =======모델 선택 및 설정==========

model = OpenAIResponsesModel('gpt-5-mini')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
)

# model = GoogleModel('gemini-2.5-flash-lite')
# settings = GoogleModelSettings(google_thinking_config={'thinking_budget': 512, 'include_thoughts': False})

# xai_client = AsyncClient(api_key=os.environ.get("XAI_API_KEY", ""))
# provider = XaiProvider(xai_client=xai_client)
# model = XaiModel('grok-4-1-fast-reasoning', provider=provider)
# settings = XaiModelSettings(xai_include_encrypted_content=True, reasoning_effort='low')

# ================================



agent = Agent(
    model=model,
    model_settings=settings,
    instructions=
    '''토론의 역할 태그([시스템],[사회자] 등)은 자동으로 붙는다. 당신은 반드시 토론에 참여한 참여자로서 발화하고자 하는 내용만을 서술해야 한다.
    예시:
    잘된 예시(O): 운동은 중요합니다. 신체 기능 ~~~ <- 역할 태그 없이 입력함 (Good)
    잘못된 예시(X): [찬성 측]: 운동은 중요합니다. ~~~ <- 역할 태그를 입력함 (Bad)
    '[System]' 태그는 시스템의 명령에 해당하는 것으로, 일반 토론자인 당신과는 무관하다. 당신은 반드시 토론자 입장에서만 발화해야 한다.
    '''
)

tavily = TavilyClient()

@agent.tool_plain
async def search_web(query: str, search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'], search_amount: Literal['less', 'medium', 'more']) -> str:
    """
    웹에서 최신 정보를 검색할 때 사용합니다. 
    구체적인 검색어(query)를 입력받아 관련성 높은 검색 결과를 반환합니다.

    Args:
        query (str): 검색할 내용
        search_depth: 검색 깊이
        search_amount: 검색 결과 수
    """
    max_results_map = {'less': 3, 'medium': 5, 'more': 10}

    response = tavily.search(query=query, search_depth=search_depth, max_results=max_results_map[search_amount])
    
    results = []
    for r in response['results']:
        results.append(f"제목: {r['title']}\n내용: {r['content']}\nURL: {r['url']}\n")
    
    return "\n---\n".join(results)

tool_name_dict = {'search_web': 'Google 검색'}

async def invoke(user_input: str, history: Sequence[ModelMessage] = None, test_mode: bool = False, mock_stream_logic=None):

    model_override = agent.override(model=FunctionModel(stream_function=mock_stream_logic)) \
                     if test_mode else nullcontext()

    output_full = ""
    with model_override:
        async with agent.iter(user_input, message_history=history) as run:
            yield {
                "type": "status",
                "status": "thinking"
            }
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                           
                            
                            # ===== Tool Call Events =====
                            # Tool 호출 시작
                            if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                                yield {
                                    "type": "status",
                                    "status": "tool_calling",
                                    "tool_name": tool_name_dict.get(event.part.tool_name, event.part.tool_name),
                                    "tool_call_id": event.part.tool_call_id,
                                    "args": event.part.args if hasattr(event.part, 'args') else None
                                }
                            
                            # ===== Text Output Events =====
                            # Text 시작 이벤트
                            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                delta = event.part.content or ""
                                output_full += delta
                                
                                # 첫 output은 status 신호
                                yield {
                                    "type": "status",
                                    "status": "output"
                                }
                                
                                # 초기 content 전송
                                if delta:
                                    yield {
                                        "type": "output",
                                        "content": delta,
                                        "full_content": output_full,
                                        "partial": True
                                    }
                            
                            # Text 델타 이벤트 (스트리밍)
                            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                delta = event.delta.content_delta or ""
                                output_full += delta
                                yield {
                                    "type": "output",
                                    "content": delta,
                                    "full_content": output_full,
                                    "partial": True
                                }

            # 완료 후 최종 output과 history 반환
            if run.result:
                yield {
                    "type": "output",
                    "content": "",
                    "full_content": output_full,
                    "partial": False
                }
                yield {
                    "type": "history",
                    "data": run.result.new_messages()
                }