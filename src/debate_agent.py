import os
import asyncio
from contextlib import nullcontext
from pydantic_ai import (
    Agent,
    PartDeltaEvent,
    PartStartEvent,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.function import FunctionModel, AgentInfo, DeltaToolCall, DeltaThinkingPart, ThinkingPart, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic import BaseModel, Field, TypeAdapter
from typing import Optional, Sequence
from dotenv import load_dotenv
import json


load_dotenv()

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

class DebateResponse(BaseModel):
    internal_monologue: str = Field(
        ..., 
        description=(
            "RAW Internal thoughts in Korean (Banmal/Informal). "
            "React to the user's stupidity, analyze the logic gap, and plot your strategy."
        )
    )

    key_evidence: Optional[str] = Field(
        None, description="검색을 통해 찾은 강력한 사실이나 통계 등 명백한 증거."
    )
    argument_speech: str = Field(
        ..., description=("당신의 주장을 논리적으로 전개한 완전한 연설문이나 상대방의 주장에 대한 반박 연설문."
                          "Write like a chat message, not an essay.")
    )
    

model = OpenAIResponsesModel('gpt-5-nano')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
)

agent = Agent(
    model=model,
    model_settings=settings,
    output_type=DebateResponse,
    system_prompt=SYSTEM_PROMPT,
)

async def mock_stream_logic(messages: list[ModelMessage], info: AgentInfo):
    """
    FunctionModel의 stream_function에서 yield 가능한 형식:
    1. str - 텍스트 응답 스트리밍
    2. dict[int, DeltaToolCall] - Tool call 스트리밍
    3. dict[int, DeltaThinkingPart] - Thinking 스트리밍
    """
    
    # [단계 1] Thinking Process 시뮬레이션
    # DeltaThinkingPart를 dict[int, DeltaThinkingPart] 형태로 yield
    thoughts = ["english"]
    thoughts_desc = [
        '''
        thoughts...
        '''
    ]
    
    for i, thought in enumerate(thoughts):
        await asyncio.sleep(0.1)
        # DeltaThinkingPart를 dict로 감싸서 yield
        yield {0: DeltaThinkingPart(content=thought)}
        chunk_size = 5
        for j in range(0, len(thoughts_desc[i]), chunk_size):
            await asyncio.sleep(0.05)
            chunk = thoughts_desc[i][j : j + chunk_size]
            yield {0: DeltaThinkingPart(content=chunk)}
        


    # [단계 2] Structured Output 시뮬레이션 (DeltaToolCall 사용)
    assert info.output_tools is not None, "Output tools not found!"
    output_tool_name = info.output_tools[0].name

    # 최종 반환할 데이터 (DebateResponse 구조)
    final_data = {
        "internal_monologue": "하, 또 이 소리네. 기저 전력도 모르면서... 팩트로 혼내줘야지.",
        "argument_speech": "재생 에너지만으로는 기저 전력을 감당할 수 없습니다. 원자력과의 조화가 필요합니다.",
        "key_evidence": "독일의 에너지 전환 실패 사례 (전기료 급등)",
    }
    json_str = json.dumps(final_data, ensure_ascii=False)

    # Tool call 시작 (이름 먼저)
    yield {1: DeltaToolCall(name=output_tool_name)}

    # JSON 인자 쪼개서 스트리밍
    chunk_size = 5
    for i in range(0, len(json_str), chunk_size):
        await asyncio.sleep(0.05)
        chunk = json_str[i : i + chunk_size]
        yield {1: DeltaToolCall(json_args=chunk)}

async def invoke(user_input: str, history: Sequence[ModelMessage] = None, test_mode: bool = False):
    part_buffers: dict[int, ThinkingPart | ToolCallPart] = {}
    last_valid_output: DebateResponse | None = None  # 마지막으로 성공한 파싱 결과

    model_override = agent.override(model=FunctionModel(stream_function=mock_stream_logic)) \
                     if test_mode else nullcontext()

    with model_override:
        async for event in agent.run_stream_events(
            user_input,
            message_history=history
        ):
            outputs, last_valid_output = await _process_event(event, part_buffers, last_valid_output)
            for out in outputs:
                yield out
            


async def _process_event(event, part_buffers: dict[int, ThinkingPart | ToolCallPart], last_valid_output: DebateResponse | None):
    outputs = []

    if isinstance(event, PartStartEvent):
        if isinstance(event.part, ThinkingPart):
            part_buffers[event.index] = event.part
            outputs.append({
                "type": "thinking",
                "content": event.part.content or "",
                "full_content": event.part.content or ""
            })
        elif isinstance(event.part, ToolCallPart):
            part_buffers[event.index] = event.part

    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta):
            part_buffers[event.index] = event.delta.apply(part_buffers[event.index])
            outputs.append({
                "type": "thinking",
                "content": event.delta.content_delta,
                "full_content": part_buffers[event.index].content
            })
        elif isinstance(event.delta, ToolCallPartDelta):
            part_buffers[event.index] = event.delta.apply(part_buffers[event.index])
            part = part_buffers[event.index]
            if part.tool_name == "final_result":
                args_str = part.args_as_json_str()
                try:
                    validated = DebateResponse.model_validate_json(args_str)
                    outputs.append({
                        "type": "output",
                        "content": validated.model_dump(),
                        "partial": False
                    })
                    last_valid_output = validated
                except Exception:
                    try:
                        ta = TypeAdapter(DebateResponse)
                        validated = ta.validate_json(args_str, experimental_allow_partial='trailing-strings')
                        if validated != last_valid_output:
                            last_valid_output = validated
                            outputs.append({
                                "type": "output",
                                "content": validated.model_dump(),
                                "partial": True
                            })
                    except Exception:
                        pass

    return outputs, last_valid_output