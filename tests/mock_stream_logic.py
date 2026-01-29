from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaThinkingPart
from pydantic_ai import ModelMessage
import json, asyncio

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
    # output_type이 설정되지 않았으면 일반 텍스트로 반환
    if not info.output_tools:
        # 일반 텍스트 응답
        text_response = "재생 에너지만으로는 기저 전력을 감당할 수 없습니다. 원자력과의 조화가 필요합니다."
        chunk_size = 5
        for i in range(0, len(text_response), chunk_size):
            await asyncio.sleep(0.05)
            chunk = text_response[i : i + chunk_size]
            yield chunk
        return
    
    output_tool_name = info.output_tools[0].name

    # 최종 반환할 데이터 (DebateResponse 구조)
    final_data = {
        "internal_monologue": "하 또 이 소리네. 기저 전력도 모르면서... 팩트로 혼내줘야지",
        "key_evidence": "독일의 에너지 전환 실패 사례 (전기료 급등)",
        "argument_speech": "재생 에너지만으로는 기저 전력을 감당할 수 없습니다. 원자력과의 조화가 필요합니다.",
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
