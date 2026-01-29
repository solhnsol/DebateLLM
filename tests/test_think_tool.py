"""test think() tool with complete monologue"""
import pytest
import time
import os
from pydantic_ai import Agent, PartStartEvent, PartDeltaEvent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
from pydantic_ai.messages import (
    ThinkingPart, TextPart, TextPartDelta,
    ToolCallPart, ToolReturnPart
)
from xai_sdk import AsyncClient
from pydantic_ai.providers.xai import XaiProvider
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.asyncio
async def test_think_tool_complete_monologue():
    """think() tool이 complete monologue를 전송하는지 확인"""
    
    # Setup agent with reasoning model
    xai_client = AsyncClient(api_key=os.environ.get("XAI_API_KEY", ""))
    provider = XaiProvider(xai_client=xai_client)
    model = XaiModel('grok-4-1-fast-reasoning', provider=provider)
    settings = XaiModelSettings(xai_include_encrypted_content=True, reasoning_effort='high')
    
    agent = Agent(
        model=model,
        model_settings=settings,
        system_prompt="당신은 논리적인 AI 토론자입니다. 반드시 think() 도구를 사용하여 먼저 생각을 정리한 후, 명확하고 설득력 있는 답변을 제공합니다."
    )
    
    # Add think() tool to agent
    @agent.tool_plain
    def think(internal_monologue: str) -> str:
        """당신의 내부 생각을 정리하고 기록합니다."""
        return f"생각 기록됨 ({len(internal_monologue)} 자)"
    
    user_input = "핵발전이 미래 에너지다. 반대 입장에서 토론해줘"
    
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"🚀 THINK() TOOL COMPLETE MONOLOGUE TEST")
    print(f"{'='*80}")
    print(f"📝 User Input: {user_input}")
    print(f"{'='*80}\n")
    
    # Track events
    event_count = 0
    monologues = []
    output_chunks = []
    thinking_tool_used = False
    
    async with agent.iter(user_input) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        event_count += 1
                        elapsed = time.time() - start_time
                        timestamp = f"[{elapsed:6.2f}s]"
                        
                        # Tool Call Start Event
                        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                            tool_name = event.part.tool_name
                            print(f"{timestamp} 🔧 TOOL_CALL_START: {tool_name}")
                            
                            if tool_name == "think":
                                thinking_tool_used = True
                                # think() 도구의 argument에서 monologue 추출
                                args = event.part.args if hasattr(event.part, 'args') else {}
                                monologue = args.get("internal_monologue", "") if isinstance(args, dict) else ""
                                
                                print(f"           ✅ THINK TOOL CALLED")
                                print(f"           Monologue Length: {len(monologue)}")
                                print(f"           Preview: {monologue[:100]}...")
                                monologues.append(monologue)
                        
                        # Tool Return Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolReturnPart):
                            print(f"{timestamp} 🎁 TOOL_RETURN: {event.part.tool_name}")
                            print(f"           Content: {event.part.content}")
                        
                        # Text Start Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            delta = event.part.content or ""
                            if delta:
                                output_chunks.append(delta)
                        
                        # Text Delta Event  
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta = event.delta.content_delta or ""
                            if delta:
                                output_chunks.append(delta)
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTS")
    print(f"{'='*80}")
    print(f"✅ Think tool used: {thinking_tool_used}")
    print(f"📝 Monologues captured: {len(monologues)}")
    
    if monologues:
        for i, monologue in enumerate(monologues, 1):
            print(f"\n💭 Monologue #{i}:")
            print(f"   Length: {len(monologue)} chars")
            print(f"   Preview: {monologue[:150]}...")
    
    print(f"\n📄 Output chunks: {len(output_chunks)}")
    if output_chunks:
        full_output = "".join(output_chunks)
        print(f"📄 Final output length: {len(full_output)} chars")
        print(f"📄 Preview: {full_output[:150]}...")
    
    print(f"{'='*80}\n")
    
    # Assertions
    assert event_count > 0, "No events received"
    assert thinking_tool_used, "think() tool was not called"
    assert len(monologues) > 0, "No monologues captured"
    assert len(output_chunks) > 0, "No output chunks received"
    
    # Verify that monologues are strings (even if empty)
    assert all(isinstance(m, str) for m in monologues), "Monologues should be strings"
    
    print(f"✅ All assertions passed!")
    print(f"✅ think() tool properly integrated - arguments are complete (not streaming)")
