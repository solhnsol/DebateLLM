import pytest
import time
import os
from pydantic_ai import Agent, PartStartEvent, PartDeltaEvent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
from pydantic_ai.messages import (
    ThinkingPart, ThinkingPartDelta, 
    TextPart, TextPartDelta,
    ToolCallPart, ToolCallPartDelta,
    ToolReturnPart
)
from xai_sdk import AsyncClient
from pydantic_ai.providers.xai import XaiProvider
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.asyncio
async def test_grok_basic():
    agent = Agent('xai:grok-4-1-fast-non-reasoning')
    result = await agent.run("Hello, world!")
    
    print(f"\n=== Grok Response ===")
    print(f"Type: {type(result)}")
    print(f"Result: {result}")
    print(f"Output: {result.output}")
    print(f"All messages: {result.all_messages()}")
    print(f"===================\n")
    
    assert result is not None


@pytest.mark.asyncio
async def test_grok_streaming_timeline():
    """Test Grok streaming response with timeline tracking"""
    
    # Setup agent with reasoning model
    xai_client = AsyncClient(api_key=os.environ.get("XAI_API_KEY", ""))
    provider = XaiProvider(xai_client=xai_client)
    model = XaiModel('grok-4-1-fast-reasoning', provider=provider)
    settings = XaiModelSettings(xai_include_encrypted_content=True, reasoning_effort='high')
    
    agent = Agent(model=model, model_settings=settings)
    
    user_input = "재생 에너지가 답이다. 너는 반대 입장을 취해서 토론해줘."
    
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"🚀 STREAMING TIMELINE TEST")
    print(f"{'='*80}")
    print(f"📝 User Input: {user_input}")
    print(f"⏰ Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"{'='*80}\n")
    
    # Track events
    event_count = 0
    thinking_accumulator = ""
    output_accumulator = ""
    
    async with agent.iter(user_input) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        event_count += 1
                        elapsed = time.time() - start_time
                        timestamp = f"[{elapsed:6.2f}s]"
                        
                        # Thinking Start Event
                        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                            delta = event.part.content or ""
                            thinking_accumulator += delta
                            print(f"{timestamp} 🧠 THINKING_START")
                            print(f"           Event #{event_count}")
                            print(f"           Part Type: {type(event.part).__name__}")
                            print(f"           Content Length: {len(delta)}")
                            print(f"           Has Signature: {hasattr(event.part, 'signature') and bool(event.part.signature)}")
                            if hasattr(event.part, 'signature') and event.part.signature:
                                print(f"           Signature Preview: {event.part.signature[:50]}...")
                            print()
                        
                        # Thinking Delta Event
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                            delta = event.delta.content_delta or ""
                            thinking_accumulator += delta
                            print(f"{timestamp} 💭 THINKING_DELTA")
                            print(f"           Event #{event_count}")
                            print(f"           Delta Length: {len(delta)}")
                            print(f"           Total Thinking Length: {len(thinking_accumulator)}")
                            if delta:
                                print(f"           Delta Preview: {delta[:50]}...")
                            print()
                        
                        # Text Start Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            delta = event.part.content or ""
                            output_accumulator += delta
                            print(f"{timestamp} 📝 TEXT_START")
                            print(f"           Event #{event_count}")
                            print(f"           Part Type: {type(event.part).__name__}")
                            print(f"           Content Length: {len(delta)}")
                            if delta:
                                print(f"           Content Preview: {delta[:100]}...")
                            print()
                        
                        # Text Delta Event
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta = event.delta.content_delta or ""
                            output_accumulator += delta
                            print(f"{timestamp} ✍️  TEXT_DELTA")
                            print(f"           Event #{event_count}")
                            print(f"           Delta Length: {len(delta)}")
                            print(f"           Total Output Length: {len(output_accumulator)}")
                            if delta:
                                print(f"           Delta: {repr(delta)}")
                            print()
                        
                        # Unknown Event
                        else:
                            print(f"{timestamp} ❓ UNKNOWN_EVENT")
                            print(f"           Event #{event_count}")
                            print(f"           Event Type: {type(event).__name__}")
                            print(f"           Event: {event}")
                            print()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📈 Total Events: {event_count}")
    print(f"🧠 Thinking Content Length: {len(thinking_accumulator)}")
    print(f"📝 Output Content Length: {len(output_accumulator)}")
    print(f"\n💬 Final Output:")
    print(f"{'-'*80}")
    print(output_accumulator)
    print(f"{'-'*80}")
    
    # Check usage
    if run.result:
        for msg in run.result.all_messages():
            if hasattr(msg, 'usage') and msg.usage:
                print(f"\n📊 Token Usage:")
                print(f"   Input Tokens: {msg.usage.input_tokens}")
                print(f"   Output Tokens: {msg.usage.output_tokens}")
                if hasattr(msg.usage, 'details') and msg.usage.details:
                    print(f"   Details: {msg.usage.details}")
    
    print(f"{'='*80}\n")
    
    assert event_count > 0
    assert len(output_accumulator) > 0


@pytest.mark.asyncio
async def test_grok_tool_calling_timeline():
    """Test Grok tool calling with timeline tracking"""
    
    # Setup agent with tool
    xai_client = AsyncClient(api_key=os.environ.get("XAI_API_KEY", ""))
    provider = XaiProvider(xai_client=xai_client)
    model = XaiModel('grok-4-1-fast-reasoning', provider=provider)
    settings = XaiModelSettings(xai_include_encrypted_content=True, reasoning_effort='medium')
    
    agent = Agent(model=model, model_settings=settings)
    
    # Define dummy tools (simple functions without context)
    @agent.tool_plain
    def get_renewable_energy_stats(energy_type: str) -> dict:
        """Get statistics about renewable energy sources.
        
        Args:
            energy_type: Type of renewable energy (solar, wind, hydro, etc.)
        """
        stats = {
            "solar": {"capacity_gw": 1000, "efficiency": 0.20, "cost_per_kwh": 0.04},
            "wind": {"capacity_gw": 850, "efficiency": 0.35, "cost_per_kwh": 0.05},
            "hydro": {"capacity_gw": 1300, "efficiency": 0.90, "cost_per_kwh": 0.02},
        }
        return stats.get(energy_type.lower(), {"error": "Unknown energy type"})
    
    @agent.tool_plain
    def get_nuclear_energy_stats() -> dict:
        """Get statistics about nuclear energy."""
        return {
            "capacity_gw": 400,
            "efficiency": 0.92,
            "cost_per_kwh": 0.03,
            "reliability": 0.93,
            "base_load_capable": True
        }
    
    user_input = "재생 에너지와 원자력 발전의 통계를 비교해서 분석해줘."
    
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"🔧 TOOL CALLING TIMELINE TEST")
    print(f"{'='*80}")
    print(f"📝 User Input: {user_input}")
    print(f"⏰ Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"{'='*80}\n")
    
    # Track events
    event_count = 0
    thinking_accumulator = ""
    output_accumulator = ""
    tool_calls = []
    
    async with agent.iter(user_input) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        event_count += 1
                        elapsed = time.time() - start_time
                        timestamp = f"[{elapsed:6.2f}s]"
                        
                        # Thinking Start Event
                        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                            delta = event.part.content or ""
                            thinking_accumulator += delta
                            print(f"{timestamp} 🧠 THINKING_START")
                            print(f"           Event #{event_count}")
                            print(f"           Has Signature: {hasattr(event.part, 'signature') and bool(event.part.signature)}")
                            print()
                        
                        # Thinking Delta Event
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
                            delta = event.delta.content_delta or ""
                            thinking_accumulator += delta
                            print(f"{timestamp} 💭 THINKING_DELTA")
                            print(f"           Event #{event_count}")
                            if delta:
                                print(f"           Delta: {delta[:50]}...")
                            print()
                        
                        # Tool Call Start Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                            print(f"{timestamp} 🛠️  TOOL_CALL_START")
                            print(f"           Event #{event_count}")
                            print(f"           Tool Name: {event.part.tool_name}")
                            print(f"           Tool Call ID: {event.part.tool_call_id}")
                            if hasattr(event.part, 'args') and event.part.args:
                                print(f"           Args: {event.part.args}")
                            tool_calls.append({
                                "name": event.part.tool_name,
                                "id": event.part.tool_call_id,
                                "start_time": elapsed
                            })
                            print()
                        
                        # Tool Call Delta Event
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
                            print(f"{timestamp} 🔨 TOOL_CALL_DELTA")
                            print(f"           Event #{event_count}")
                            if hasattr(event.delta, 'tool_name') and event.delta.tool_name:
                                print(f"           Tool Name Delta: {event.delta.tool_name}")
                            if hasattr(event.delta, 'json_args') and event.delta.json_args:
                                print(f"           Args Delta: {event.delta.json_args}")
                            print()
                        
                        # Tool Return Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, ToolReturnPart):
                            print(f"{timestamp} 🎁 TOOL_RETURN")
                            print(f"           Event #{event_count}")
                            print(f"           Tool Name: {event.part.tool_name}")
                            print(f"           Tool Call ID: {event.part.tool_call_id}")
                            print(f"           Content: {event.part.content}")
                            print()
                        
                        # Text Start Event
                        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            delta = event.part.content or ""
                            output_accumulator += delta
                            print(f"{timestamp} 📝 TEXT_START")
                            print(f"           Event #{event_count}")
                            print(f"           Content Length: {len(delta)}")
                            if delta:
                                print(f"           Preview: {delta[:100]}...")
                            print()
                        
                        # Text Delta Event
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            delta = event.delta.content_delta or ""
                            output_accumulator += delta
                            print(f"{timestamp} ✍️  TEXT_DELTA")
                            print(f"           Event #{event_count}")
                            print(f"           Delta: {repr(delta)}")
                            print()
                        
                        # Unknown Event
                        else:
                            print(f"{timestamp} ❓ UNKNOWN_EVENT")
                            print(f"           Event #{event_count}")
                            print(f"           Event Type: {type(event).__name__}")
                            if hasattr(event, 'part'):
                                print(f"           Part Type: {type(event.part).__name__}")
                            if hasattr(event, 'delta'):
                                print(f"           Delta Type: {type(event.delta).__name__}")
                            print()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📈 Total Events: {event_count}")
    print(f"🛠️  Tool Calls: {len(tool_calls)}")
    for i, tool_call in enumerate(tool_calls):
        print(f"   {i+1}. {tool_call['name']} (ID: {tool_call['id']}) @ {tool_call['start_time']:.2f}s")
    print(f"🧠 Thinking Content Length: {len(thinking_accumulator)}")
    print(f"📝 Output Content Length: {len(output_accumulator)}")
    print(f"\n💬 Final Output:")
    print(f"{'-'*80}")
    print(output_accumulator)
    print(f"{'-'*80}")
    
    # Check usage
    if run.result:
        for msg in run.result.all_messages():
            if hasattr(msg, 'usage') and msg.usage:
                print(f"\n📊 Token Usage:")
                print(f"   Input Tokens: {msg.usage.input_tokens}")
                print(f"   Output Tokens: {msg.usage.output_tokens}")
                if hasattr(msg.usage, 'details') and msg.usage.details:
                    print(f"   Details: {msg.usage.details}")
    
    print(f"{'='*80}\n")
    
    assert event_count > 0
    assert len(tool_calls) > 0
    assert len(output_accumulator) > 0