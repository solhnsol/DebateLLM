import pytest
import os
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
from xai_sdk import AsyncClient
from pydantic_ai.providers.xai import XaiProvider
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.asyncio
async def test_grok_reasoning_encrypted():
    """Test if encrypted reasoning content is returned"""
    xai_client = AsyncClient(api_key=os.environ.get("XAI_API_KEY", ""))
    provider = XaiProvider(xai_client=xai_client)
    model = XaiModel('grok-3-mini', provider=provider)
    settings = XaiModelSettings(xai_include_encrypted_content=True, reasoning_effort='high')
    
    agent = Agent(model=model, model_settings=settings)
    result = await agent.run("What is 25 * 37? Think step by step.")
    
    print(f"\n=== Reasoning Test ===")
    print(f"Output: {result.output}")
    print(f"\n--- Messages ---")
    for i, msg in enumerate(result.all_messages()):
        print(f"\nMessage {i}: {type(msg).__name__}")
        print(f"  {msg}")
        
        # Check for reasoning content
        if hasattr(msg, 'parts'):
            for j, part in enumerate(msg.parts):
                print(f"  Part {j}: {type(part).__name__}")
                if hasattr(part, 'content'):
                    print(f"    Content: {part.content[:100] if len(str(part.content)) > 100 else part.content}")
                # Check for reasoning/thinking parts
                if 'reasoning' in str(type(part)).lower() or 'thinking' in str(type(part)).lower():
                    print(f"    !!! REASONING FOUND: {part}")
    
    # Check usage for reasoning tokens
    if hasattr(result, 'all_messages'):
        for msg in result.all_messages():
            if hasattr(msg, 'usage'):
                print(f"\n--- Usage ---")
                print(f"Usage: {msg.usage}")
                if hasattr(msg.usage, 'details'):
                    print(f"Details: {msg.usage.details}")
    
    print(f"===================\n")
    
    assert result is not None


@pytest.mark.asyncio
async def test_grok_mini_reasoning():
    """Test if grok-3-mini returns reasoning_content in message"""
    agent = Agent('xai:grok-3-mini')
    result = await agent.run("What is 25 * 37? Think step by step.")
    
    print(f"\n=== Grok-3-Mini Test ===")
    print(f"Output: {result.output}")
    print(f"\n--- Messages ---")
    for i, msg in enumerate(result.all_messages()):
        print(f"\nMessage {i}: {type(msg).__name__}")
        print(f"  {msg}")
        
        # Check for reasoning_content attribute
        if hasattr(msg, 'reasoning_content'):
            print(f"  !!! reasoning_content found: {msg.reasoning_content}")
        
        if hasattr(msg, 'parts'):
            for j, part in enumerate(msg.parts):
                print(f"  Part {j}: {type(part).__name__}")
                # Check all attributes
                for attr in dir(part):
                    if 'reason' in attr.lower() or 'think' in attr.lower():
                        print(f"    {attr}: {getattr(part, attr, None)}")
    
    print(f"===================\n")
    
    assert result is not None
