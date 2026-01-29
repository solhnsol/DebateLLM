import pytest
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.debate_agent
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.asyncio
async def test_new_event_flow():
    """Test new event flow with status updates"""
    
    user_input = "재생 에너지가 답이다. 반박해봐."
    
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"📊 NEW EVENT FLOW TEST")
    print(f"{'='*80}")
    print(f"📝 Input: {user_input}")
    print(f"⏰ Start: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"{'='*80}\n")
    
    event_count = 0
    status_events = []
    output_chunks = []
    
    async for event in src.debate_agent.invoke(user_input, history=None, test_mode=False):
        event_count += 1
        elapsed = time.time() - start_time
        timestamp = f"[{elapsed:6.2f}s]"
        event_type = event.get("type")
        
        if event_type == "status":
            status = event.get("status")
            message = event.get("message", "")
            status_events.append(event)
            
            print(f"{timestamp} 🔔 STATUS: {status}")
            print(f"           Message: {message}")
            
            if status == "tool_calling":
                print(f"           Tool: {event.get('tool_name')}")
                print(f"           Args: {event.get('args')}")
            elif status == "tool_complete":
                print(f"           Tool: {event.get('tool_name')}")
                result_preview = event.get('result', '')[:100]
                print(f"           Result: {result_preview}...")
            
            print()
        
        elif event_type == "output":
            partial = event.get("partial", False)
            content = event.get("content", "")
            
            if not partial:
                # Final output
                print(f"{timestamp} ✅ OUTPUT_FINAL")
                print(f"           Total Length: {len(event.get('full_content', ''))}")
                output_chunks.append(event)
            elif content:
                # Only print first few and sample deltas
                if len(output_chunks) < 5 or len(output_chunks) % 20 == 0:
                    print(f"{timestamp} ✍️  OUTPUT_DELTA: {repr(content)}")
                output_chunks.append(event)
        
        elif event_type == "history":
            print(f"{timestamp} 📚 HISTORY_UPDATE")
            print()
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📈 Total Events: {event_count}")
    print(f"🔔 Status Events: {len(status_events)}")
    
    print(f"\n📋 Status Timeline:")
    for i, evt in enumerate(status_events):
        status = evt.get('status')
        msg = evt.get('message', '')
        print(f"   {i+1}. {status}: {msg}")
    
    print(f"\n📝 Output Chunks: {len(output_chunks)}")
    
    # Get final output
    final_output = None
    for evt in output_chunks:
        if not evt.get('partial'):
            final_output = evt.get('full_content')
            break
    
    if final_output:
        print(f"\n💬 Final Output ({len(final_output)} chars):")
        print(f"{'-'*80}")
        print(final_output[:500] + ("..." if len(final_output) > 500 else ""))
        print(f"{'-'*80}")
    
    print(f"{'='*80}\n")
    
    # Assertions
    assert event_count > 0
    assert len(status_events) > 0
    
    # Check that we got thinking status
    thinking_events = [e for e in status_events if e['status'] == 'thinking']
    assert len(thinking_events) > 0, "Should have at least one thinking event"
    
    # Check that thinking completed
    thinking_complete = [e for e in status_events if e['status'] == 'thinking_complete']
    assert len(thinking_complete) > 0, "Should have thinking_complete event"
    
    print("✅ All assertions passed!")
