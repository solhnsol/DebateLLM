import asyncio
import src.debate_agent
import time

# 로그 설정 (확인용)
import logfire
logfire.configure()
logfire.instrument_pydantic_ai()
        
async def test_streaming():
    user_input = "재생 에너지가 답이다."
    
    # 기존 에이전트 가져오기 (src.debate_agent)
    # 테스트를 위해 output_type이 명시되어 있어야 함
    agent = src.debate_agent.agent
    
    start_time = time.time()
    print(f"\n=== 스트리밍 테스트 시작: {user_input} ===\n")

    # Mock 모델 주입
    async for item in src.debate_agent.invoke(
        user_input,
        history=None,
        test_mode=True
        ):
        elapsed = f"{time.time() - start_time:.2f}s"
        event_type = item["type"]
        
        if event_type == "thinking":
            print(f"[{elapsed}] [Thinking] {item['full_content']}")
        
        elif event_type == "output":
            # validated된 DebateResponse
            partial = item.get("partial", False)
            status = "..." if partial else "✓ FINAL"
            output_str = f"[{elapsed}] [Output {status}] {item['content']}".replace("\n", " ")
            print(output_str)
        elif event_type == "unknown_delta":
            print(f"[{elapsed}] [Unknown Event] {item}")

    print(f"\n\n=== 스트리밍 테스트 종료 (총 {time.time() - start_time:.2f}초) ===\n")


if __name__ == "__main__":
    asyncio.run(test_streaming())