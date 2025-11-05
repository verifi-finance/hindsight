"""
Example: Running many think operations in parallel with optimized connection pooling.

For 100 parallel think operations:
- Each think does 3 searches (world, agent, opinion)
- Each search acquires 1-3 connections briefly
- Total: ~300 concurrent connection requests

Solution: Increase pool_max_size to handle the concurrency.
"""
import asyncio
from memora import TemporalSemanticMemory


async def main():
    # For 100 parallel think operations, use a larger pool
    # Rule of thumb: pool_max_size >= (num_parallel_thinks * 3)
    memory = TemporalSemanticMemory(
        pool_min_size=10,        # Keep some connections warm
        pool_max_size=200        # Allow up to 200 concurrent connections
    )
    await memory.initialize()

    # Example: Run 100 think operations in parallel
    queries = [f"Query {i}" for i in range(100)]

    tasks = [
        memory.think_async(
            agent_id="test_agent",
            query=query,
            thinking_budget=50,
            top_k=10
        )
        for query in queries
    ]

    # Run all thinks in parallel
    results = await asyncio.gather(*tasks)

    print(f"Completed {len(results)} think operations")

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
