"""
Profile slow database queries to identify optimization opportunities.

Usage:
    uv run python scripts/profile_queries.py
"""
import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def profile_entry_points_query():
    """Profile the vector similarity entry points query."""
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    # Generate a dummy embedding vector (384 dimensions for bge-small-en-v1.5)
    dummy_embedding = str([0.1] * 384)

    print("=" * 80)
    print("PROFILING: Entry Points Query (Vector Similarity)")
    print("=" * 80)

    # Run EXPLAIN ANALYZE
    explain = await conn.fetch("""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        SELECT id, text, context, event_date, access_count, embedding,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memory_units
        WHERE agent_id = $2
          AND embedding IS NOT NULL
          AND (1 - (embedding <=> $1::vector)) >= 0.5
        ORDER BY embedding <=> $1::vector
        LIMIT 3
    """, dummy_embedding, "test_agent")

    for row in explain:
        print(row[0])

    await conn.close()


async def profile_neighbors_query(sample_node_ids):
    """Profile the neighbors JOIN query."""
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    print("\n" + "=" * 80)
    print("PROFILING: Neighbors Query (Graph Traversal)")
    print(f"Sample size: {len(sample_node_ids)} nodes")
    print("=" * 80)

    # Run EXPLAIN ANALYZE
    explain = await conn.fetch("""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        SELECT ml.from_unit_id, ml.to_unit_id, ml.weight, ml.link_type, ml.entity_id,
               mu.text, mu.context, mu.event_date, mu.access_count,
               mu.id as neighbor_id
        FROM memory_links ml
        JOIN memory_units mu ON ml.to_unit_id = mu.id
        WHERE ml.from_unit_id = ANY($1::uuid[])
          AND ml.weight >= 0.1
        ORDER BY ml.from_unit_id, ml.weight DESC
    """, sample_node_ids)

    for row in explain:
        print(row[0])

    await conn.close()


async def profile_embeddings_query(sample_node_ids):
    """Profile the batch embeddings fetch query."""
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    print("\n" + "=" * 80)
    print("PROFILING: Embeddings Query (Batch Fetch)")
    print(f"Sample size: {len(sample_node_ids)} nodes")
    print("=" * 80)

    # Run EXPLAIN ANALYZE
    explain = await conn.fetch("""
        EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
        SELECT id, embedding
        FROM memory_units
        WHERE id = ANY($1::uuid[])
    """, sample_node_ids)

    for row in explain:
        print(row[0])

    await conn.close()


async def get_sample_node_ids(batch_size=50):
    """Get sample node IDs for profiling."""
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    rows = await conn.fetch(f"""
        SELECT id FROM memory_units
        LIMIT {batch_size}
    """)

    await conn.close()
    return [row['id'] for row in rows]


async def check_indexes():
    """Check what indexes exist."""
    db_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)

    print("\n" + "=" * 80)
    print("CURRENT INDEXES")
    print("=" * 80)

    indexes = await conn.fetch("""
        SELECT
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename IN ('memory_units', 'memory_links')
        ORDER BY tablename, indexname
    """)

    for idx in indexes:
        print(f"\nTable: {idx['tablename']}")
        print(f"Index: {idx['indexname']}")
        print(f"Definition: {idx['indexdef']}")

    await conn.close()


async def main():
    print("Starting Query Profiling...")

    # Check indexes first
    await check_indexes()

    # Get sample node IDs
    sample_ids = await get_sample_node_ids(50)

    if sample_ids:
        print(f"\nGot {len(sample_ids)} sample node IDs for profiling")

        # Profile each query type
        await profile_entry_points_query()
        await profile_neighbors_query(sample_ids)
        await profile_embeddings_query(sample_ids)
    else:
        print("\nNo data in database - run ingestion first")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
