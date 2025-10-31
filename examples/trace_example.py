"""
Example demonstrating search tracing functionality.

This script shows how to:
1. Enable search tracing
2. Retrieve the trace object
3. Export trace to JSON for visualization
"""
import asyncio
import json
from datetime import datetime, timezone
from memory import TemporalSemanticMemory


async def main():
    """Run the trace example."""
    # Initialize memory system
    memory = TemporalSemanticMemory()

    try:
        # Create a test agent
        agent_id = f"trace_demo_{datetime.now(timezone.utc).timestamp()}"

        print("=" * 70)
        print("SEARCH TRACE EXAMPLE")
        print("=" * 70)

        # Store some test memories
        print("\n1. Storing test memories...")
        await memory.put_async(
            agent_id=agent_id,
            content="Alice works at Google as a software engineer in Mountain View",
            context="conversation",
        )
        await memory.put_async(
            agent_id=agent_id,
            content="Bob also works at Google but in the New York office",
            context="conversation",
        )
        await memory.put_async(
            agent_id=agent_id,
            content="Charlie founded TechCorp, a startup in San Francisco",
            context="conversation",
        )
        await memory.put_async(
            agent_id=agent_id,
            content="Alice and Bob met at a Google conference last year",
            context="conversation",
        )
        print("   ✓ 4 memories stored")

        # Perform search with tracing enabled
        print("\n2. Searching with trace enabled...")
        query = "Who works at Google?"

        results, trace = await memory.search_async(
            agent_id=agent_id,
            query=query,
            thinking_budget=30,
            top_k=5,
            enable_trace=True,
        )

        print(f"   ✓ Search completed")

        # Display trace summary
        print("\n3. Trace Summary:")
        print(f"   - Query: {trace.query.query_text}")
        print(f"   - Thinking budget: {trace.query.thinking_budget}")
        print(f"   - Entry points found: {len(trace.entry_points)}")
        print(f"   - Total nodes visited: {trace.summary.total_nodes_visited}")
        print(f"   - Total nodes pruned: {trace.summary.total_nodes_pruned}")
        print(f"   - Budget used: {trace.summary.budget_used}")
        print(f"   - Budget remaining: {trace.summary.budget_remaining}")
        print(f"   - Results returned: {trace.summary.results_returned}")
        print(f"   - Total duration: {trace.summary.total_duration_seconds:.3f}s")
        print(f"   - Temporal links followed: {trace.summary.temporal_links_followed}")
        print(f"   - Semantic links followed: {trace.summary.semantic_links_followed}")
        print(f"   - Entity links followed: {trace.summary.entity_links_followed}")

        # Show entry points
        print("\n4. Entry Points:")
        for ep in trace.entry_points:
            print(f"   [{ep.rank}] {ep.text[:60]}... (similarity: {ep.similarity_score:.3f})")

        # Show visited nodes with their paths
        print("\n5. Search Path (First 5 visits):")
        for i, visit in enumerate(trace.visits[:5], 1):
            indent = "   "
            if visit.is_entry_point:
                print(f"{indent}[{i}] ENTRY POINT: {visit.text[:60]}...")
            else:
                parent = f"from {visit.parent_node_id[:8]}" if visit.parent_node_id else "?"
                link_info = f"via {visit.link_type}" if visit.link_type else ""
                print(f"{indent}[{i}] {parent} {link_info}: {visit.text[:60]}...")

            print(f"{indent}    - Activation: {visit.weights.activation:.3f}")
            print(f"{indent}    - Semantic sim: {visit.weights.semantic_similarity:.3f}")
            print(f"{indent}    - Recency: {visit.weights.recency:.3f}")
            print(f"{indent}    - Final weight: {visit.weights.final_weight:.3f}")

            if visit.neighbors_explored:
                followed = sum(1 for n in visit.neighbors_explored if n.followed)
                pruned = len(visit.neighbors_explored) - followed
                print(f"{indent}    - Neighbors: {followed} followed, {pruned} pruned")

        # Show pruning decisions
        if trace.pruned:
            print(f"\n6. Pruning Decisions (showing first 5 of {len(trace.pruned)}):")
            for prune in trace.pruned[:5]:
                print(f"   - Node {prune.node_id[:8]}: {prune.reason} (activation: {prune.activation:.3f})")

        # Show phase metrics
        print("\n7. Phase Metrics:")
        for pm in trace.summary.phase_metrics:
            print(f"   - {pm.phase_name}: {pm.duration_seconds:.3f}s")
            if pm.details:
                for key, value in pm.details.items():
                    if isinstance(value, float):
                        print(f"      • {key}: {value:.3f}")
                    else:
                        print(f"      • {key}: {value}")

        # Export to JSON
        print("\n8. Exporting trace to JSON...")
        trace_json = trace.to_json()
        output_file = f"trace_{agent_id}.json"
        with open(output_file, "w") as f:
            f.write(trace_json)
        print(f"   ✓ Trace saved to: {output_file}")
        print(f"   ✓ JSON size: {len(trace_json):,} bytes")

        # Show search results
        print("\n9. Search Results:")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] {result['text'][:70]}...")
            print(f"       Weight: {result['weight']:.3f} "
                  f"(act: {result['activation']:.2f}, "
                  f"sem: {result['semantic_similarity']:.2f}, "
                  f"rec: {result['recency']:.2f})")

        # Test helper methods
        print("\n10. Testing Helper Methods:")

        # Get path to first result
        if results:
            first_result_id = results[0]['id']
            path = trace.get_search_path_to_node(first_result_id)
            print(f"   - Path to top result has {len(path)} steps")

        # Count nodes by link type
        temporal_nodes = trace.get_nodes_by_link_type("temporal")
        semantic_nodes = trace.get_nodes_by_link_type("semantic")
        entity_nodes = trace.get_nodes_by_link_type("entity")
        print(f"   - Nodes reached via temporal links: {len(temporal_nodes)}")
        print(f"   - Nodes reached via semantic links: {len(semantic_nodes)}")
        print(f"   - Nodes reached via entity links: {len(entity_nodes)}")

        print("\n" + "=" * 70)
        print("TRACE EXAMPLE COMPLETE!")
        print("=" * 70)
        print(f"\nYou can now build a visualization using the trace data in:")
        print(f"  {output_file}")
        print("\nThe trace contains:")
        print(f"  - Complete search path with all nodes visited")
        print(f"  - Weight calculations for each node")
        print(f"  - Link information (type, weight, whether followed)")
        print(f"  - Pruning decisions with reasons")
        print(f"  - Performance metrics for each phase")

        # Cleanup
        print("\nCleaning up test agent...")
        await memory.delete_agent(agent_id)

    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
