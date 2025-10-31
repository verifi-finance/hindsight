-- Migration: Add composite index for spreading activation neighbor queries
-- This index optimizes the WHERE ml.from_unit_id::text = ANY($1) AND ml.weight >= 0.1 query
-- which is used during spreading activation search.

-- Composite index for spreading activation neighbor queries (from_unit_id + weight filter)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_from_weight
ON memory_links(from_unit_id, weight DESC)
WHERE weight >= 0.1;
