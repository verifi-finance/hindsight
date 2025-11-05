#!/bin/bash
# Start the Benchmark Visualizer server with hot reload
cd "$(dirname "$0")"
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8001
