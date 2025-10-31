#!/bin/bash
# Start the FastAPI server with hot reload
uv run uvicorn web.server:app --reload --host 0.0.0.0 --port 8080
