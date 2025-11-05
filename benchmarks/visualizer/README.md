# Benchmark Visualizer

A standalone web service for visualizing benchmark results. Currently supports the LoComo benchmark with plans to add more benchmarks in the future.

## Features

- Interactive web interface for viewing benchmark results
- Detailed breakdown by category (Multi-hop, Single-hop, Temporal, Open-domain)
- Filter options to view all, correct, or incorrect answers
- Expandable Q&A details with reasoning and retrieved memories
- Overall and per-item accuracy statistics

## Running the Visualizer

### Option 1: Using the serve script (recommended)

```bash
cd benchmarks/visualizer
./serve.sh
```

### Option 2: Using uvicorn directly

```bash
cd benchmarks/visualizer
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8001
```

Then open your browser to: http://localhost:8001

## Usage

1. Select a benchmark from the dropdown:
   - **LoComo (search)**: Traditional two-step approach (search → LLM answer generation)
   - **LoComo (think)**: Integrated approach using think API (single call for retrieval + reasoning)
2. The visualization will automatically load and display:
   - Overall accuracy statistics
   - Category-wise performance breakdown
   - Detailed results for each conversation
3. Use the filter controls to show all answers, only incorrect, or only correct answers
4. Expand individual conversations to see Q&A details, reasoning, and retrieved memories

## API Endpoints

- `GET /` - Main visualizer page
- `GET /api/locomo?mode={search|think}` - Returns LoComo benchmark results as JSON
  - `mode=search` (default): Returns results from `benchmark_results.json`
  - `mode=think`: Returns results from `benchmark_results_think.json`

## Requirements

- FastAPI
- Uvicorn
- Python 3.11+

The visualizer reads benchmark results from:
- `benchmarks/locomo/benchmark_results.json` for search mode
- `benchmarks/locomo/benchmark_results_think.json` for think mode

Make sure to run the benchmark first to generate results:
- Search mode: `cd benchmarks/locomo && uv run python run_benchmark.py`
- Think mode: `cd benchmarks/locomo && uv run python run_benchmark.py --use-think`
