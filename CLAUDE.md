# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Commands

**Package manager:** `uv` (not pip/conda). All commands should be prefixed with `uv run`.

```bash
# Setup
uv sync

# Run original MuSR eval
export TOGETHER_API_KEY="..."
uv run python -m eval.eval

# Build the full CS422 benchmark (long_context + dynamic_belief)
uv run python scripts/build_cs422_v1.py --outdir benchmark_runs/cs422_v1 --seed 7

# Build only the dynamic_belief track (finer control)
uv run python scripts/build_dynamic_belief.py \
  --outdir benchmark_runs/cs422_v1/dynamic_belief \
  --seed 7 --max-rounds 40 --setup-sentences 2 \
  --counterfactual-rate 0.7 --flip-rate 0.7

# Convert base dataset to stream format
uv run python musr_stream/convert_murder_mystery_to_stream.py

# Run streaming eval (defaults to 1 case smoke test)
uv run python -m eval_stream.eval_stream \
  --input benchmark_runs/cs422_v1/dynamic_belief/dev.jsonl \
  --output outputs/eval_out.json --limit 10

# Run build_cs422_v1.py from scripts/ directory (it imports build_dynamic_belief via relative import)
cd scripts && uv run python build_cs422_v1.py --outdir ../benchmark_runs/cs422_v1 --seed 7
```

**Redis caching** (optional): speeds up repeated LLM calls. Start with `redis-server`. Disabled by default in `src/__init__.py` (`cache = RedisCache(disabled=True)`).

## Architecture

### Data Flow

```
datasets/murder_mystery.json          (base dataset, ~44MB)
        │
        ├─► scripts/build_cs422_v1.py ──► benchmark_runs/cs422_v1/
        │         └── imports build_dynamic_belief.py              ├── long_context/{train,dev,test}.jsonl
        └─► scripts/build_dynamic_belief.py ──────────────────────► └── dynamic_belief/{train,dev,test}.jsonl
                                                                           │
                                                          eval_stream/eval_stream.py
                                                                           │
                                                                  outputs/*.json
```

### Core Layers

**`src/`** — Shared library (used by dataset generation, not benchmark scripts):
- `dataset_builder.py`: Recursive logic-tree expansion via LLM calls (Together API). This is the expensive generation step that produced `datasets/murder_mystery.json`.
- `model/`: LLM abstraction — `TogetherModel` (primary) and `HFModel`. Used by eval_stream for per-round predictions.
- `utils/redis_cache.py`: Optional caching layer. Import `from src import cache` then `cache.enable()`.
- `logic_tree/tree.py` + `madlib/madlib.py`: Core data structures for logic tree nodes and template substitution.

**`scripts/`** — Benchmark builders (pure Python, no LLM calls):
- `build_dynamic_belief.py`: Core logic for constructing dynamic_belief instances. Extracts `explicit` leaf facts from `intermediate_trees`, interleaves setup sentences and tree facts as rounds, optionally injects a counterfactual block that corrects one suspect's evidence and may flip the gold answer.
- `build_cs422_v1.py`: Orchestrates both tracks. Imports `build_dynamic_belief_case` from `build_dynamic_belief.py` (must be run from `scripts/` or with `scripts/` on PYTHONPATH).

**`eval_stream/eval_stream.py`** — Streaming evaluation:
- Iterates rounds, prompts model at each round, collects per-round `top_suspect` + `probabilities`
- Key metrics: `final_accuracy`, `update_consistency`, `brier_final`, `brier_mean`
- Counterfactual-specific: `flip_when_required`, `stability_when_not_required`, `recovery_rate`, `recovery_latency`

### CS422 Benchmark Tracks

| Track | Key field | What it tests |
|-------|-----------|---------------|
| `long_context` | `context` with injected distractor cases | Attention over long narratives |
| `dynamic_belief` | `rounds[]` with `evidence_type` + counterfactual metadata | Belief updates, counterfactual correction |

### Dynamic Belief Case Types

- **`stream_only`**: All rounds are sequential evidence (no correction)
- **`counterfactual`**: Late-round `counterfactual_update` block corrects prior evidence; `flip_required=true` means the gold answer changes after correction (`gold_before_text` → `gold_after_text`)

### Data Schemas

JSON schemas in `benchmark/schemas/`:
- `dynamic_belief.schema.json` — stream_only cases
- `dynamic_belief_cf.schema.json` — counterfactual cases
- `benchmark_instance.schema.json` — generic instance
- `prediction.schema.json` — model output format

### Benchmark Outputs

`benchmark_runs/cs422_v1/manifest.json` records generation config (seed, rates, counts) for reproducibility. A per-track manifest is also written for `dynamic_belief`.

## Key Conventions

- All benchmark JSONL files use `ensure_ascii=False`
- `build_cs422_v1.py` uses a 70/15/15 train/dev/test split
- The `intermediate_trees` field in base dataset cases uses `nodes[0]` as the root; `fact_type == "explicit"` leaf nodes are the facts used to build dynamic_belief rounds
- Redis cache is disabled by default; enable only when doing large-scale LLM evals
- `TOGETHER_API_KEY` env var required for any LLM-calling script
