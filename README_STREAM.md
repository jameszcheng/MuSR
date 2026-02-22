# MuSR-Stream (Phase 1 Scaffold)

MuSR-Stream extends MuSR with streaming evidence evaluation. Instead of one-shot QA, models update beliefs as evidence rounds are revealed.

This scaffold focuses on the murder mystery domain and keeps existing MuSR generation/eval paths intact.

## Setup

```bash
uv sync
```

## Stream Dataset Format

Each line in `datasets_stream/murder_mystery_stream.jsonl` is one streamed case:

```json
{
  "case_id": "murder_mystery_0_q0",
  "domain": "murder_mystery",
  "question": "Who is the most likely murderer?",
  "suspects": ["Mackenzie", "Ana"],
  "gold_suspect": "Mackenzie",
  "gold_index": 0,
  "metadata": {
    "source_case_index": 0,
    "source_question_index": 0,
    "chunk_mode": "sentence",
    "n_rounds": 22,
    "story_hash_id": 12345
  },
  "rounds": [
    {"round_id": 1, "evidence_text": "..."},
    {"round_id": 2, "evidence_text": "..."}
  ]
}
```

Notes:
- Evidence is chunked only from the final narrative context.
- Tree reasoning artifacts are not exposed to the model in prompts.

## 1) Convert Dataset to Stream Format

Sentence-by-sentence (default):

```bash
uv run python musr_stream/convert_murder_mystery_to_stream.py
```

Fixed chunking (`N=3` sentences per round):

```bash
uv run python musr_stream/convert_murder_mystery_to_stream.py --chunk-mode fixed --fixed-n 3
```

Useful options:
- `--input` (default: `datasets/murder_mystery.json`)
- `--output` (default: `datasets_stream/murder_mystery_stream.jsonl`)
- `--limit` to convert only first N source cases

## 2) Run Streaming Eval

Set API key:

```bash
export TOGETHER_API_KEY="..."
```

Run eval (default limits to 1 case for smoke test):

```bash
uv run python -m eval_stream.eval_stream
```

Optional flags:
- `--input datasets_stream/murder_mystery_stream.jsonl`
- `--output outputs/musr_stream_eval.json`
- `--model ServiceNow-AI/Apriel-1.6-15b-Thinker`
- `--limit 10`
- `--temperature 0.0`
- `--max-tokens 512`

## 3) Metrics (Phase 1)

- `final_accuracy`: final-round top suspect matches gold suspect
- `update_consistency`:
  - penalizes top-suspect flips across rounds
  - penalizes confidence decreases for the final predicted suspect
- `brier_final`: Brier score at final round
- `brier_mean`: mean Brier across rounds

## Expected Output Files

- `datasets_stream/murder_mystery_stream.jsonl`: streamed dataset
- `outputs/musr_stream_eval.json`: eval config + aggregate summary + per-case round traces
