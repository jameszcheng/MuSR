# CS422 Benchmark Quickstart (2-3 Weeks)

This plan is optimized for your timeline and builds directly on MuSR artifacts already in this repo.

## Goal Coverage
- Long-context reasoning: add distractors + length tiers around MuSR cases.
- Joint logical + social (ToM): use murder-mystery cases as joint reasoning items.
- Dynamic belief consistency + counterfactual updates: use streamed rounds and belief-update metrics.

## Week-by-Week Plan

### Week 1 (Ship a vertical slice)
- Build a unified benchmark format for 3 tracks:
  - `long_context`
  - `tom`
  - `dynamic_belief`
- Generate small splits and run at least 2 models.
- Produce one summary table by track.

Exit criteria:
- Reproducible command creates benchmark files.
- Reproducible command runs eval and computes track-level metrics.

### Week 2 (Improve validity + scale)
- Scale number of examples (at least 150-300 total across tracks).
- Add stronger long-context controls (distractor count and repeat factor).
- Add dynamic-belief checks:
  - final accuracy
  - update consistency
  - flip count

Exit criteria:
- Stable data generation and metrics over multiple runs with fixed seed.

### Week 3 (Optional polish)
- Add counterfactual variants for a subset of dynamic-belief cases.
- Error analysis: 20-30 failures bucketed by failure type.
- Final report plots by track and model.

## Immediate Commands
1. Sync environment:
```bash
uv sync
```

2. Build benchmark files:
```bash
uv run python scripts/build_cs422_benchmark.py --outdir benchmark_runs/cs422_v0 --seed 7
```

3. Inspect outputs:
```bash
find benchmark_runs/cs422_v0 -maxdepth 2 -type f | sort
```

4. Start with smoke eval on `dynamic_belief` using existing stream eval path.

## Notes
- Keep MuSR internals largely unchanged for speed.
- Put your custom logic in new scripts and benchmark folders.
- If time gets tight, prioritize robust metrics + clear ablation over larger dataset size.
