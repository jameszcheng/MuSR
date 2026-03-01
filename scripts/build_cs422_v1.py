import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from build_dynamic_belief import build_dynamic_belief_case


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c and c.strip()]


def to_stream_from_musr(musr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cidx, case in enumerate(musr_rows):
        rounds = split_sentences(case.get("context", ""))
        for qidx, q in enumerate(case.get("questions", [])):
            choices = q["choices"]
            answer_idx = int(q["answer"])
            out.append(
                {
                    "case_id": f"murder_mystery_{cidx}_q{qidx}",
                    "domain": "murder_mystery",
                    "question": q["question"],
                    "suspects": choices,
                    "gold_suspect": choices[answer_idx],
                    "gold_index": answer_idx,
                    "rounds": [
                        {"round_id": ridx + 1, "evidence_text": text}
                        for ridx, text in enumerate(rounds)
                    ],
                }
            )
    return out


def build_long_context(
    musr_rows: List[Dict[str, Any]],
    rng: random.Random,
    distractors: int,
    repeat_factor: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(musr_rows)
    for cidx, case in enumerate(musr_rows):
        pool = [i for i in range(n) if i != cidx]
        sampled_ids = rng.sample(pool, k=min(distractors, len(pool)))
        distractor_blocks = [musr_rows[i]["context"] for i in sampled_ids]
        distractor_text = "\n\n".join(distractor_blocks * max(1, repeat_factor))

        long_context = (
            "Background files from related investigations (some may be irrelevant):\n\n"
            f"{distractor_text}\n\n"
            "Target case file:\n\n"
            f"{case['context']}"
        )

        for qidx, q in enumerate(case.get("questions", [])):
            out.append(
                {
                    "id": f"long_context_{cidx}_q{qidx}",
                    "track": "long_context",
                    "context": long_context,
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer_index": int(q["answer"]),
                    "answer_text": q["choices"][int(q["answer"])],
                    "metadata": {
                        "source_domain": "murder_mystery",
                        "source_case_index": cidx,
                        "source_question_index": qidx,
                        "num_distractor_cases": len(sampled_ids),
                        "repeat_factor": repeat_factor,
                        "context_chars": len(long_context),
                    },
                }
            )
    return out



def build_dynamic_belief(
    musr_rows: List[Dict[str, Any]],
    rng: random.Random,
    setup_sentences: int,
    max_rounds: int,
    counterfactual_rate: float,
    flip_rate: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cidx, case in enumerate(musr_rows):
        for qidx, q in enumerate(case.get("questions", [])):
            out.append(
                build_dynamic_belief_case(
                    case=case,
                    case_idx=cidx,
                    question=q,
                    question_idx=qidx,
                    rng=rng,
                    setup_sentences=setup_sentences,
                    max_rounds=max_rounds,
                    counterfactual_rate=counterfactual_rate,
                    flip_rate=flip_rate,
                    track_name="dynamic_belief",
                    id_prefix="dynamic_belief",
                )
            )
    return out


def split_rows(rows: List[Dict[str, Any]], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(0.7 * n)
    n_dev = int(0.15 * n)
    train = rows[:n_train]
    dev = rows[n_train:n_train + n_dev]
    test = rows[n_train + n_dev:]
    return train, dev, test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CS422 v1 benchmark with long_context and dynamic_belief.")
    p.add_argument("--musr", type=Path, default=Path("datasets/murder_mystery.json"))
    p.add_argument("--stream", type=Path, default=Path("datasets_stream/murder_mystery_stream.jsonl"))
    p.add_argument("--outdir", type=Path, default=Path("benchmark_runs/cs422_v1"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of source cases.")
    p.add_argument("--distractors", type=int, default=2)
    p.add_argument("--repeat-factor", type=int, default=1)
    p.add_argument("--setup-sentences", type=int, default=2)
    p.add_argument("--max-rounds", type=int, default=40)
    p.add_argument("--counterfactual-rate", type=float, default=0.7)
    p.add_argument("--flip-rate", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    musr_rows = read_json(args.musr)
    if args.limit is not None:
        musr_rows = musr_rows[: args.limit]

    # Keep stream artifact loading for compatibility with existing workflow/debugging.
    if args.stream.exists():
        _ = read_jsonl(args.stream)
    else:
        _ = to_stream_from_musr(musr_rows)

    tracks = {
        "long_context": build_long_context(musr_rows, rng, args.distractors, args.repeat_factor),
        "dynamic_belief": build_dynamic_belief(
            musr_rows,
            rng,
            setup_sentences=args.setup_sentences,
            max_rounds=args.max_rounds,
            counterfactual_rate=args.counterfactual_rate,
            flip_rate=args.flip_rate,
        ),
    }

    manifest: Dict[str, Any] = {
        "name": "cs422_v2",
        "seed": args.seed,
        "source_dataset": str(args.musr),
        "limit": args.limit,
        "config": {
            "distractors": args.distractors,
            "repeat_factor": args.repeat_factor,
            "setup_sentences": args.setup_sentences,
            "max_rounds": args.max_rounds,
            "counterfactual_rate": args.counterfactual_rate,
            "flip_rate": args.flip_rate,
        },
        "tracks": {},
    }

    for track, rows in tracks.items():
        train, dev, test = split_rows(rows, rng)
        write_jsonl(args.outdir / track / "train.jsonl", train)
        write_jsonl(args.outdir / track / "dev.jsonl", dev)
        write_jsonl(args.outdir / track / "test.jsonl", test)
        track_meta: Dict[str, Any] = {
            "total": len(rows),
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        }
        if track == "dynamic_belief":
            track_meta.update(
                {
                    "has_counterfactual_count": sum(
                        1 for r in rows if r.get("metadata", {}).get("has_counterfactual")
                    ),
                    "stream_only_count": sum(
                        1 for r in rows if not r.get("metadata", {}).get("has_counterfactual")
                    ),
                    "counterfactual_flip_required_count": sum(
                        1
                        for r in rows
                        if r.get("metadata", {}).get("has_counterfactual")
                        and r.get("metadata", {}).get("flip_required")
                    ),
                    "counterfactual_flip_not_required_count": sum(
                        1
                        for r in rows
                        if r.get("metadata", {}).get("has_counterfactual")
                        and not r.get("metadata", {}).get("flip_required")
                    ),
                }
            )
        manifest["tracks"][track] = track_meta

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote benchmark to {args.outdir}")


if __name__ == "__main__":
    main()
