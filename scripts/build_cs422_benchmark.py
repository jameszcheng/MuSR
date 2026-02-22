import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def build_tom(musr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cidx, case in enumerate(musr_rows):
        for qidx, q in enumerate(case.get("questions", [])):
            out.append(
                {
                    "id": f"tom_{cidx}_q{qidx}",
                    "track": "tom",
                    "context": case["context"],
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer_index": int(q["answer"]),
                    "answer_text": q["choices"][int(q["answer"])],
                    "metadata": {
                        "source_domain": "murder_mystery",
                        "source_case_index": cidx,
                        "source_question_index": qidx,
                        "reasoning_tags": ["logical", "social", "tom"],
                    },
                }
            )
    return out


def build_dynamic_belief(stream_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in stream_rows:
        out.append(
            {
                "id": row["case_id"],
                "track": "dynamic_belief",
                "question": row["question"],
                "suspects": row["suspects"],
                "answer_index": int(row["gold_index"]),
                "answer_text": row["gold_suspect"],
                "rounds": row["rounds"],
                "metadata": {
                    "source_domain": row.get("domain", "murder_mystery"),
                    "num_rounds": len(row.get("rounds", [])),
                    "source_case_id": row["case_id"],
                },
            }
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


def main() -> None:
    p = argparse.ArgumentParser(description="Build CS422 benchmark starter tracks from MuSR files.")
    p.add_argument("--musr", type=Path, default=Path("datasets/murder_mystery.json"))
    p.add_argument("--stream", type=Path, default=Path("datasets_stream/murder_mystery_stream.jsonl"))
    p.add_argument("--outdir", type=Path, default=Path("benchmark_runs/cs422_v0"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--distractors", type=int, default=2)
    p.add_argument("--repeat-factor", type=int, default=1)
    args = p.parse_args()

    rng = random.Random(args.seed)

    musr_rows = read_json(args.musr)
    if args.stream.exists():
        stream_rows = read_jsonl(args.stream)
    else:
        stream_rows = to_stream_from_musr(musr_rows)

    tracks = {
        "long_context": build_long_context(musr_rows, rng, args.distractors, args.repeat_factor),
        "tom": build_tom(musr_rows),
        "dynamic_belief": build_dynamic_belief(stream_rows),
    }

    manifest: Dict[str, Any] = {
        "name": "cs422_v0",
        "seed": args.seed,
        "config": {
            "distractors": args.distractors,
            "repeat_factor": args.repeat_factor,
        },
        "tracks": {},
    }

    for track, rows in tracks.items():
        train, dev, test = split_rows(rows, rng)
        write_jsonl(args.outdir / track / "train.jsonl", train)
        write_jsonl(args.outdir / track / "dev.jsonl", dev)
        write_jsonl(args.outdir / track / "test.jsonl", test)
        manifest["tracks"][track] = {
            "total": len(rows),
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
        }

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote benchmark to {args.outdir}")


if __name__ == "__main__":
    main()
