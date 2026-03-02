import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── I/O helpers ────────────────────────────────────────────────────────────────

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n\n+", text.strip())
    return [c.strip() for c in chunks if c and c.strip()]


def split_rows(
    rows: List[Dict[str, Any]], rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(0.7 * n)
    n_dev = int(0.15 * n)
    return rows[:n_train], rows[n_train:n_train + n_dev], rows[n_train + n_dev:]


# ── Long context track ─────────────────────────────────────────────────────────

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
        distractor_text = "\n\n".join(
            [musr_rows[i]["context"] for i in sampled_ids] * max(1, repeat_factor)
        )
        long_context = (
            "Background files from related investigations (some may be irrelevant):\n\n"
            f"{distractor_text}\n\n"
            "Target case file:\n\n"
            f"{case['context']}"
        )
        for qidx, q in enumerate(case.get("questions", [])):
            out.append({
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
            })
    return out


# ── Dynamic belief track ───────────────────────────────────────────────────────

def select_counterfactual_target(
    choices: List[str], base_answer_index: int, rng: random.Random, flip_rate: float
) -> Tuple[int, bool]:
    if len(choices) <= 1:
        return base_answer_index, False
    if rng.random() >= flip_rate:
        return base_answer_index, False
    pool = [i for i in range(len(choices)) if i != base_answer_index]
    return rng.choice(pool), True


def build_counterfactual_rounds(
    before_suspect: str, after_suspect: str, flip_required: bool, start_round_id: int
) -> List[Dict[str, Any]]:
    if flip_required:
        cf_a = (
            f"Correction: The key witness timeline previously used against {before_suspect} was entered "
            f"with an incorrect timestamp and is now withdrawn."
        )
        cf_b = (
            f"Follow-up records now place {after_suspect}'s phone near the scene during the murder window, "
            f"show a recent purchase of the same weapon model, and include messages documenting escalating "
            f"conflict with the victim."
        )
    else:
        cf_a = (
            f"Correction: A previously suspicious clue about {after_suspect} was misattributed after lab "
            f"review and is now ruled irrelevant."
        )
        cf_b = (
            f"Additional checks confirm {before_suspect}'s phone location near the scene during the murder "
            f"window, a matching weapon purchase receipt, and recent threatening messages to the victim."
        )
    return [
        {"round_id": start_round_id,     "evidence_text": cf_a, "evidence_type": "counterfactual_update", "is_counterfactual": True},
        {"round_id": start_round_id + 1, "evidence_text": cf_b, "evidence_type": "counterfactual_update", "is_counterfactual": True},
    ]


def build_dynamic_belief_case(
    case: Dict[str, Any],
    case_idx: int,
    question: Dict[str, Any],
    question_idx: int,
    rng: random.Random,
    counterfactual_rate: float,
    flip_rate: float,
) -> Dict[str, Any]:
    choices = question["choices"]
    base_answer_idx = int(question["answer"])
    has_counterfactual = rng.random() < counterfactual_rate

    if has_counterfactual:
        cf_answer_idx, flip_required = select_counterfactual_target(choices, base_answer_idx, rng, flip_rate)
    else:
        cf_answer_idx, flip_required = base_answer_idx, False

    paragraphs = split_paragraphs(case.get("context", ""))

    rounds: List[Dict[str, Any]] = []
    rid = 1
    cf_round_start = None

    for para in paragraphs:
        rounds.append({"round_id": rid, "evidence_text": para, "evidence_type": "narrative_paragraph", "is_counterfactual": False})
        rid += 1

    if has_counterfactual:
        rounds.extend(build_counterfactual_rounds(choices[base_answer_idx], choices[cf_answer_idx], flip_required, rid))
        cf_round_start = rid

    emitted_cf_ids = [
        int(r["round_id"]) for r in rounds
        if r.get("is_counterfactual") and r.get("evidence_type") == "counterfactual_update"
    ]
    has_cf = len(emitted_cf_ids) > 0
    if not has_cf:
        cf_answer_idx, flip_required, cf_round_start = base_answer_idx, False, None
    else:
        cf_round_start = min(emitted_cf_ids)

    inter_data = (question.get("intermediate_data") or [{}])[0]
    return {
        "id": f"dynamic_belief_{case_idx}_q{question_idx}",
        "track": "dynamic_belief",
        "question": question["question"],
        "suspects": choices,
        "answer_index": cf_answer_idx,
        "answer_text": choices[cf_answer_idx],
        "rounds": rounds,
        "metadata": {
            "source_domain": "murder_mystery",
            "source_case_index": case_idx,
            "source_question_index": question_idx,
            "source_story_hash_id": inter_data.get("story_hash_id"),
            "update_type": "counterfactual" if has_cf else "stream_only",
            "has_counterfactual": has_cf,
            "flip_required": flip_required,
            "gold_before_index": base_answer_idx,
            "gold_before_text": choices[base_answer_idx],
            "gold_after_index": cf_answer_idx,
            "gold_after_text": choices[cf_answer_idx],
            "counterfactual_round_id": cf_round_start,
            "counterfactual_round_count": len(emitted_cf_ids),
            "n_rounds": len(rounds),
            "n_narrative_paragraphs": len(paragraphs),
        },
    }


def build_dynamic_belief(
    musr_rows: List[Dict[str, Any]],
    rng: random.Random,
    counterfactual_rate: float,
    flip_rate: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cidx, case in enumerate(musr_rows):
        for qidx, q in enumerate(case.get("questions", [])):
            out.append(build_dynamic_belief_case(case, cidx, q, qidx, rng, counterfactual_rate, flip_rate))
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CS422 v3 benchmark (long_context + dynamic_belief, CF at end).")
    p.add_argument("--musr", type=Path, default=Path("datasets/murder_mystery.json"))
    p.add_argument("--outdir", type=Path, default=Path("benchmark_runs/cs422_v3"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--distractors", type=int, default=2)
    p.add_argument("--repeat-factor", type=int, default=1)
    p.add_argument("--counterfactual-rate", type=float, default=0.7)
    p.add_argument("--flip-rate", type=float, default=0.7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    musr_rows = read_json(args.musr)
    if args.limit is not None:
        musr_rows = musr_rows[:args.limit]

    tracks = {
        "long_context": build_long_context(musr_rows, rng, args.distractors, args.repeat_factor),
        "dynamic_belief": build_dynamic_belief(musr_rows, rng, args.counterfactual_rate, args.flip_rate),
    }

    manifest: Dict[str, Any] = {
        "name": "cs422_v3",
        "seed": args.seed,
        "source_dataset": str(args.musr),
        "limit": args.limit,
        "config": {
            "distractors": args.distractors,
            "repeat_factor": args.repeat_factor,
            "counterfactual_rate": args.counterfactual_rate,
            "flip_rate": args.flip_rate,
            "evidence_source": "narrative_paragraphs",
        },
        "tracks": {},
    }

    for track, rows in tracks.items():
        train, dev, test = split_rows(rows, rng)
        write_jsonl(args.outdir / track / "train.jsonl", train)
        write_jsonl(args.outdir / track / "dev.jsonl", dev)
        write_jsonl(args.outdir / track / "test.jsonl", test)
        track_meta: Dict[str, Any] = {"total": len(rows), "train": len(train), "dev": len(dev), "test": len(test)}
        if track == "dynamic_belief":
            track_meta.update({
                "has_counterfactual_count": sum(1 for r in rows if r["metadata"]["has_counterfactual"]),
                "stream_only_count": sum(1 for r in rows if not r["metadata"]["has_counterfactual"]),
                "counterfactual_flip_required_count": sum(1 for r in rows if r["metadata"]["has_counterfactual"] and r["metadata"]["flip_required"]),
                "counterfactual_flip_not_required_count": sum(1 for r in rows if r["metadata"]["has_counterfactual"] and not r["metadata"]["flip_required"]),
            })
        manifest["tracks"][track] = track_meta

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote benchmark to {args.outdir}")


if __name__ == "__main__":
    main()
