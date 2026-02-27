import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c and c.strip()]


def iter_explicit_leaf_facts(node: Dict[str, Any]) -> Iterable[str]:
    children = node.get("children", [])
    if children:
        for child in children:
            yield from iter_explicit_leaf_facts(child)
        return

    if node.get("fact_type") == "explicit":
        value = str(node.get("value", "")).strip()
        if value:
            yield value


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def extract_choice_facts(question: Dict[str, Any]) -> List[List[str]]:
    facts_per_choice: List[List[str]] = []
    for tree_json in question.get("intermediate_trees", []):
        nodes = tree_json.get("nodes", [])
        if not nodes:
            facts_per_choice.append([])
            continue
        root = nodes[0]
        facts = ordered_unique(iter_explicit_leaf_facts(root))
        facts_per_choice.append(facts)
    return facts_per_choice


def split_rows(
    rows: List[Dict[str, Any]], rng: random.Random
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(0.7 * n)
    n_dev = int(0.15 * n)
    train = rows[:n_train]
    dev = rows[n_train : n_train + n_dev]
    test = rows[n_train + n_dev :]
    return train, dev, test


def select_counterfactual_target(
    choices: List[str], base_answer_index: int, rng: random.Random, flip_rate: float
) -> Tuple[int, bool]:
    if len(choices) <= 1:
        return base_answer_index, False

    flip_required = rng.random() < flip_rate
    if not flip_required:
        return base_answer_index, False

    pool = [idx for idx in range(len(choices)) if idx != base_answer_index]
    return rng.choice(pool), True


def build_counterfactual_rounds(
    before_suspect: str, after_suspect: str, flip_required: bool, start_round_id: int
) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []

    if flip_required:
        cf_a = (
            f"Counterfactual update: A corrected report invalidates the strongest earlier evidence "
            f"against {before_suspect}."
        )
        cf_b = (
            f"Counterfactual update: New evidence now best supports means, motive, and opportunity "
            f"for {after_suspect}."
        )
    else:
        cf_a = (
            f"Counterfactual update: A previously suspicious clue about {after_suspect} is ruled out "
            "as irrelevant."
        )
        cf_b = (
            f"Counterfactual update: Revised evidence still best supports means, motive, and opportunity "
            f"for {before_suspect}."
        )

    rounds.append(
        {
            "round_id": start_round_id,
            "evidence_text": cf_a,
            "evidence_type": "counterfactual_update",
            "source_choice_index": None,
            "is_counterfactual": True,
        }
    )
    rounds.append(
        {
            "round_id": start_round_id + 1,
            "evidence_text": cf_b,
            "evidence_type": "counterfactual_update",
            "source_choice_index": None,
            "is_counterfactual": True,
        }
    )
    return rounds


def build_dynamic_belief_case(
    case: Dict[str, Any],
    case_idx: int,
    question: Dict[str, Any],
    question_idx: int,
    rng: random.Random,
    setup_sentences: int,
    max_rounds: int,
    counterfactual_rate: float,
    flip_rate: float,
    track_name: str,
    id_prefix: str,
) -> Dict[str, Any]:
    choices = question["choices"]
    base_answer_idx = int(question["answer"])
    has_counterfactual = rng.random() < counterfactual_rate

    if has_counterfactual:
        cf_answer_idx, flip_required = select_counterfactual_target(
            choices, base_answer_idx, rng, flip_rate=flip_rate
        )
    else:
        cf_answer_idx = base_answer_idx
        flip_required = False

    facts_per_choice = extract_choice_facts(question)
    setup = split_sentences(case.get("context", ""))[: max(0, setup_sentences)]

    rounds: List[Dict[str, Any]] = []
    rid = 1
    for sent in setup:
        rounds.append(
            {
                "round_id": rid,
                "evidence_text": sent,
                "evidence_type": "setup_sentence",
                "source_choice_index": None,
                "is_counterfactual": False,
            }
        )
        rid += 1

    pointers = [0 for _ in facts_per_choice]
    cf_round_start = None
    cf_insert_round = max(2, min(max_rounds - 2, max(3, max_rounds // 2)))

    while len(rounds) < max_rounds and any(
        pointers[cidx] < len(facts_per_choice[cidx])
        for cidx in range(len(facts_per_choice))
    ):
        for cidx in range(len(facts_per_choice)):
            if len(rounds) >= max_rounds:
                break
            if has_counterfactual and cf_round_start is None and len(rounds) >= cf_insert_round:
                rounds.extend(
                    build_counterfactual_rounds(
                        before_suspect=choices[base_answer_idx],
                        after_suspect=choices[cf_answer_idx],
                        flip_required=flip_required,
                        start_round_id=rid,
                    )
                )
                cf_round_start = rid
                rid += 2
                if len(rounds) >= max_rounds:
                    break
            if pointers[cidx] >= len(facts_per_choice[cidx]):
                continue
            fact = facts_per_choice[cidx][pointers[cidx]]
            pointers[cidx] += 1
            rounds.append(
                {
                    "round_id": rid,
                    "evidence_text": fact,
                    "evidence_type": "tree_fact",
                    "source_choice_index": cidx,
                    "is_counterfactual": False,
                }
            )
            rid += 1

    # Ensure counterfactual cases always include one update block.
    if has_counterfactual and cf_round_start is None and len(rounds) <= max_rounds - 2:
        cf_round_start = rid
        rounds.extend(
            build_counterfactual_rounds(
                before_suspect=choices[base_answer_idx],
                after_suspect=choices[cf_answer_idx],
                flip_required=flip_required,
                start_round_id=rid,
            )
        )
        rid += 2

    rounds = rounds[:max_rounds]

    # Reconcile metadata/labels with what was actually emitted after truncation.
    emitted_cf_round_ids = [
        int(r["round_id"])
        for r in rounds
        if bool(r.get("is_counterfactual")) and r.get("evidence_type") == "counterfactual_update"
    ]
    emitted_setup_count = sum(1 for r in rounds if r.get("evidence_type") == "setup_sentence")
    emitted_has_counterfactual = len(emitted_cf_round_ids) > 0
    if not emitted_has_counterfactual:
        cf_answer_idx = base_answer_idx
        flip_required = False
        cf_round_start = None
    else:
        cf_round_start = min(emitted_cf_round_ids)

    inter_data = (question.get("intermediate_data") or [{}])[0]
    return {
        "id": f"{id_prefix}_{case_idx}_q{question_idx}",
        "track": track_name,
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
            "update_type": "counterfactual" if emitted_has_counterfactual else "stream_only",
            "has_counterfactual": emitted_has_counterfactual,
            "flip_required": flip_required,
            "gold_before_index": base_answer_idx,
            "gold_before_text": choices[base_answer_idx],
            "gold_after_index": cf_answer_idx,
            "gold_after_text": choices[cf_answer_idx],
            "counterfactual_round_id": cf_round_start,
            "counterfactual_round_count": len(emitted_cf_round_ids),
            "n_rounds": len(rounds),
            "setup_sentence_count": emitted_setup_count,
            "facts_per_choice": [len(x) for x in facts_per_choice],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dynamic belief benchmark with streamed evidence and optional counterfactual updates."
        )
    )
    parser.add_argument("--musr", type=Path, default=Path("datasets/murder_mystery.json"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmark_runs/cs422_v1/dynamic_belief"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--setup-sentences", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument(
        "--counterfactual-rate",
        type=float,
        default=0.7,
        help="Probability that a case contains a counterfactual update block.",
    )
    parser.add_argument(
        "--flip-rate",
        type=float,
        default=0.7,
        help="Among counterfactual cases, probability that final gold answer flips.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    musr_rows: List[Dict[str, Any]] = read_json(args.musr)
    if args.limit is not None:
        musr_rows = musr_rows[: args.limit]

    track_name = "dynamic_belief"
    id_prefix = "dynamic_belief"

    rows: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(musr_rows):
        for qidx, question in enumerate(case.get("questions", [])):
            rows.append(
                build_dynamic_belief_case(
                    case=case,
                    case_idx=case_idx,
                    question=question,
                    question_idx=qidx,
                    rng=rng,
                    setup_sentences=args.setup_sentences,
                    max_rounds=args.max_rounds,
                    counterfactual_rate=args.counterfactual_rate,
                    flip_rate=args.flip_rate,
                    track_name=track_name,
                    id_prefix=id_prefix,
                )
            )

    train, dev, test = split_rows(rows, rng)
    write_jsonl(args.outdir / "train.jsonl", train)
    write_jsonl(args.outdir / "dev.jsonl", dev)
    write_jsonl(args.outdir / "test.jsonl", test)

    manifest: Dict[str, Any] = {
        "name": "dynamic_belief_v1",
        "source_dataset": str(args.musr),
        "seed": args.seed,
        "limit": args.limit,
        "track_name": track_name,
        "config": {
            "setup_sentences": args.setup_sentences,
            "max_rounds": args.max_rounds,
            "counterfactual_rate": args.counterfactual_rate,
            "flip_rate": args.flip_rate,
            "id_prefix": id_prefix,
        },
        "tracks": {
            track_name: {
                "total": len(rows),
                "train": len(train),
                "dev": len(dev),
                "test": len(test),
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
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    print(f"Wrote dynamic belief benchmark to {args.outdir}")


if __name__ == "__main__":
    main()
