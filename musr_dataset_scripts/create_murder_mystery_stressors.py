import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def build_long_context_case(
    base_cases: List[Dict[str, Any]],
    case_idx: int,
    question_idx: int,
    distractor_count: int,
    placement: str,
    repeat_factor: int,
    rng: random.Random,
) -> Dict[str, Any]:
    case = base_cases[case_idx]
    question = case["questions"][question_idx]

    pool = [i for i in range(len(base_cases)) if i != case_idx]
    sampled = rng.sample(pool, k=min(distractor_count, len(pool)))
    distractors = [base_cases[i]["context"] for i in sampled]

    distractor_blocks = distractors * max(1, repeat_factor)
    if placement == "prefix":
        context = (
            "Background files from related investigations (some may be irrelevant):\n\n"
            + "\n\n".join(distractor_blocks)
            + "\n\nTarget case file:\n\n"
            + case["context"]
        )
    else:
        target_sents = split_sentences(case["context"])
        injected: List[str] = []
        if not target_sents:
            target_sents = [case["context"]]
        stride = max(1, len(target_sents) // (len(distractor_blocks) + 1))
        dptr = 0
        for sidx, sentence in enumerate(target_sents):
            injected.append(sentence)
            if dptr < len(distractor_blocks) and (sidx + 1) % stride == 0:
                injected.append(f"[Distractor Evidence]\n{distractor_blocks[dptr]}")
                dptr += 1
        while dptr < len(distractor_blocks):
            injected.append(f"[Distractor Evidence]\n{distractor_blocks[dptr]}")
            dptr += 1
        context = "\n\n".join(injected)

    answer_idx = int(question["answer"])
    choices = question["choices"]
    return {
        "id": f"mm_long_context_{case_idx}_q{question_idx}",
        "domain": "murder_mystery",
        "stressor": "long_context",
        "question": question["question"],
        "choices": choices,
        "answer_index": answer_idx,
        "answer_text": choices[answer_idx],
        "context": context,
        "metadata": {
            "source_case_index": case_idx,
            "source_question_index": question_idx,
            "source_story_hash_id": (
                (question.get("intermediate_data") or [{}])[0].get("story_hash_id")
                if question.get("intermediate_data")
                else None
            ),
            "distractor_count": len(sampled),
            "distractor_case_indices": sampled,
            "placement": placement,
            "repeat_factor": repeat_factor,
            "context_chars": len(context),
        },
    }


def build_dynamic_updates_case(
    base_cases: List[Dict[str, Any]],
    case_idx: int,
    question_idx: int,
    max_rounds: int,
    setup_sentences: int,
) -> Dict[str, Any]:
    case = base_cases[case_idx]
    question = case["questions"][question_idx]
    choices = question["choices"]
    answer_idx = int(question["answer"])

    facts_per_choice = extract_choice_facts(question)
    setup = split_sentences(case["context"])[: max(0, setup_sentences)]

    rounds: List[Dict[str, Any]] = []
    pivot_round_ids: List[int] = []

    rid = 1
    for sent in setup:
        rounds.append(
            {
                "round_id": rid,
                "evidence_text": sent,
                "evidence_type": "setup_sentence",
                "source_choice_index": None,
                "is_pivot": False,
            }
        )
        rid += 1

    pointers = [0 for _ in facts_per_choice]
    produced_from_choice = [0 for _ in facts_per_choice]

    while len(rounds) < max_rounds and any(
        pointers[cidx] < len(facts_per_choice[cidx]) for cidx in range(len(facts_per_choice))
    ):
        for cidx in range(len(facts_per_choice)):
            if len(rounds) >= max_rounds:
                break
            if pointers[cidx] >= len(facts_per_choice[cidx]):
                continue

            fact = facts_per_choice[cidx][pointers[cidx]]
            pointers[cidx] += 1
            produced_from_choice[cidx] += 1

            is_pivot = cidx == answer_idx and produced_from_choice[cidx] > max(0, len(facts_per_choice[cidx]) - 3)
            rounds.append(
                {
                    "round_id": rid,
                    "evidence_text": fact,
                    "evidence_type": "tree_fact",
                    "source_choice_index": cidx,
                    "is_pivot": bool(is_pivot),
                }
            )
            if is_pivot:
                pivot_round_ids.append(rid)
            rid += 1

    return {
        "id": f"mm_dynamic_updates_{case_idx}_q{question_idx}",
        "domain": "murder_mystery",
        "stressor": "dynamic_updates",
        "question": question["question"],
        "choices": choices,
        "suspects": choices,
        "answer_index": answer_idx,
        "answer_text": choices[answer_idx],
        "rounds": rounds,
        "metadata": {
            "source_case_index": case_idx,
            "source_question_index": question_idx,
            "source_story_hash_id": (
                (question.get("intermediate_data") or [{}])[0].get("story_hash_id")
                if question.get("intermediate_data")
                else None
            ),
            "round_construction": "setup_sentences_plus_ordered_tree_facts",
            "setup_sentence_count": len(setup),
            "n_rounds": len(rounds),
            "pivot_round_ids": pivot_round_ids,
            "facts_per_choice": [len(x) for x in facts_per_choice],
        },
    }


def build_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    base_cases: List[Dict[str, Any]] = read_json(args.input)
    if args.limit is not None:
        base_cases = base_cases[: args.limit]

    rng = random.Random(args.seed)

    long_rows: List[Dict[str, Any]] = []
    dynamic_rows: List[Dict[str, Any]] = []

    for case_idx, case in enumerate(base_cases):
        for qidx, _ in enumerate(case.get("questions", [])):
            if args.stressor in {"long_context", "both"}:
                long_rows.append(
                    build_long_context_case(
                        base_cases=base_cases,
                        case_idx=case_idx,
                        question_idx=qidx,
                        distractor_count=args.distractor_count,
                        placement=args.placement,
                        repeat_factor=args.repeat_factor,
                        rng=rng,
                    )
                )
            if args.stressor in {"dynamic_updates", "both"}:
                dynamic_rows.append(
                    build_dynamic_updates_case(
                        base_cases=base_cases,
                        case_idx=case_idx,
                        question_idx=qidx,
                        max_rounds=args.max_rounds,
                        setup_sentences=args.setup_sentences,
                    )
                )

    return {
        "long_context": long_rows,
        "dynamic_updates": dynamic_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create murder mystery stressor datasets with first-class long-context or dynamic-update structure."
    )
    parser.add_argument("--input", type=Path, default=Path("datasets/murder_mystery.json"))
    parser.add_argument(
        "--stressor",
        choices=["long_context", "dynamic_updates", "both"],
        default="both",
    )
    parser.add_argument("--outdir", type=Path, default=Path("datasets_stressors/murder_mystery"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--distractor-count", type=int, default=2)
    parser.add_argument("--placement", choices=["prefix", "interleave"], default="prefix")
    parser.add_argument("--repeat-factor", type=int, default=1)

    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--setup-sentences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    built = build_rows(args)

    manifest = {
        "domain": "murder_mystery",
        "input": str(args.input),
        "stressor": args.stressor,
        "seed": args.seed,
        "limit": args.limit,
        "config": {
            "distractor_count": args.distractor_count,
            "placement": args.placement,
            "repeat_factor": args.repeat_factor,
            "max_rounds": args.max_rounds,
            "setup_sentences": args.setup_sentences,
        },
        "counts": {},
    }

    if args.stressor in {"long_context", "both"}:
        out = args.outdir / "long_context.jsonl"
        write_jsonl(out, built["long_context"])
        manifest["counts"]["long_context"] = len(built["long_context"])

    if args.stressor in {"dynamic_updates", "both"}:
        out = args.outdir / "dynamic_updates.jsonl"
        write_jsonl(out, built["dynamic_updates"])
        manifest["counts"]["dynamic_updates"] = len(built["dynamic_updates"])

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote stressor datasets to {args.outdir}")


if __name__ == "__main__":
    main()
