import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c and c.strip()]


def chunk_sentences(sentences: List[str], mode: str, fixed_n: int) -> List[str]:
    if mode == "sentence":
        return sentences
    if mode == "fixed":
        out = []
        for idx in range(0, len(sentences), fixed_n):
            out.append(" ".join(sentences[idx:idx + fixed_n]).strip())
        return out
    raise ValueError(f"Unknown chunk mode: {mode}")


def convert_case(case: Dict, case_idx: int, chunk_mode: str, fixed_n: int) -> List[Dict]:
    context = case["context"]
    sentences = split_sentences(context)
    evidence_rounds = chunk_sentences(sentences, mode=chunk_mode, fixed_n=fixed_n)

    stream_cases: List[Dict] = []
    for qidx, question in enumerate(case.get("questions", [])):
        choices = question["choices"]
        answer_idx = int(question["answer"])
        metadata = {}

        inter_data = question.get("intermediate_data") or []
        if len(inter_data) > 0 and isinstance(inter_data[0], dict):
            md = inter_data[0]
            metadata["story_hash_id"] = md.get("story_hash_id")
            metadata["using_counterfactuals"] = md.get("using_counterfactuals")
            metadata["has_counterfactual_variant"] = md.get("cf_with_negated_mmo") is not None

        stream_cases.append(
            {
                "case_id": f"murder_mystery_{case_idx}_q{qidx}",
                "domain": "murder_mystery",
                "question": question["question"],
                "suspects": choices,
                "gold_suspect": choices[answer_idx],
                "gold_index": answer_idx,
                "metadata": {
                    "source_case_index": case_idx,
                    "source_question_index": qidx,
                    "chunk_mode": chunk_mode,
                    "n_rounds": len(evidence_rounds),
                    **metadata,
                },
                "rounds": [
                    {"round_id": ridx + 1, "evidence_text": evidence}
                    for ridx, evidence in enumerate(evidence_rounds)
                ],
            }
        )
    return stream_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MuSR murder mysteries into streamed evidence rounds.")
    parser.add_argument("--input", type=Path, default=Path("datasets/murder_mystery.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets_stream/murder_mystery_stream.jsonl"))
    parser.add_argument("--chunk-mode", choices=["sentence", "fixed"], default="sentence")
    parser.add_argument("--fixed-n", type=int, default=3, help="Sentences per chunk when --chunk-mode fixed.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of source cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit is not None:
        data = data[: args.limit]

    out_cases: List[Dict] = []
    for idx, case in enumerate(data):
        out_cases.extend(convert_case(case, idx, args.chunk_mode, args.fixed_n))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for case in out_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_cases)} streamed cases to {args.output}")


if __name__ == "__main__":
    main()
