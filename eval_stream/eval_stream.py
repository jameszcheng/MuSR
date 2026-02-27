import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src import cache
from src.model import TogetherModel
from src.model.model import extract_text_from_response


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def case_id_of(case: Dict[str, Any]) -> str:
    return str(case.get("case_id", case.get("id", "unknown_case")))


def gold_suspect_of(case: Dict[str, Any]) -> str:
    if "gold_suspect" in case:
        return str(case["gold_suspect"])
    if "answer_text" in case:
        return str(case["answer_text"])
    suspects = case.get("suspects", [])
    answer_index = case.get("answer_index", 0)
    if isinstance(suspects, list) and suspects and isinstance(answer_index, int) and 0 <= answer_index < len(suspects):
        return str(suspects[answer_index])
    return ""


def counterfactual_info(case: Dict[str, Any]) -> Dict[str, Any]:
    md = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
    has_cf = bool(md.get("has_counterfactual", False))
    cf_round_id = md.get("counterfactual_round_id")
    if not has_cf:
        cf_round_id = None
    return {
        "update_type": md.get("update_type", "stream_only"),
        "has_counterfactual": has_cf,
        "flip_required": bool(md.get("flip_required", False)),
        "counterfactual_round_id": cf_round_id,
        "gold_before_text": md.get("gold_before_text"),
        "gold_after_text": md.get("gold_after_text", gold_suspect_of(case)),
    }


def build_prompt(case: Dict[str, Any], revealed_evidence: List[str]) -> str:
    suspects = "\n".join(f"- {s}" for s in case["suspects"])
    evidence = "\n".join(f"{idx + 1}. {x}" for idx, x in enumerate(revealed_evidence))
    return f"""You are evaluating a murder mystery as evidence arrives.

Question: {case["question"]}
Suspects:
{suspects}

Evidence so far:
{evidence}

Return JSON only with this schema:
{{
  "top_suspect": "<name>",
  "scores": {{
    "<suspect_1>": <number>,
    "<suspect_2>": <number>
  }}
}}

Rules:
- Include every suspect exactly once in "scores".
- Scores can be any non-negative numbers.
- "top_suspect" must be one of the suspects.
""".strip()


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def parse_scores_text_fallback(text: str, suspects: List[str]) -> Tuple[Optional[str], Dict[str, float]]:
    top = None
    scores: Dict[str, float] = {}

    top_match = re.search(r"top[_\s-]*suspect\s*[:=]\s*([^\n,]+)", text, re.IGNORECASE)
    if top_match:
        top = top_match.group(1).strip().strip('"').strip("'")

    for s in suspects:
        pattern = re.escape(s) + r"\s*[:=]\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            scores[s] = float(m.group(1))
    return top, scores


def normalize_scores(scores: Dict[str, float], suspects: List[str]) -> Dict[str, float]:
    vec = {s: max(0.0, float(scores.get(s, 0.0))) for s in suspects}
    total = sum(vec.values())
    if total <= 0:
        p = 1.0 / max(1, len(suspects))
        return {s: p for s in suspects}
    return {s: v / total for s, v in vec.items()}


def parse_model_belief(raw_text: str, suspects: List[str]) -> Dict[str, Any]:
    parsed = try_parse_json(raw_text)
    top_suspect = None
    raw_scores: Dict[str, float] = {}

    if parsed:
        maybe_top = parsed.get("top_suspect")
        if isinstance(maybe_top, str):
            top_suspect = maybe_top.strip()

        maybe_scores = parsed.get("scores", {})
        if isinstance(maybe_scores, dict):
            lowered_map = {k.lower(): k for k in suspects}
            for k, v in maybe_scores.items():
                if not isinstance(k, str):
                    continue
                canonical = lowered_map.get(k.lower())
                if canonical is None:
                    continue
                try:
                    raw_scores[canonical] = float(v)
                except (TypeError, ValueError):
                    continue
    else:
        top_suspect, raw_scores = parse_scores_text_fallback(raw_text, suspects)

    probs = normalize_scores(raw_scores, suspects)
    if top_suspect not in suspects:
        top_suspect = max(probs.items(), key=lambda kv: kv[1])[0]

    return {
        "top_suspect": top_suspect,
        "raw_scores": raw_scores,
        "probabilities": probs,
    }


def brier_score(probabilities: Dict[str, float], gold_suspect: str, suspects: List[str]) -> float:
    out = 0.0
    for s in suspects:
        y = 1.0 if s == gold_suspect else 0.0
        p = float(probabilities.get(s, 0.0))
        out += (p - y) ** 2
    return out


def update_consistency(round_preds: List[Dict[str, Any]]) -> float:
    if len(round_preds) <= 1:
        return 1.0

    tops = [r["top_suspect"] for r in round_preds]
    flips = sum(1 for i in range(1, len(tops)) if tops[i] != tops[i - 1])
    flip_penalty = flips / (len(tops) - 1)

    final_top = tops[-1]
    traj = [float(r["probabilities"].get(final_top, 0.0)) for r in round_preds]
    decreases = sum(max(0.0, traj[i - 1] - traj[i]) for i in range(1, len(traj)))
    trajectory_penalty = decreases / (len(traj) - 1)

    score = 1.0 - 0.5 * (flip_penalty + trajectory_penalty)
    return max(0.0, min(1.0, score))


def flip_when_required(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    if not cf_info["has_counterfactual"] or not cf_info["flip_required"]:
        return None

    cf_round_id = cf_info["counterfactual_round_id"]
    try:
        cf_round_id = int(cf_round_id)
    except (TypeError, ValueError):
        return None
    if cf_round_id is None:
        return None

    pre = [r for r in round_preds if r["round_id"] < cf_round_id]
    post = [r for r in round_preds if r["round_id"] >= cf_round_id]
    if not pre or not post:
        return None

    pre_top = pre[-1]["top_suspect"]
    post_final_top = post[-1]["top_suspect"]
    gold_after = cf_info.get("gold_after_text")
    if not gold_after:
        return None
    return float(pre_top != post_final_top and post_final_top == gold_after)


def stability_when_not_required(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    if not cf_info["has_counterfactual"] or cf_info["flip_required"]:
        return None

    cf_round_id = cf_info["counterfactual_round_id"]
    try:
        cf_round_id = int(cf_round_id)
    except (TypeError, ValueError):
        return None
    if cf_round_id is None:
        return None

    pre = [r for r in round_preds if r["round_id"] < cf_round_id]
    post = [r for r in round_preds if r["round_id"] >= cf_round_id]
    if not pre or not post:
        return None

    pre_top = pre[-1]["top_suspect"]
    post_final_top = post[-1]["top_suspect"]
    return float(pre_top == post_final_top)


def recovery_latency(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    if not cf_info["has_counterfactual"] or not cf_info["flip_required"]:
        return None

    cf_round_id = cf_info["counterfactual_round_id"]
    gold_after = cf_info["gold_after_text"]
    try:
        cf_round_id = int(cf_round_id)
    except (TypeError, ValueError):
        return None
    if not gold_after:
        return None

    for r in round_preds:
        if r["round_id"] >= cf_round_id and r["top_suspect"] == gold_after:
            return float(r["round_id"] - cf_round_id)
    return None


def evaluate_case(model: TogetherModel, case: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    revealed: List[str] = []
    round_preds: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    case_id = case_id_of(case)
    gold_suspect = gold_suspect_of(case)
    cf_info = counterfactual_info(case)

    rounds_iter = tqdm(case["rounds"], desc=f"{case_id} rounds", leave=False)
    for rnd in rounds_iter:
        revealed.append(rnd["evidence_text"])
        prompt = build_prompt(case, revealed)
        raw = model.inference(prompt)
        output = extract_text_from_response(raw)
        belief = parse_model_belief(output, case["suspects"])
        belief["round_id"] = int(rnd["round_id"])
        round_preds.append(belief)
        rounds_iter.set_postfix({"top": belief["top_suspect"]})
        if verbose:
            probs_str = ", ".join(f"{k}={v:.3f}" for k, v in belief["probabilities"].items())
            print(f"[{case_id} r{rnd['round_id']}] top={belief['top_suspect']} | probs: {probs_str}")
        traces.append(
            {
                "round_id": rnd["round_id"],
                "prompt": prompt,
                "raw_output": output,
                "parsed": belief,
            }
        )

    final_pred = round_preds[-1]["top_suspect"] if round_preds else None
    final_correct = int(final_pred == gold_suspect)
    round_briers = [
        brier_score(r["probabilities"], gold_suspect, case["suspects"])
        for r in round_preds
    ]

    return {
        "case_id": case_id,
        "gold_suspect": gold_suspect,
        "final_top_suspect": final_pred,
        "final_correct": final_correct,
        "counterfactual": cf_info,
        "metrics": {
            "final_accuracy": final_correct,
            "update_consistency": update_consistency(round_preds),
            "brier_final": round_briers[-1] if round_briers else None,
            "brier_mean": (sum(round_briers) / len(round_briers)) if round_briers else None,
            "flip_when_required": flip_when_required(round_preds, cf_info),
            "stability_when_not_required": stability_when_not_required(round_preds, cf_info),
            "recovery_latency": recovery_latency(round_preds, cf_info),
        },
        "round_predictions": round_preds,
        "round_traces": traces,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "n_cases": 0,
            "final_accuracy": None,
            "update_consistency": None,
            "brier_final": None,
            "brier_mean": None,
            "flip_when_required": None,
            "stability_when_not_required": None,
            "recovery_latency": None,
            "counterfactual_cases": 0,
            "stream_only_cases": 0,
        }

    n = len(results)
    cf_results = [r for r in results if r.get("counterfactual", {}).get("has_counterfactual")]
    stream_only_results = [r for r in results if not r.get("counterfactual", {}).get("has_counterfactual")]
    fwr = [r["metrics"]["flip_when_required"] for r in results if r["metrics"]["flip_when_required"] is not None]
    swnr = [
        r["metrics"]["stability_when_not_required"]
        for r in results
        if r["metrics"]["stability_when_not_required"] is not None
    ]
    lat = [r["metrics"]["recovery_latency"] for r in results if r["metrics"]["recovery_latency"] is not None]

    return {
        "n_cases": n,
        "final_accuracy": sum(r["metrics"]["final_accuracy"] for r in results) / n,
        "update_consistency": sum(r["metrics"]["update_consistency"] for r in results) / n,
        "brier_final": sum(r["metrics"]["brier_final"] for r in results if r["metrics"]["brier_final"] is not None)
        / max(1, sum(1 for r in results if r["metrics"]["brier_final"] is not None)),
        "brier_mean": sum(r["metrics"]["brier_mean"] for r in results if r["metrics"]["brier_mean"] is not None)
        / max(1, sum(1 for r in results if r["metrics"]["brier_mean"] is not None)),
        "flip_when_required": (sum(fwr) / len(fwr)) if fwr else None,
        "stability_when_not_required": (sum(swnr) / len(swnr)) if swnr else None,
        "recovery_latency": (sum(lat) / len(lat)) if lat else None,
        "counterfactual_cases": len(cf_results),
        "stream_only_cases": len(stream_only_results),
        "subsets": {
            "counterfactual": {
                "n_cases": len(cf_results),
                "final_accuracy": (
                    sum(r["metrics"]["final_accuracy"] for r in cf_results) / len(cf_results)
                    if cf_results
                    else None
                ),
            },
            "stream_only": {
                "n_cases": len(stream_only_results),
                "final_accuracy": (
                    sum(r["metrics"]["final_accuracy"] for r in stream_only_results) / len(stream_only_results)
                    if stream_only_results
                    else None
                ),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MuSR-Stream evaluation over streamed murder mystery evidence.")
    parser.add_argument("--input", type=Path, default=Path("datasets_stream/murder_mystery_stream.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/musr_stream_eval.json"))
    parser.add_argument("--model", type=str, default="ServiceNow-AI/Apriel-1.6-15b-Thinker")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--verbose", action="store_true", help="Print per-round parsed belief updates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache.enable()

    model = TogetherModel(
        engine=args.model,
        api_endpoint="chat",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_max_attempts=5,
    )

    data = read_jsonl(args.input)
    if args.limit is not None:
        data = data[: args.limit]

    results: List[Dict[str, Any]] = []
    cases_iter = tqdm(data, desc="cases")
    for case in cases_iter:
        case_result = evaluate_case(model, case, verbose=args.verbose)
        results.append(case_result)
        cases_iter.set_postfix(
            {
                "last_case": case_result["case_id"],
                "final_top": case_result["final_top_suspect"],
                "correct": case_result["final_correct"],
            }
        )

    summary = summarize(results)

    payload = {
        "config": {
            "input": str(args.input),
            "model": args.model,
            "limit": args.limit,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "summary": summary,
        "cases": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote eval output to {args.output}")


if __name__ == "__main__":
    main()
