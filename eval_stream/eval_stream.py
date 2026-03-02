import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src import cache
from src.model import TogetherModel, extract_text_from_response


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


def parse_cf_round_id(cf_info: Dict[str, Any]) -> Optional[int]:
    cf_round_id = cf_info.get("counterfactual_round_id")
    try:
        return int(cf_round_id)
    except (TypeError, ValueError):
        return None


def active_gold_for_round(
    round_id: int,
    fallback_gold: str,
    cf_info: Dict[str, Any],
) -> str:
    if not cf_info.get("has_counterfactual"):
        return fallback_gold

    cf_round_id = parse_cf_round_id(cf_info)
    if cf_round_id is None:
        return fallback_gold

    gold_before = cf_info.get("gold_before_text")
    gold_after = cf_info.get("gold_after_text") or fallback_gold
    if round_id < cf_round_id and gold_before:
        return str(gold_before)
    return str(gold_after)


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


def parse_model_belief(raw_text: str, suspects: List[str], case_id: str = "", round_id: int = -1) -> Dict[str, Any]:
    parsed = try_parse_json(raw_text)
    top_suspect = None
    raw_scores: Dict[str, float] = {}
    parse_warning: Optional[str] = None

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
        if not raw_scores:
            parse_warning = "JSON parsed but no valid suspect scores found"
    else:
        top_suspect, raw_scores = parse_scores_text_fallback(raw_text, suspects)
        if not raw_scores:
            parse_warning = "No valid JSON in output — model likely truncated before producing JSON (increase --max-tokens)"

    if parse_warning:
        loc = f"[{case_id} r{round_id}] " if case_id else ""
        print(f"WARNING {loc}{parse_warning}. Falling back to uniform distribution.")

    probs = normalize_scores(raw_scores, suspects)
    if top_suspect not in suspects:
        top_suspect = max(probs.items(), key=lambda kv: kv[1])[0]

    return {
        "top_suspect": top_suspect,
        "raw_scores": raw_scores,
        "probabilities": probs,
        "parse_warning": parse_warning,
    }


def brier_score(probabilities: Dict[str, float], gold_suspect: str, suspects: List[str]) -> float:
    out = 0.0
    for s in suspects:
        y = 1.0 if s == gold_suspect else 0.0
        p = float(probabilities.get(s, 0.0))
        out += (p - y) ** 2
    return out


def belief_convergence(round_preds: List[Dict[str, Any]], gold_suspect: str, cf_info: Dict[str, Any]) -> Optional[float]:
    """Slope of P(gold) over rounds via linear regression, normalized to [−1, 1] range.

    For CF cases, only uses pre-CF rounds (narrative portion) since the gold
    changes at the CF point. For stream-only cases, uses all rounds.
    Returns None if fewer than 2 applicable rounds.
    """
    cf_round_id = parse_cf_round_id(cf_info) if cf_info.get("has_counterfactual") else None
    pre_cf_gold = cf_info.get("gold_before_text") or gold_suspect

    points: List[Tuple[int, float]] = []
    for r in round_preds:
        rid = int(r["round_id"])
        if cf_round_id is not None and rid >= cf_round_id:
            break
        active_gold = pre_cf_gold if cf_info.get("has_counterfactual") else gold_suspect
        p_gold = float(r["probabilities"].get(active_gold, 0.0))
        points.append((rid, p_gold))

    if len(points) < 2:
        return None

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    # Normalize x to [0, 1] so slope is comparable across different round counts
    x_norm = (x - x[0]) / (x[-1] - x[0])
    slope = float(np.polyfit(x_norm, y, 1)[0])
    return slope


def evidence_responsiveness(round_preds: List[Dict[str, Any]], suspects: List[str], cf_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Mean total variation distance between consecutive round distributions.

    TV = 0.5 * sum(|p_i - q_i|). A value near 0 means the model ignores new
    evidence; higher values mean the distribution shifts each round.
    Only considers narrative rounds (excludes CF rounds) so the metric is
    comparable across stream-only and counterfactual cases.
    Returns None if fewer than 2 applicable rounds.
    """
    cf_round_id = parse_cf_round_id(cf_info) if cf_info and cf_info.get("has_counterfactual") else None
    narrative = [r for r in round_preds if cf_round_id is None or int(r["round_id"]) < cf_round_id]

    if len(narrative) < 2:
        return None

    tvs: List[float] = []
    for i in range(1, len(narrative)):
        prev = narrative[i - 1]["probabilities"]
        curr = narrative[i]["probabilities"]
        tv = 0.5 * sum(abs(float(curr.get(s, 0.0)) - float(prev.get(s, 0.0))) for s in suspects)
        tvs.append(tv)

    return float(np.mean(tvs))


def flip_when_required(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    if not cf_info["has_counterfactual"] or not cf_info["flip_required"]:
        return None

    cf_round_id = parse_cf_round_id(cf_info)
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
    # Count as success either when the model flips correctly, or when it was already
    # on the revised gold before the correction point and stays correct.
    return float(post_final_top == gold_after and (pre_top != post_final_top or pre_top == gold_after))


def flip_diagnostic(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    """flip_when_required restricted to diagnostic cases: those where the model
    was correctly on gold_before pre-CF, so the flip genuinely tests revision.

    Returns None if the case isn't flip_required or if the model was already
    (incorrectly) on gold_after pre-CF — those cases are non-diagnostic because
    the model failed to track pre-CF evidence, and its error happened to align
    with the post-CF gold.
    """
    if not cf_info["has_counterfactual"] or not cf_info["flip_required"]:
        return None

    cf_round_id = parse_cf_round_id(cf_info)
    if cf_round_id is None:
        return None

    pre = [r for r in round_preds if r["round_id"] < cf_round_id]
    post = [r for r in round_preds if r["round_id"] >= cf_round_id]
    if not pre or not post:
        return None

    pre_top = pre[-1]["top_suspect"]
    gold_after = cf_info.get("gold_after_text")
    if not gold_after:
        return None

    # Skip non-diagnostic cases: model was already on gold_after before CF
    if pre_top == gold_after:
        return None

    post_final_top = post[-1]["top_suspect"]
    return float(post_final_top == gold_after)


def pre_cf_on_gold_after(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[bool]:
    """Whether the model was already predicting gold_after before seeing
    the counterfactual block. Only defined for flip_required cases."""
    if not cf_info["has_counterfactual"] or not cf_info["flip_required"]:
        return None

    cf_round_id = parse_cf_round_id(cf_info)
    if cf_round_id is None:
        return None

    pre = [r for r in round_preds if r["round_id"] < cf_round_id]
    if not pre:
        return None

    gold_after = cf_info.get("gold_after_text")
    if not gold_after:
        return None

    return pre[-1]["top_suspect"] == gold_after


def stability_when_not_required(round_preds: List[Dict[str, Any]], cf_info: Dict[str, Any]) -> Optional[float]:
    if not cf_info["has_counterfactual"] or cf_info["flip_required"]:
        return None

    cf_round_id = parse_cf_round_id(cf_info)
    if cf_round_id is None:
        return None

    pre = [r for r in round_preds if r["round_id"] < cf_round_id]
    post = [r for r in round_preds if r["round_id"] >= cf_round_id]
    if not pre or not post:
        return None

    pre_top = pre[-1]["top_suspect"]
    post_final_top = post[-1]["top_suspect"]
    return float(pre_top == post_final_top)


def evaluate_case_no_stream(model: TogetherModel, case: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Non-streamed baseline: feed all evidence at once, get a single prediction."""
    case_id = case_id_of(case)
    gold_suspect = gold_suspect_of(case)
    cf_info = counterfactual_info(case)

    all_evidence = [rnd["evidence_text"] for rnd in case["rounds"]]
    prompt = build_prompt(case, all_evidence)
    raw = model.inference(prompt)
    output = extract_text_from_response(raw)
    belief = parse_model_belief(output, case["suspects"], case_id=case_id, round_id=-1)

    final_pred = belief["top_suspect"]
    final_correct = int(final_pred == gold_suspect)
    bf = brier_score(belief["probabilities"], gold_suspect, case["suspects"])

    if verbose:
        probs_str = ", ".join(f"{k}={v:.3f}" for k, v in belief["probabilities"].items())
        print(f"[{case_id} no-stream] top={final_pred} | probs: {probs_str}")

    return {
        "case_id": case_id,
        "gold_suspect": gold_suspect,
        "final_top_suspect": final_pred,
        "final_correct": final_correct,
        "counterfactual": cf_info,
        "metrics": {
            "final_accuracy": final_correct,
            "brier_final": bf,
            "flip_when_required": None,
            "flip_diagnostic": None,
            "pre_cf_on_gold_after": None,
            "stability_when_not_required": None,
            "belief_convergence": None,
            "evidence_responsiveness": None,
        },
        "round_predictions": [belief],
        "round_traces": [{"round_id": -1, "prompt": prompt, "raw_output": output, "parsed": belief}],
    }


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
        belief = parse_model_belief(output, case["suspects"], case_id=case_id, round_id=int(rnd["round_id"]))
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
    round_briers = []
    for r in round_preds:
        round_gold = active_gold_for_round(
            round_id=int(r["round_id"]),
            fallback_gold=gold_suspect,
            cf_info=cf_info,
        )
        round_briers.append(brier_score(r["probabilities"], round_gold, case["suspects"]))

    return {
        "case_id": case_id,
        "gold_suspect": gold_suspect,
        "final_top_suspect": final_pred,
        "final_correct": final_correct,
        "counterfactual": cf_info,
        "metrics": {
            "final_accuracy": final_correct,
            "brier_final": round_briers[-1] if round_briers else None,
            "flip_when_required": flip_when_required(round_preds, cf_info),
            "flip_diagnostic": flip_diagnostic(round_preds, cf_info),
            "pre_cf_on_gold_after": pre_cf_on_gold_after(round_preds, cf_info),
            "stability_when_not_required": stability_when_not_required(round_preds, cf_info),
            "belief_convergence": belief_convergence(round_preds, gold_suspect, cf_info),
            "evidence_responsiveness": evidence_responsiveness(round_preds, case["suspects"], cf_info),
        },
        "round_predictions": round_preds,
        "round_traces": traces,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "n_cases": 0,
            "parse_failure_cases": 0,
            "final_accuracy": None,
            "brier_final": None,
            "flip_when_required": None,
            "flip_diagnostic": None,
            "flip_diagnostic_n": 0,
            "flip_nondiagnostic_n": 0,
            "flip_required_n": 0,
            "stability_when_not_required": None,
            "belief_convergence": None,
            "evidence_responsiveness": None,
            "counterfactual_cases": 0,
            "stream_only_cases": 0,
        }

    n = len(results)
    cf_results = [r for r in results if r.get("counterfactual", {}).get("has_counterfactual")]
    stream_only_results = [r for r in results if not r.get("counterfactual", {}).get("has_counterfactual")]
    fwr = [r["metrics"]["flip_when_required"] for r in results if r["metrics"]["flip_when_required"] is not None]
    fdiag = [r["metrics"]["flip_diagnostic"] for r in results if r["metrics"]["flip_diagnostic"] is not None]
    nondiagnostic_flip_cases = sum(
        1 for r in results
        if r["metrics"].get("pre_cf_on_gold_after") is True
    )
    total_flip_required = sum(
        1 for r in results
        if r["metrics"].get("flip_when_required") is not None
    )
    swnr = [
        r["metrics"]["stability_when_not_required"]
        for r in results
        if r["metrics"]["stability_when_not_required"] is not None
    ]
    bc = [r["metrics"]["belief_convergence"] for r in results if r["metrics"]["belief_convergence"] is not None]
    er = [r["metrics"]["evidence_responsiveness"] for r in results if r["metrics"]["evidence_responsiveness"] is not None]
    parse_failure_cases = sum(
        1 for r in results
        if any(rp.get("parse_warning") for rp in r.get("round_predictions", []))
    )

    return {
        "n_cases": n,
        "parse_failure_cases": parse_failure_cases,
        "final_accuracy": sum(r["metrics"]["final_accuracy"] for r in results) / n,
        "brier_final": sum(r["metrics"]["brier_final"] for r in results if r["metrics"]["brier_final"] is not None)
        / max(1, sum(1 for r in results if r["metrics"]["brier_final"] is not None)),
        "flip_when_required": (sum(fwr) / len(fwr)) if fwr else None,
        "flip_diagnostic": (sum(fdiag) / len(fdiag)) if fdiag else None,
        "flip_diagnostic_n": len(fdiag),
        "flip_nondiagnostic_n": nondiagnostic_flip_cases,
        "flip_required_n": total_flip_required,
        "stability_when_not_required": (sum(swnr) / len(swnr)) if swnr else None,
        "belief_convergence": (sum(bc) / len(bc)) if bc else None,
        "evidence_responsiveness": (sum(er) / len(er)) if er else None,
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
    parser.add_argument("--input", type=Path, default=Path("benchmark_runs/cs422_v3/dynamic_belief/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_out.json"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-Turbo")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true", help="Print per-round parsed belief updates.")
    parser.add_argument("--no-stream", action="store_true", help="Non-streamed baseline: feed all evidence at once.")
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

    eval_fn = evaluate_case_no_stream if args.no_stream else evaluate_case
    results: List[Dict[str, Any]] = []
    cases_iter = tqdm(data, desc="cases")
    for case in cases_iter:
        case_result = eval_fn(model, case, verbose=args.verbose)
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
            "no_stream": args.no_stream,
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
