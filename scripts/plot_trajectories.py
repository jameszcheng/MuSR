"""Generate presentation-ready plots from streamed eval output.

Usage:
    uv run python scripts/plot_trajectories.py \
        --input outputs/cs422_v3_qwen_stream.json \
        --outdir plots/

Produces:
    1. p_gold_trajectory.png  — P(gold) over round percentile, averaged across cases
    2. evidence_responsiveness.png — distribution of per-round TV distance
    3. streamed_vs_no_stream.png — accuracy comparison (needs --no-stream-input)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_eval(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_gold_for_round(
    round_id: int, gold_suspect: str, cf_info: Dict[str, Any]
) -> str:
    if not cf_info.get("has_counterfactual"):
        return gold_suspect
    cf_rid = cf_info.get("counterfactual_round_id")
    if cf_rid is None:
        return gold_suspect
    gold_before = cf_info.get("gold_before_text")
    gold_after = cf_info.get("gold_after_text") or gold_suspect
    if round_id < int(cf_rid) and gold_before:
        return str(gold_before)
    return str(gold_after)


# ── Plot 1: P(gold) trajectory ──────────────────────────────────────────────


def plot_p_gold_trajectory(
    cases: List[Dict[str, Any]],
    outdir: Path,
    n_bins: int = 20,
    cf_band_start_pct: float = 95.0,
) -> None:
    """P(gold) averaged across cases at round-percentile checkpoints.

    Tracks P(final gold) throughout — for flip_required CF cases this means
    P(gold_after), so the CF revision jump is visible. Only flip_required CF
    cases are included in the counterfactual line to isolate the revision signal.
    A vertical band marks where CF rounds typically fall in percentile space.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    flip_required_cases = [
        c for c in cases
        if c["counterfactual"].get("has_counterfactual")
        and c["counterfactual"].get("flip_required")
    ]
    subsets = {
        "stream_only": [c for c in cases if not c["counterfactual"].get("has_counterfactual")],
        "counterfactual": flip_required_cases,
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"stream_only": "#2196F3", "counterfactual": "#E91E63"}
    labels = {
        "stream_only": f"Stream-only (n={len(subsets['stream_only'])})",
        "counterfactual": f"CF flip-required (n={len(flip_required_cases)})",
    }

    for key, subset in subsets.items():
        if not subset:
            continue
        binned: List[List[float]] = [[] for _ in range(n_bins)]
        for case in subset:
            preds = case.get("round_predictions", [])
            n = len(preds)
            if n == 0:
                continue
            if key == "counterfactual":
                gold = case["counterfactual"].get("gold_after_text") or case["gold_suspect"]
            else:
                gold = case["gold_suspect"]
            for i, r in enumerate(preds):
                pct = i / max(1, n - 1)
                p_gold = float(r["probabilities"].get(gold, 0.0))
                b = min(int(pct * n_bins), n_bins - 1)
                binned[b].append(p_gold)

        means = [np.mean(b) if b else np.nan for b in binned]
        stds = [np.std(b) if b else 0.0 for b in binned]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        valid = ~np.isnan(means_arr)

        x = bin_centers[valid] * 100
        y = means_arr[valid]
        y_lo = means_arr[valid] - stds_arr[valid]
        y_hi = means_arr[valid] + stds_arr[valid]
        ax.plot(x, y, color=colors[key], label=labels[key], linewidth=2)
        ax.fill_between(
            x,
            y_lo,
            y_hi,
            color=colors[key], alpha=0.15,
        )

    # Compute where CF rounds fall in percentile space across flip_required cases
    cf_round_pcts: List[float] = []
    for case in flip_required_cases:
        preds = case.get("round_predictions", [])
        n = len(preds)
        cf_start_rid = case["counterfactual"].get("counterfactual_round_id")
        cf_round_count = int(case["counterfactual"].get("counterfactual_round_count") or 1)
        if n == 0 or cf_start_rid is None:
            continue
        cf_start_rid = int(cf_start_rid)
        rid_to_idx = {int(r.get("round_id", i)): i for i, r in enumerate(preds)}
        for rid in range(cf_start_rid, cf_start_rid + cf_round_count):
            if rid in rid_to_idx:
                idx = rid_to_idx[rid]
                cf_round_pcts.append(idx / max(1, n - 1))

    if cf_round_pcts:
        observed_lo = float(np.percentile(cf_round_pcts, 10)) * 100
        observed_hi = float(np.percentile(cf_round_pcts, 90)) * 100
        band_lo = max(0.0, min(100.0, cf_band_start_pct))
        band_hi = 100.0
        ax.axvspan(
            band_lo,
            band_hi,
            color="#FFC107",
            alpha=0.2,
            label=f"CF rounds (~{band_lo:.0f}–100th pct; observed {observed_lo:.1f}–{observed_hi:.1f})",
        )

    ax.set_xlabel("Round percentile (%)", fontsize=12)
    ax.set_ylabel("P(gold_after)", fontsize=12)
    ax.set_title("Belief Trajectory: P(gold) over Evidence Stream", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "p_gold_trajectory.png", dpi=200)
    plt.close(fig)
    print(f"Saved {outdir / 'p_gold_trajectory.png'}")


# ── Plot 2: Evidence responsiveness distribution ─────────────────────────────


def plot_responsiveness(cases: List[Dict[str, Any]], outdir: Path) -> None:
    """Histogram of per-case mean TV distance between consecutive rounds."""
    values: List[float] = []
    for case in cases:
        preds = case.get("round_predictions", [])
        suspects = list(preds[0]["probabilities"].keys()) if preds else []
        if len(preds) < 2:
            continue
        tvs = []
        for i in range(1, len(preds)):
            prev = preds[i - 1]["probabilities"]
            curr = preds[i]["probabilities"]
            tv = 0.5 * sum(abs(float(curr.get(s, 0)) - float(prev.get(s, 0))) for s in suspects)
            tvs.append(tv)
        values.append(float(np.mean(tvs)))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(values), color="#E91E63", linestyle="--", linewidth=2, label=f"Mean = {np.mean(values):.3f}")
    ax.set_xlabel("Mean TV distance (per case)", fontsize=12)
    ax.set_ylabel("Number of cases", fontsize=12)
    ax.set_title("Evidence Responsiveness Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "evidence_responsiveness.png", dpi=200)
    plt.close(fig)
    print(f"Saved {outdir / 'evidence_responsiveness.png'}")


# ── Plot 3: Streamed vs non-streamed accuracy ────────────────────────────────


def plot_stream_vs_no_stream(
    stream_path: Path, no_stream_path: Path, outdir: Path
) -> None:
    """Bar chart comparing final accuracy: streamed vs non-streamed."""
    stream_data = load_eval(stream_path)
    no_stream_data = load_eval(no_stream_path)

    def accuracy_by_subset(data: Dict[str, Any]):
        cases = data["cases"]
        all_acc = sum(c["final_correct"] for c in cases) / max(1, len(cases))
        cf = [c for c in cases if c["counterfactual"].get("has_counterfactual")]
        so = [c for c in cases if not c["counterfactual"].get("has_counterfactual")]
        cf_acc = sum(c["final_correct"] for c in cf) / max(1, len(cf)) if cf else 0
        so_acc = sum(c["final_correct"] for c in so) / max(1, len(so)) if so else 0
        return {"All": all_acc, "Stream-only": so_acc, "Counterfactual": cf_acc}

    s_acc = accuracy_by_subset(stream_data)
    ns_acc = accuracy_by_subset(no_stream_data)

    groups = list(s_acc.keys())
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, [s_acc[g] for g in groups], width, label="Streamed", color="#2196F3")
    bars2 = ax.bar(x + width / 2, [ns_acc[g] for g in groups], width, label="Non-streamed", color="#FF9800")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Final Accuracy", fontsize=12)
    ax.set_title("Streamed vs Non-Streamed Final Accuracy", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "streamed_vs_no_stream.png", dpi=200)
    plt.close(fig)
    print(f"Saved {outdir / 'streamed_vs_no_stream.png'}")


# ── Plot 4: CF revision — P(gold_after) aligned to CF onset ──────────────────


def plot_cf_revision(cases: List[Dict[str, Any]], outdir: Path, window_before: int = 5, window_after: int = 3) -> None:
    """P(gold_after) for flip_required CF cases, aligned so CF onset = round 0.

    Shows the model's belief in the post-CF gold answer before and after the
    correction block, making the revision jump clearly visible.
    """
    flip_cases = [
        c for c in cases
        if c["counterfactual"].get("has_counterfactual")
        and c["counterfactual"].get("flip_required")
    ]
    if not flip_cases:
        return

    # Collect P(gold_after) at each offset relative to CF onset
    offsets: Dict[int, List[float]] = {}
    for case in flip_cases:
        cf_rid = case["counterfactual"].get("counterfactual_round_id")
        if cf_rid is None:
            continue
        cf_rid = int(cf_rid)
        gold_after = case["counterfactual"].get("gold_after_text") or case["gold_suspect"]

        preds = case.get("round_predictions", [])
        for r in preds:
            rid = int(r.get("round_id", 0))
            offset = rid - cf_rid
            if offset < -window_before or offset > window_after:
                continue
            p = float(r["probabilities"].get(gold_after, 0.0))
            offsets.setdefault(offset, []).append(p)

    if not offsets:
        return

    xs = sorted(offsets.keys())
    means = [np.mean(offsets[x]) for x in xs]
    stds = [np.std(offsets[x]) for x in xs]
    counts = [len(offsets[x]) for x in xs]
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, means_arr, "o-", color="#E91E63", linewidth=2, markersize=6, zorder=3)
    ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr, color="#E91E63", alpha=0.15)
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.5, label="CF onset")
    ax.axhline(0.5, color="#ccc", linestyle=":", linewidth=1)

    ax.set_xlabel("Round offset from CF onset", fontsize=12)
    ax.set_ylabel("P(gold_after)", fontsize=12)
    ax.set_title(f"Counterfactual Revision (n={len(flip_cases)} flip-required cases)", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "cf_revision.png", dpi=200)
    plt.close(fig)
    print(f"Saved {outdir / 'cf_revision.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots from eval output.")
    p.add_argument("--input", type=Path, required=True, help="Streamed eval JSON")
    p.add_argument("--no-stream-input", type=Path, default=None, help="Non-streamed eval JSON (for comparison plot)")
    p.add_argument("--outdir", type=Path, default=Path("plots"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load_eval(args.input)
    cases = data["cases"]

    plot_p_gold_trajectory(cases, args.outdir)
    plot_responsiveness(cases, args.outdir)
    plot_cf_revision(cases, args.outdir)

    if args.no_stream_input:
        plot_stream_vs_no_stream(args.input, args.no_stream_input, args.outdir)


if __name__ == "__main__":
    main()
