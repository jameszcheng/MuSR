---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 28px;
  }
  h1 {
    font-size: 52px;
    font-weight: 700;
  }
  h2 {
    font-size: 36px;
    font-weight: 600;
    color: #222;
  }
  code {
    background: #e8f5e9;
    color: #2e7d32;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  table {
    font-size: 24px;
    border-collapse: collapse;
    width: 100%;
  }
  th {
    background: #f5f5f5;
    padding: 8px 16px;
  }
  td {
    padding: 6px 16px;
  }
  img {
    max-width: min(92%, 1000px);
    max-height: min(62vh, 520px);
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 0.4em auto 0;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
  }
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.plot {
    font-size: 24px;
  }
  section.plot h2 {
    font-size: 32px;
    margin-bottom: 0.2em;
  }
  section.plot p {
    margin: 0.2em 0;
  }
  section.plot ul {
    margin-top: 0.3em;
  }
  section.plot img {
    max-width: min(88%, 900px);
    max-height: min(44vh, 400px);
  }
---

<!-- _class: title -->

# MuSR-Stress
## Stress-Testing LLMs on Long-Context and Revisable Belief Reasoning

James Cheng · CS 422

---

## Motivation & Background

- LLMs do well on **static QA** — but real reasoning requires updating beliefs as evidence evolves
- **MuSR** (Sprague et al., 2024) synthetically generates logic-tree-grounded QA across murder mystery, object placement, and team allocation
- **MuSR-Stress** extends the murder mystery domain to trajectory-level belief tracking across 2 tracks:

| Track | Tests |
|---|---|
| `long_context` | Attention over long narratives with distractors |
| `dynamic_belief` | Belief updates + counterfactual revision |

Each track: **250 cases · 175 train / 37 dev / 38 test**

---

## Base Story Generation Pipeline

```
 Madlib pools          Logic Tree              Narrative            QA Instance
(no LLM)              (LLM fills values)      (LLM writes)
─────────────         ──────────────────      ────────────         ───────────
names                 "Suspect is killer"     "Detective            question
weapons         ───►  ├─ has means      ───►  Winston              choices
crime scenes          │  ├─ [fact]            observed..."    ───►  gold answer
motives               │  └─ [commonsense]                          logic trees
                      ├─ has motive
                      └─ has opportunity
```

**Validators gate the logic tree step** — wrong structure or forbidden keywords → retry

---

## Dynamic Belief Track

```
base case
  │  rounds 1–N:  narrative paragraphs (full story, no cap)
  │                13–43 rounds/case · median 25 · each round = one scene beat
  │
  └─► 170/250 cases (68%): counterfactual correction appended after final paragraph
          ├── flip_required=True  (120/170 CF, ~71%): gold answer changes → model must revise
          └── flip_required=False  (50/170 CF, ~29%): gold unchanged     → model must stay stable
```

Config: `counterfactual_rate=0.7` (70% of cases get CF block) · `flip_rate=0.7` (of those, 70% require gold to change) · `seed=7` (reproducibility)

---

## Evaluation Protocol

Evidence streamed round-by-round; model re-prompted at each round.

| Metric | What it captures |
|---|---|
| `final_accuracy` / `brier_final` | Correctness and calibration |
| `flip_when_required` / `stability_when_not_required` | CF revision success / spurious flip rate |
| `belief_trajectory` | P(gold) curve over rounds — how beliefs evolve |
| `evidence_responsiveness` | Mean distributional shift between rounds (TV distance) |

---

## Example: Counterfactual Case

**Suspects:** Dale vs. Letti · Gold flips after final narrative round (`flip_required=True`)

| Rounds | Type | Evidence |
|---|---|---|
| 1–37 | narrative | Full story evidence stream |
| **38** | **CF** | **Correction: key witness timestamp used against Letti was wrong and withdrawn** |
| **39** | **CF** | **Dale phone/location + weapon purchase + escalating conflict records** |

CF block appended as final 2 rounds — tests whether model can override accumulated belief.

---

## Results (Qwen, cs422_v3 test, n=38)

| Metric | Qwen2.5-7B (v3) |
|---|---|
| `final_accuracy` | 0.816 |
| — CF (n=22) | 0.955 |
| — stream-only (n=16) | 0.625 |
| `flip_when_required` (n=18) | 1.000 |
| — `flip_diagnostic` (n=12) | 1.000 |
| `stability_when_not_required` | 0.750 |
| `evidence_responsiveness` | 0.041 |

6/18 flip-required cases are **non-diagnostic** (already on `gold_after` pre-CF). On the 12 diagnostic cases, Qwen flips correctly in all.

---

## Belief Trajectory: P(gold) over Evidence Stream
<!-- _class: plot -->

- Trajectory CF line uses only `flip_required` cases.
- Yellow band marks where CF rounds occur (`~95th–100th` percentile).
- Read this plot as end-of-stream separation; use the aligned view below for jump sharpness.

**Qwen2.5-7B**
![Qwen2.5-7B](./plots/qwen/p_gold_trajectory.png)

---

## CF Revision (Aligned to CF Onset)
<!-- _class: plot -->

- This alignment makes the post-CF belief shift directly visible.

**Qwen2.5-7B**
![Qwen2.5-7B](./plots/qwen/cf_revision.png)

---

## Evidence Responsiveness Distribution
<!-- _class: plot -->

**Qwen2.5-7B**
![Qwen2.5-7B](./plots/qwen/evidence_responsiveness.png)

---

## Next Steps & Conclusion

**Next steps**
- Primary: integrate counterfactual and long-context changes during base story generation (not post-processing)
- Stronger prompting: few-shot, CoT belief tracking, symbolic evidence graph
- Scale evaluation to larger models and add human baseline

**Conclusion**
- P(gold) shifts gradually and responsiveness is low (~0.04), indicating modest per-round updates
- Stream-only accuracy lags CF accuracy for Qwen (0.625 vs 0.955)
- Streaming eval exposes reasoning dynamics invisible to static benchmarks

---

<!-- _class: title -->

# Thank You!

James Cheng · CS 422
