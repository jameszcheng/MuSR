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
---

<!-- _class: title -->

# MuSR-Stress
## Stress-Testing LLMs on Long-Context and Revisable Belief Reasoning

James Cheng · CS 422

---

## Motivation & Background

- LLMs do well on **static QA** — but real reasoning requires updating beliefs as evidence evolves
- **MuSR** (Sprague et al., 2024) introduced logic-tree-grounded murder mystery reasoning
- **MuSR-Stress** extends it to trajectory-level belief tracking across 2 tracks:

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
  │  rounds 1–N:  narrative sentences split from story prose
  │                (model sees the same text a human reader would)
  │
  └─► 170/250 cases (68%): counterfactual correction injected at midpoint
          ├── flip_required=True  (120/170 CF, ~71%): gold answer changes → model must revise
          └── flip_required=False  (50/170 CF, ~29%): gold unchanged     → model must stay stable
```

Config: `max_rounds=40` · `counterfactual_rate=0.7` · `flip_rate=0.7` · `seed=7`

---

## Evaluation Protocol

At **every round**, model receives accumulated evidence and must return:

```json
{ "top_suspect": "Dale", "scores": { "Dale": 8, "Letti": 2 } }
```

Scores normalize to probabilities; model is re-prompted after each sentence.

| Metric | Measures |
|---|---|
| `final_accuracy` | Final prediction matches gold |
| `update_consistency` | Penalizes unjustified trajectory flips |
| `brier_final / mean` | Calibration error |
| `flip_when_required` | Does model revise after required correction? |
| `stability_when_not_required` | Does model avoid spurious flips? |
| `recovery_rate` | Does model ever reach revised gold? |

---

## Example: Counterfactual Case

**Suspects:** Dale vs. Letti · Gold flips at round 21 (`flip_required=True`)

| Rounds | Type | Evidence |
|---|---|---|
| 1–20 | narrative | Story prose: Dale confronts victim, suspicious licenses, café presence, invitation to her house on day of murder |
| **21** | **CF** | **Correction: Letti witness timeline had wrong timestamp — withdrawn** |
| **22** | **CF** | **Dale's phone near scene + weapon purchase + threatening messages** |
| 23–40 | narrative | Story continues — model must hold revised belief (**Dale**) |

---

## Results & Analysis

temp=0 · test split (n=38)

| Metric | Qwen2.5-7B-Instruct | Llama-3.3-70B-Instruct |
|---|---|---|
| `final_accuracy` | 0.789 | TBD |
| — counterfactual (n=28) | 0.893 | TBD |
| — stream-only (n=10) | 0.500 | TBD |
| `update_consistency` | 0.940 | TBD |
| `brier_final` | 0.344 | TBD |
| `flip_when_required` | **0.952** | TBD |
| `stability_when_not_req` | 1.000 | TBD |
| `recovery_rate` | **1.000** | TBD |
| `recovery_latency` (rounds) | 0.43 | TBD |

---

## Next Steps & Conclusion

**Next steps**
- Scale to Llama-3.3-70B for stronger baseline comparison
- Failure taxonomy: missed flip vs. spurious flip vs. calibration
- Stronger prompting baselines (explicit belief-state tracking)

**Conclusion**
- MuSR-Stress enables **fine-grained diagnosis** of belief reasoning failures
- Qwen2.5-7B handles counterfactual revision well — but **stream-only cases are harder** (no explicit correction anchor)

---

<!-- _class: title -->

# Thank You

James Cheng · CS 422
