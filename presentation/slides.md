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

# MuSR-Dynamic
## A Benchmark for Long-Context and Revisable Belief Reasoning

James Cheng · CS 422

---

## Motivation & Background

- LLMs do well on **static QA** — but real reasoning requires updating beliefs as evidence evolves
- **MuSR** (Sprague et al., 2024) introduced logic-tree-grounded murder mystery reasoning
- **MuSR-Dynamic** extends it to trajectory-level belief tracking across 2 tracks:

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

**Validators gate every LLM step** — wrong structure or forbidden keywords → retry

---

## Dynamic Belief Track

```
base case
  │  round 1-2:  setup sentences
  │  round 3+:   evidence facts (from logic tree leaf nodes)
  │
  └─► 68% of cases: counterfactual correction injected late
          ├── flip_required=True  (120/170): gold answer changes → model must revise
          └── flip_required=False  (50/170): gold unchanged     → model must stay stable
```

Config: `max_rounds=40` · `counterfactual_rate=0.7` · `flip_rate=0.7` · `seed=7`

---

## Evaluation Protocol

Model outputs `top_suspect` + probability distribution **each round**

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

**Suspects:** Milton vs. Alice · **Murder weapon:** shotgun

| Round | Type | Evidence |
|---|---|---|
| 1–2 | setup | Detective Winston investigates Isaac's murder at a fitness center |
| 3 | tree_fact | Having a shotgun at Milton's disposal aligns with the murder weapon |
| 4 | tree_fact | Alice recently purchased a shotgun |
| 5 | tree_fact | Milton owns a shotgun |
| … | … | … |
| **21** | **counterfactual** | **Correction: key witness timeline against Alice had an incorrect timestamp and is withdrawn** |

Gold before round 21: **Alice** → Gold after: **Milton** (`flip_required=True`)

---

## Results & Analysis

Model: **Apriel-1.6-15b-Thinker** · temp=0 · test split (n=38)

<div class="columns">
<div>

**Accuracy**
| Subset | Value |
|---|---|
| Overall | 0.342 |
| Counterfactual | 0.321 |
| Stream-only | 0.400 |

**Stability** ✅
| Metric | Value |
|---|---|
| `update_consistency` | 0.958 |
| `stability_when_not_req` | 1.000 |

</div>
<div>

**Revisability** ❌
| Metric | Value |
|---|---|
| `flip_when_required` | **0.286** |
| `recovery_rate` | **0.381** |

<br>

> **Stable but anchored** — never spuriously flips, but ignores corrections when it should revise

</div>
</div>

---

## Next Steps & Conclusion

**Next steps**
- Multi-model eval across both tracks
- Failure taxonomy: missed flip vs. spurious flip vs. calibration
- Stronger prompting baselines (explicit belief-state tracking)

**Conclusion**
- MuSR-Dynamic enables **fine-grained diagnosis** of belief reasoning failures
- Key finding: current models are stable but fail at **non-monotonic revision**

---

<!-- _class: title -->

# Thank You

James Cheng · CS 422
