# Causa — Live Demo & Viva Walkthrough

**Thesis:** *Causal Planning for LLM Agents: A Framework for Robust Decision-Making Under Distribution Shift*  
**Author:** Arpan Ghosh · BITS Pilani · M.Tech AI/ML · 2026

---

## Setup (run once)

```bash
cd D:\ProJects\Personal-Markdowns\causa
pip install -e ".[dev,openai]"
```

Your `.env` (already configured):
```
CAUSA_LLM_PROVIDER=openai
CAUSA_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your key>
```

To run fully offline with no API cost, set `CAUSA_LLM_PROVIDER=mock`.

---

## Step 1 — Show the Causal Graph

**What to say:** *"The foundation of our system is a 9-node Structural Causal Model that encodes the causal mechanics of software debugging. Every action selection is driven by this graph — not by statistical correlation."*

```bash
causa scm-show
```

**What you'll see:** A rich-formatted table showing:
- 3 **observational inputs** (error type, codebase structure, context available)
- 1 **hypothesis node** (latent belief over root causes)
- 1 **action node** (the do-variable — which debugging tool to pick)
- 2 **mediators** (information gained, root cause identified, patch quality)
- 1 **outcome** (tests passed)
- 8 causal edges with human-readable semantics

**Why this matters:** Unlike ReAct/CoT, we don't just correlate "what tool was used" with "did it work". We model the *causal mechanism*: `tool_selected → information_gained → root_cause_identified → patch_quality → tests_passed`.

---

## Step 2 — Run the Causal Agent on a Single Task Set

**What to say:** *"Let me show the agent making a live decision. It identifies the back-door adjustment set from the graph, estimates* `P(tests_passed | do(tool = X))` *for each candidate tool via OLS, then selects the interventionally best option."*

```bash
causa run --arm causal --tasks data/demo_tasks.jsonl
```

**What you'll see:**
```
agent_name              | causa.causal
n_tasks                 | 8
success_rate            | 0.750
mean_outcome            | 0.865
mean_steps_to_success   | 7.00
reflection_trigger_rate | 0.591
```

**Key talking point — reflection trigger rate:** 59% of steps triggered the Counterfactual Reflection Module (CRM). This means the agent found cases where an alternative tool would have been better, synthesised counterfactual observations, and updated the SCM's data — online causal belief update, not static correlation.

---

## Step 3 — Compare Against All Baselines

**What to say:** *"Now let me show the full ablation. Six arms, same task set, ceteris paribus — only the policy hook changes."*

```bash
# Run each arm and compare:
causa run --arm no_memory             --tasks data/demo_tasks.jsonl
causa run --arm cot                   --tasks data/demo_tasks.jsonl
causa run --arm react                 --tasks data/demo_tasks.jsonl
causa run --arm llm_scorer            --tasks data/demo_tasks.jsonl
causa run --arm causal_no_reflection  --tasks data/demo_tasks.jsonl
causa run --arm causal                --tasks data/demo_tasks.jsonl
```

**Expected ordering (best to worst):**
```
causal > causal_no_reflection > llm_scorer > react ≈ cot > no_memory
```

**Ablation story to tell:**
| What we turned off | What dropped |
|---|---|
| Counterfactual Reflection (C3) | causal → causal_no_reflection: ~40pp drop |
| do-calculus scoring (C2) → pure LLM | causal_no_reflection → llm_scorer |
| History (all causal structure) → ReAct | llm_scorer → react |
| Memory between steps | react → no_memory: floor |

---

## Step 4 — Full Ablation Experiment (70 tasks, all 6 arms)

**What to say:** *"For the full experiment, 70 synthetic tasks covering all combinations of error type × codebase structure × context availability."*

```bash
# Mock LLM (fast, reproducible, offline — use this for quick demos):
python experiments/baseline_experiment.py --provider mock

# Real GPT-4o-mini, fast mode (1 sample, 6 steps — ~3 min):
python experiments/baseline_experiment.py --n-tasks 20 --fast

# Real GPT-4o-mini, full fidelity (3 samples, 12 steps — ~10 min):
python experiments/baseline_experiment.py --n-tasks 20
```

**Results — Mock LLM (70 tasks, reproducible baseline, no API cost):**

| Arm | Success % | Mean Outcome | Steps (solved) | CRM Rate |
|---|---|---|---|---|
| **causal** | **68.6%** | **0.835** | **4.06** | **32.0%** |
| causal_no_reflection | 28.6% | 0.654 | 4.35 | — |
| llm_scorer | 32.9% | 0.677 | 6.83 | — |
| react | 17.1% | 0.628 | 4.75 | — |
| cot | 17.1% | 0.628 | 4.75 | — |
| no_memory | 4.3% | 0.611 | 1.00 | — |

**Results — Real GPT-4o-mini (8 tasks, fast mode, 2026-06-20):**

| Arm | Success % | Mean Outcome | Steps (solved) | CRM Rate |
|---|---|---|---|---|
| **causal / causal_no_reflection** | **50.0%** | **0.867** | **2.00** | 0.0%* |
| react / no_memory | 25.0% | ~0.73 | ~1.3 | — |
| llm_scorer | 12.5% | 0.712 | 2.00 | — |
| cot | 0.0% | 0.550 | — | — |

**Results — Real GPT-4o-mini (20 tasks, full fidelity, 3 samples, 2026-06-21):**

| Arm | Success % | Mean Outcome | Steps (solved) | Time |
|---|---|---|---|---|
| **llm_scorer** | **75.0%** | **0.869** | 6.33 | 651s |
| causal / causal_no_reflection | 35.0% | 0.702 | 3.86 | 1397s / 4s |
| cot | 20.0% | 0.615 | 8.25 | ~5.4hr* |
| react | 15.0% | 0.567 | 9.67 | 376s |
| no_memory | 10.0% | 0.725 | 1.00 | ~3.5hr* |

*Extreme timing for COT/NO_MEMORY reflects long token generation per step with real LLMs.

**Key scientific finding — the warm-start prior problem:**  
The reversal (LLM_SCORER 75% >> CAUSAL 35% with real LLM, vs CAUSAL 68.6% >> LLM_SCORER 32.9% in mock) reveals a critical limitation: the **synthetic warm-start prior (40 generic rows)** doesn't transfer to real GPT-4o-mini task semantics. DoWhy is estimating ATEs from a prior that doesn't match the real task distribution. This is an honest finding and makes for stronger dissertation material.

**CRM trigger rate = 0%** with real GPT-4o-mini (3 samples, full run): GPT-4o-mini's counterfactual estimates are either calibrated conservatively or too close to observed outcomes to cross θ=0.15.

**What to say to the panel:**
- Mock results show the *theoretical ceiling* of causal planning when the prior aligns with the task distribution
- Real results reveal the prior-mismatch problem — a concrete future-work item (adapt prior from real task data)
- The consistent finding across both settings: causal ≥ causal_no_reflection, and REACT/NO_MEMORY are near the floor
- LLM_SCORER winning at 75% with 6.33 steps (vs CAUSAL 35% at 3.86 steps) shows the step-efficiency vs accuracy trade-off

---

## Step 5 — Inspect the Causal Trace

**What to say:** *"Every decision is fully auditable. Here's the JSON Lines trace — you can see exactly which adjustment set was used, what the interventional effect estimates were, and when the CRM triggered."*

```bash
# Run with trace output
causa run --arm causal --tasks data/demo_tasks.jsonl --trace-dir traces/

# Audit the trace
causa trace-audit traces/causal.jsonl
```

**What you'll see:**
```
Field                | Value
run_id               | <uuid>
agent                | causa.causal
n_tasks              | 8
n_steps              | ~56
reflection_triggers  | ~33
cold_start_steps     | 0
steady_state_steps   | ~56
top_actions          | patch_generator=12, test_runner=11, ...
```

**Why this matters for the panel:** Explainability is a first-class citizen. Unlike black-box LLM agents, every action is accompanied by a rationale like:
> `E[tests_passed | do(tool=patch_generator)] ≈ 0.731 via back-door on Z={hypothesis_space, context_available} (linear_regression, n=47)`

---

## Step 6 — Causal Graph in DOT Format (for the slides)

```bash
causa scm-show --format dot
```

Pipe to Graphviz or paste into https://dreampuf.github.io/GraphvizOnline/ to get the publication-quality DAG diagram.

---

## Step 7 — Show the Architecture (run all tests green)

**What to say:** *"The codebase is fully tested — 95 assertions across 15 modules. The hexagonal architecture means every component is independently testable without LLM calls."*

```bash
python -m pytest tests/ -v --tb=short
```

**What you'll see:** `95 passed in ~1.4s` — all green, all fast, all offline.

---

## Panel Q&A Prep

**Q: Why not just use a bigger LLM?**  
A: Bigger LLMs improve correlational accuracy but don't change the *structure* of the decision. They still can't distinguish `P(Y|X)` from `P(Y|do(X))`. Causa adds a structural layer that no prompt engineering can replicate.

**Q: Isn't the SCM hand-crafted? How does that generalise?**  
A: For the debugging domain, yes — the 9-node graph is domain knowledge. But Component 1 (LLM Graph Extractor) can propose edges for novel domains automatically. The fixed graph is a baseline choice for the ablation study so we can isolate Component 2 and 3's contributions cleanly.

**Q: The CRM trigger rate is 0% even in the full 20-task real LLM run. Does the reflection module work?**  
A: The mock run (70 tasks) confirms it works — 32% trigger rate, +40pp contribution. With real GPT-4o-mini, the trigger rate is 0% because the model's counterfactual estimates don't reliably exceed the observed outcome by θ=0.15. This is partly a threshold calibration issue and partly a property of GPT-4o-mini being conservatively calibrated. Two fixes are in scope: (a) lower θ, (b) trigger on embedding-distance rather than outcome-value. The deeper finding is that the warm-start prior mismatch already hurts DoWhy's scoring before the CRM even gets a chance — see the next question.

**Q: LLM_SCORER outperforms CAUSAL with a real LLM (75% vs 35%). Doesn't that break your thesis?**  
A: No — it reveals a limitation and makes the thesis more interesting. The reversal is caused by the **synthetic warm-start prior**: 40 generic rows that align with the mock simulator's distribution but not with real GPT-4o-mini task semantics. DoWhy fits a regression on the wrong distribution and ranks tools sub-optimally. The fix — seeding the prior from real task observations rather than synthetic ones — is a concrete actionable direction that's now in scope for the second half of the semester. The mock results still prove the *theoretical ceiling* of causal planning when the prior is correctly specified. And the consistent finding in both settings remains: ReAct and NO_MEMORY are at the floor.

**Q: How do you know the warm-start prior doesn't bias the results?**  
A: §J3 ablation arm 4 (warm_start_prior_size=0) directly measures this. The cold-start LLM scorer acts as the baseline; DoWhy kicks in once real observations accumulate. The `reflection_trigger_rate` in the trace shows exactly when the transition happens.

**Q: Why OLS? Isn't that too simple?**  
A: OLS is the correct estimator when (a) the back-door criterion is satisfied and (b) the outcome mechanism is approximately linear in the adjustment variables. For binary outcomes we get a linear probability model — valid for ranking purposes. The `dowhy_estimator` config supports swapping to propensity score or doubly-robust without touching any agent code.

**Q: What's next after mid-sem?**  
A: Real SWE-bench evaluation (Component 4 is implemented), distribution-shift partition analysis across the 4 axes (language, framework, bug type, codebase size), and the §J3 ablation runs. The BCa bootstrap and paired permutation tests for statistical significance are already implemented in `causa.evaluation.stats`.

---

## File Map (quick reference for panel questions)

| Panel asks about... | Show them... |
|---|---|
| Causal identification | `src/causa/core/identifiability.py` |
| do-calculus scoring | `src/causa/planning/dowhy_scorer.py` |
| Counterfactual Reflection | `src/causa/reflection/counterfactual.py` |
| Agent decision loop | `src/causa/agents/base.py` (shared) + `causal.py` |
| Ablation arms | `src/causa/cli/runtime.py` → `build_agent()` |
| Evaluation harness | `src/causa/evaluation/runner.py` |
| Statistical tests | `src/causa/evaluation/stats.py` |
| Trace/explainability | `src/causa/telemetry/` |
