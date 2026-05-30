# Causa — Project Wiki

**Causal Planning for LLM Agents: A Framework for Robust Decision-Making Under Distribution Shift**

> Arpan Ghosh · BITS Pilani · M.Tech Dissertation · May 2026

---

## What is Causa?

Causa is a reference implementation of the dissertation's central claim:
**a Pearl-style structural causal model, fitted online and guarded by a
counterfactual reflection module, produces more robust LLM-agent
decisions under distribution shift than the strongest LLM-only
baselines.**

The codebase is deliberately small, hexagonally layered, and design-pattern-driven
so every ablation in §J reduces to a one-line swap in
`causa.cli.runtime.build_agent`. There is no orchestration framework, no
DI container — just narrow ports, concrete adapters, and pure-domain
core types.

| | |
|---|---|
| **Language** | Python 3.10+ |
| **Style** | Hexagonal (Ports & Adapters), Strategy, Builder, Factory |
| **Causal libraries** | DoWhy, NetworkX (behind adapters only) |
| **LLM client** | Anthropic adapter + deterministic MockLLMClient |
| **Test surface** | ~200 pytest assertions, ~13 test modules |
| **Trace format** | Append-only JSON Lines, schema-versioned |

---

## Repository layout

```
causa/
├── src/causa/
│   ├── core/                       # 1 — pure SCM algebra (no I/O)
│   │   ├── graph.py                #     CausalGraph, CausalEdge, GraphInvariantError
│   │   ├── scm.py                  #     SCM + SCMBuilder
│   │   ├── identifiability.py      #     back-door criterion (Pearl 2009)
│   │   ├── intervention.py         #     mutilation under do(X = x)
│   │   └── variables.py            #     typed domains
│   ├── ports/                      # 2 — narrow contracts
│   │   ├── llm.py                  #     LLMClient
│   │   ├── scorer.py               #     ActionScorer + ActionScore/Candidate
│   │   ├── estimator.py            #     EffectEstimator
│   │   ├── history.py              #     ObservationHistory
│   │   ├── tool.py                 #     DebuggingTool
│   │   └── threshold.py            #     Threshold (for the CRM)
│   ├── adapters/                   # 3 — concrete implementations
│   │   ├── llm/                    #     anthropic_client, mock_client, factory
│   │   ├── estimators/             #     linear_regression
│   │   ├── history/                #     pandas_history
│   │   ├── tools/                  #     8 typed debugging tool stubs
│   │   └── threshold/              #     static, adaptive_median
│   ├── domain/                     # 4 — domain-specific glue
│   │   ├── scm_debugging.py        #     build_debugging_scm() — canonical 9-node DAG
│   │   └── tasks.py                #     DebuggingTask, TaskOutcome
│   ├── planning/                   # 5 — Component 2 (action scoring)
│   │   ├── dowhy_scorer.py
│   │   └── llm_scorer.py           #     cold-start fallback
│   ├── reflection/                 # 6 — Component 4 (CRM)
│   │   └── module.py               #     CounterfactualReflectionModule
│   ├── extraction/                 # 7 — Component 1 (novel domains)
│   │   └── llm_extractor.py        #     LLMGraphExtractor
│   ├── agents/                     # 8 — Component 3 (planning loop)
│   │   ├── base.py                 #     BaseAgent (shared scaffold)
│   │   ├── causal.py               #     CausalPlanningAgent
│   │   ├── react.py                #     Yao 2023 baseline
│   │   ├── chain_of_thought.py     #     Wei 2022 baseline
│   │   └── no_memory.py            #     ablation floor
│   ├── evaluation/                 # 9 — evaluation harness
│   │   ├── runner.py               #     EvaluationRunner, RunObserver
│   │   ├── partition.py            #     IID vs OOD partitioner
│   │   ├── metrics.py              #     EvaluationMetrics, MetricRow
│   │   ├── stats.py                #     BCa bootstrap + paired permutation
│   │   └── swebench.py             #     JSONL loader + 4-axis classifier
│   ├── telemetry/                  # 10 — trace plumbing
│   │   ├── events.py               #     TraceEvent + builders (TRACE_VERSION = 1)
│   │   ├── writer.py               #     TraceWriter, NullTraceWriter
│   │   └── auditor.py              #     post-hoc TraceAudit
│   └── cli/                        # 11 — driving adapter
│       ├── main.py                 #     typer commands
│       └── runtime.py              #     AgentArm enum + build_agent
├── tests/                          # ~200 pytest assertions
│   ├── unit/                       #     13 modules (core through CLI runtime)
│   └── integration/                #     DoWhy round-trip with synthetic data
├── docs/wiki/                      # this directory
│   ├── WIKI.md                     #     the document you are reading
│   └── diagrams/                   #     15 print-quality HTML diagrams
└── pyproject.toml
```

---

## Diagrams

All 15 diagrams live in [`docs/wiki/diagrams/`](diagrams/) and render
standalone in any browser. They are designed for 1700×1100 export to PDF
for inclusion in the dissertation.

| # | File | What it shows |
|---|---|---|
| 01 | [`01-system-overview.html`](diagrams/01-system-overview.html) | Hexagonal architecture — core / ports / adapters / application layers |
| 02 | [`02-debugging-scm.html`](diagrams/02-debugging-scm.html) | The canonical 9-node debugging SCM with roles and 8 edges |
| 03 | [`03-do-calculus-pipeline.html`](diagrams/03-do-calculus-pipeline.html) | Identify → Estimate → Rank → Audit |
| 04 | [`04-causal-planning-agent.html`](diagrams/04-causal-planning-agent.html) | The `CausalPlanningAgent` decision loop |
| 05 | [`05-ablation-arms.html`](diagrams/05-ablation-arms.html) | Six ablation arms compared as a capability matrix |
| 06 | [`06-counterfactual-reflection.html`](diagrams/06-counterfactual-reflection.html) | The Counterfactual Reflection Module |
| 07 | [`07-distribution-shift-partition.html`](diagrams/07-distribution-shift-partition.html) | 4-axis IID / OOD / Unlabeled partitioner |
| 08 | [`08-trace-event-flow.html`](diagrams/08-trace-event-flow.html) | JSON Lines trace pipeline + `TraceAudit` |
| 09 | [`09-cold-start-vs-steady.html`](diagrams/09-cold-start-vs-steady.html) | LLM cold-start vs steady-state DoWhy regime |
| 10 | [`10-evaluation-runner.html`](diagrams/10-evaluation-runner.html) | Serial &amp; parallel runner paths |
| 11 | [`11-bca-permutation-stats.html`](diagrams/11-bca-permutation-stats.html) | BCa bootstrap + paired permutation test |
| 12 | [`12-llm-graph-extraction.html`](diagrams/12-llm-graph-extraction.html) | `LLMGraphExtractor` for novel domains |
| 13 | [`13-end-to-end-data-flow.html`](diagrams/13-end-to-end-data-flow.html) | Tasks → Trace → Report |
| 14 | [`14-four-components.html`](diagrams/14-four-components.html) | The four components of Causa (§C–§E) |
| 15 | [`15-dissertation-roadmap.html`](diagrams/15-dissertation-roadmap.html) | §A–§N chapter → module mapping |

---

## The four components (§C–§E)

### Component 1 — Causal Graph Specification (`causa.core`, `causa.domain.scm_debugging`, `causa.extraction`)

A typed DAG of variables with roles (`observational`, `mediator`,
`action`, `outcome`). For the debugging domain the graph is hand-authored
in `build_debugging_scm()`; for novel domains, the `LLMGraphExtractor`
proposes edges and the core's DAG invariants drop any that would create
cycles or reference unknown variables.

### Component 2 — Identifiable Action Scorer (`causa.planning`)

For each candidate action `a`, we identify a back-door adjustment set
`Z` from the SCM graph, fit a plug-in linear estimator with `Z`
controlled, and emit an `ActionScore(action, score, rationale,
adjustment_set)`. When data is too sparse (`history_size < n_min`) the
scorer cleanly falls back to an LLM cold-start scorer, stamping the
score with `[cold-start LLM]` so the auditor can attribute decisions to
the correct regime.

### Component 3 — Causal Planning Agent (`causa.agents`)

A shared `BaseAgent` outer loop with three policy hooks
(`_choose_action`, `_after_step`, `_next_state`) plus a step budget,
success threshold, and tool registry. All six ablation arms — including
ReAct, CoT, and No-Memory — share the loop; **only the hook bodies
differ**, making the ablation a strict ceteris-paribus comparison.

### Component 4 — Counterfactual Reflection Module (`causa.reflection`)

After each step, the CRM computes the discrepancy `δ = |y_hat - y_obs|`
between the SCM's prediction and the observed outcome. If `δ` exceeds
the adaptive threshold (default: median of past discrepancies + ε), it
calls the LLM via the `[CAUSA::counterfactual]` prompt to estimate
alternative outcomes and appends synthetic rows to the history. This is
the mechanism that keeps the empirical SCM honest under distribution
shift.

---

## Quickstart

```bash
# 1. Install
pip install -e ".[dev,swebench]"

# 2. Inspect the canonical SCM
causa scm-show

# 3. Single-task dry run with the causal arm (uses MockLLMClient)
causa run --arm causal --task-id swebench/example

# 4. Full evaluation
causa eval \
  --arm causal \
  --tasks data/swebench.jsonl \
  --trace traces/causal.jsonl \
  --parallelism 8

# 5. Audit a finished trace
causa trace-audit traces/causal.jsonl
```

## Six ablation arms

| Arm | Scorer | Reflection | Memory | Purpose |
|---|---|---|---|---|
| **CAUSAL** | DoWhy | adaptive | full | dissertation main claim |
| **CAUSAL_NO_REFLECTION** | DoWhy | none | full | isolates CRM contribution |
| **LLM_SCORER** | LLM-only | none | full | isolates causal identification value |
| **REACT** | LLM (ReAct) | scratchpad | scratchpad | Yao 2023 baseline |
| **COT** | LLM (CoT) | none | none | Wei 2022 baseline |
| **NO_MEMORY** | LLM | none | wiped | ablation floor |

---

## Where to start reading the code

1. **`causa.core.scm`** — the typed SCM algebra. Five minutes to grok.
2. **`causa.core.identifiability`** — the back-door identifier. NetworkX
   under the hood; the function takes a graph and returns an admissible
   set in `O(|V|²)`.
3. **`causa.agents.base.BaseAgent.run`** — the canonical decision loop.
   Read this and you understand the agent shell every arm specialises.
4. **`causa.planning.dowhy_scorer`** — Component 2. The cleanest piece
   of the system.
5. **`causa.reflection.module`** — Component 4. Short, surprising file.

---

## Testing

```bash
pytest                           # all tests
pytest -m unit                   # fast, no I/O
pytest -m integration            # DoWhy round-trip
pytest tests/unit/test_runner.py # one file
```

The integration test (`tests/integration/test_dowhy_roundtrip.py`)
constructs a 3-node `Z → X, Z → Y, X → Y` SCM with a known
interventional effect of +0.5 and verifies the
`LinearRegressionEstimator` recovers it within ±0.05 — the
end-to-end sanity check that wires identifiability, estimation, and
scoring together.

---

## Design principles

- **Causally pure core.** `causa.core` knows nothing about LLMs, files,
  or DoWhy. Everything else depends on the core; the core depends on
  nothing.
- **Ports as Protocols.** Every external resource is a `runtime_checkable
  Protocol` with one or two methods. Adapters are concrete classes;
  swapping them is a single-line change in `causa.cli.runtime`.
- **No magic.** No DI container, no plugin loader, no service registry.
  The dependency graph is constructed in `build_agent` and inspectable
  in 60 lines.
- **Traces over telemetry.** Every decision lands in a JSON Lines event
  that the `TraceAudit` can replay. No mutable summary, no flush
  buffering issues — append-only, schema-versioned, durable.
- **Tests pin behaviour, not implementation.** `test_agents.py`
  exercises the arms through their public hooks; refactoring the loop
  body without breaking behaviour leaves the tests green.

---

## License & attribution

MIT. Cite as:

> Ghosh, A. (2026). *Causal Planning for LLM Agents: A Framework for
> Robust Decision-Making Under Distribution Shift*. M.Tech dissertation,
> BITS Pilani.
