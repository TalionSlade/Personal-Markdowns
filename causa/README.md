# Causa — Causal Planning for LLM Agents

> **Arpan Ghosh · BITS Pilani · M.Tech AI/ML Dissertation · May 2026**
> **Thesis:** *Causal Planning for LLM Agents — A Framework for Robust Decision-Making Under Distribution Shift*

A reference implementation of Pearl's `do(·)`-based action selection inside an LLM agent loop, with online causal-belief update via counterfactual reflection, evaluated against SWE-bench under controlled distribution shift.

## TL;DR — One-Paragraph Read

Today's LLM agents (ReAct, CoT, tool-use scaffolds) are **correlational** at the action-selection layer. They learn what tends to work. Causa replaces that with **interventional** scoring — for each candidate tool `a`, compute `P(tests_passed | do(tool_selected = a))` via DoWhy on a hand-authored Structural Causal Model. After execution, the **Counterfactual Reflection Module** queries the LLM for alternative-outcome estimates and updates the SCM's observation history when an alternative looks materially better than the chosen action. The agent is evaluated on **SWE-bench partitioned along four axes** (language, framework, bug type, codebase size) to measure generalisation under distribution shift.

## Why this is novel

| Existing work | What it does | Gap |
|---|---|---|
| ReAct (Yao+ 2022) | Interleaves reasoning + action | Correlational at action-selection |
| CLadder, Kıcıman+ | LLMs *answer* causal questions | Static text, not active planning |
| Reflexion (Shinn+ 2023) | Verbal self-criticism conditions next prompt | No structured belief update |
| DoWhy (Sharma+ 2020) | Causal inference library | Not inside an agent loop |
| **Causa** | **`do()` scoring + online causal-belief update via the SCM** | — |

## Quickstart

```bash
# 1. Install
cd causa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the synthetic 3-node DoWhy round-trip (no LLM cost)
pytest tests/integration/test_dowhy_roundtrip.py -v

# 3. Run a single causal-agent decision on a stub debugging task
causa run --task examples/null_pointer.json --agent causal

# 4. Compare causal vs ReAct on a held-out partition
causa eval --agents causal,react --partition framework=sklearn --n 50
```

## Repository Layout

```
causa/
├── src/causa/
│   ├── core/         # Pure domain: SCM, graph, intervention, identifiability
│   ├── ports/        # Protocols (LLM, Scorer, Estimator, History, Tool)
│   ├── adapters/     # Concrete impls behind ports
│   ├── domain/       # Software-debugging SCM + tool registry
│   ├── planning/     # Component 2: do-calculus scoring + warm-start
│   ├── reflection/   # Component 3: counterfactual update loop ★ NOVEL
│   ├── extraction/   # Component 1: LLM graph extractor
│   ├── evaluation/   # Component 4: partition + runner + metrics + stats
│   ├── agents/       # CausalAgent + ReAct + CoT + NoMemory baselines
│   ├── telemetry/    # JSON-Lines causal traces, audit
│   ├── cli/          # Typer entrypoints
│   └── config/       # Pydantic Settings
├── tests/
│   ├── unit/         # Pure-domain (no I/O, no LLM)
│   └── integration/  # DoWhy round-trip, eval harness
├── docs/wiki/
│   ├── WIKI.md       # Architecture wiki (the human-readable companion)
│   ├── diagrams/     # 15 HTML diagrams, paper-ready
│   └── images/       # Rendered diagram PNGs
└── scripts/          # One-off operational utilities
```

## Architecture in One Diagram

See **[`docs/wiki/diagrams/01-system-overview.html`](docs/wiki/diagrams/01-system-overview.html)** for the canonical view. The short version:

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  CAUSAL PLANNING AGENT              │
                    │                                                     │
   task description │  ┌─────────┐    ┌──────────┐    ┌────────────────┐  │  action
   ─────────────────▶│  │  Graph  │───▶│ Planning │───▶│ Counterfactual │  │──────────▶
                    │  │Extractor│    │  Layer   │    │   Reflection   │  │
                    │  │  (C1)   │    │   (C2)   │    │     (C3) ★     │  │
                    │  └─────────┘    └────┬─────┘    └───────┬────────┘  │
                    │                     │                  │           │
                    │                     │   DoWhy          │           │
                    │                     ▼                  ▼           │
                    │                ┌─────────┐    ┌────────────────┐   │
                    │                │   SCM   │◀───│  Observation   │   │
                    │                │ (9 nodes│    │    History     │   │
                    │                │ 8 edges)│    │   (pandas)     │   │
                    │                └─────────┘    └────────────────┘   │
                    └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────────┐
                        │      Distribution Shift Eval Suite      │
                        │  SWE-bench × {lang, framework, bug, n}  │
                        │                  (C4)                   │
                        └─────────────────────────────────────────┘
```

## Documentation

The full architectural wiki lives in **[`docs/wiki/WIKI.md`](docs/wiki/WIKI.md)** with 15 paper-ready HTML diagrams in `docs/wiki/diagrams/`. The diagrams are 1700×1100 SVG-driven HTML — screenshot at the same resolution and drop directly into the dissertation.

## Status

| Component | Status | Milestone |
|---|---|---|
| Core domain (SCM, graph, intervention) | ✅ | done |
| Causal Graph Extractor (C1) | ✅ | May 29 |
| Causal Planning Layer (C2) | ✅ | May 29 |
| Counterfactual Reflection Module (C3) | ✅ | Jun 19 |
| Distribution Shift Eval Suite (C4) | ✅ | Jul 24 |
| Baselines (ReAct, CoT, NoMemory) | ✅ | Jun 5 |
| Ablations | scheduled | Jul 24 |

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@mastersthesis{ghosh2026causa,
  author = {Arpan Ghosh},
  title  = {Causal Planning for LLM Agents: A Framework for Robust Decision-Making Under Distribution Shift},
  school = {Birla Institute of Technology and Science, Pilani},
  year   = {2026},
  type   = {M.Tech. dissertation, Artificial Intelligence and Machine Learning}
}
```
