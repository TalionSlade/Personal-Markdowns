# Outline Viva — Complete Prep Manual
**Arpan Ghosh · BITS Pilani M.Tech AI/ML · May 2026**
**Thesis: Causal Planning for LLM Agents — A Framework for Robust Decision-Making Under Distribution Shift**

---

## How to use this document

This is a long document. Three ways to use it:

1. **First read** — read it end-to-end once to internalize the structure.
2. **Practice runs** — use Part 1 as the script for at least three out-loud practice runs of the deck.
3. **Night-before drill** — re-read Part 2 (FAQs) the night before, focusing on the ★★ questions.

The viva runs **12–15 minutes for presentation + ~10–15 minutes Q&A**. Plan for the presentation to land at ~13 minutes so you have buffer.

---

# PART 1 — DETAILED SPEAKER NOTES (slide by slide)

For each slide, the structure is:
- **Opening line** — the literal sentence to start the slide with
- **Core points** — what to say in your own words
- **Key concept to land** — the one idea the examiner should remember
- **Visual cues** — what to point to on the slide
- **Transition** — how to bridge to the next slide
- **If they interrupt** — how to handle being cut off
- **Timing** — target seconds

---

## SLIDE 1 — Title (~20–25 seconds)

### Opening line
> "Good morning, Sir / Good morning, Sirs. Thank you for the opportunity to present my dissertation outline."

### Core points
- State name and BITS ID clearly: "My name is Arpan Ghosh, BITS ID [fill in]."
- Name your supervisor: "My supervisor is [name]."
- State the title and one-line framing:
  > "The dissertation is on **causal planning for LLM agents** — specifically, whether grounding an agent's action selection in Pearl's do-calculus produces more robust behaviour under distribution shift. The empirical setting is software debugging, evaluated against the SWE-bench benchmark."

### Key concept to land
This is positioning, not content. The examiner should walk away from slide 1 knowing: (1) this is about LLM agents, (2) the angle is causal reasoning, (3) the experiment is on software debugging.

### Visual cues
Don't dwell. Move on within 25 seconds.

### Transition
> "Let me start with why this problem matters."

### If they interrupt
On a title slide, they probably won't. If they ask "tell me in one sentence what you're proposing," answer:
> "I'm proposing an LLM agent that picks its next action using do-calculus interventional inference, instead of correlational prompting, and evaluating whether it generalises better when task conditions shift."

### Timing
20–25 seconds. **Do not exceed 30s on the title.**

---

## SLIDE 2 — Background & Research Question (~90–100 seconds)

### Opening line
> "Today's LLM agents are correlation engines at their core."

### Core points

**Paragraph 1 — the problem framing (30s):**
> "ReAct, Chain-of-Thought, and the various tool-use scaffolds in production today share an underlying approach at the action-selection layer. They learn what tends to work — they associate task contexts with action sequences observed during training, and reproduce those associations at inference time. They do not reason about *why* a particular action succeeded or failed."

**Paragraph 2 — the failure mode (30s):**
> "This works fine inside their training distribution. Outside it, performance often falls off sharply. For a concrete example: a debugging agent that handles Python type errors competently may struggle on the same class of bug in Rust, or on Python errors raised inside an unfamiliar framework. The structural reason is that these agents have no model of how their actions affect outcomes — only a frequency count of what has worked before."

**Paragraph 3 — the theoretical grounding (15s):**
> "The causal-inference literature, going back to Pearl in the 1990s, has shown that correlational associations are systematically less stable across distribution shift than the underlying causal mechanisms. This is well-established formally — it's the foundation of why interventional reasoning matters."

**Paragraph 4 — the research question and hypothesis (20s):**
> *(Read the research question aloud, slowly — it's the centerpiece of the slide.)*
> "Does an LLM agent that selects actions via Pearl's do-calculus generalise better under distribution shift than standard correlational agents?"
> "My hypothesis is that causal structure is more stable across distributions than statistical correlations. So an agent grounded in a structural causal model should degrade more gracefully when task conditions change."

### Key concept to land
**Correlational vs causal at the action-selection layer.** That phrase. The examiner should hear it twice.

### Visual cues
Point at the pull-quote when reading the research question. Pause for one full second after reading it — let it land.

### Transition
> "Given that motivation, here's what the dissertation aims to deliver."

### If they interrupt
- **"What do you mean by 'correlation engine'?"** → "I mean the agent's action selection step is a learned mapping from context to action, with no explicit model of how actions cause outcomes. The mapping is statistical, not causal."
- **"How do you know LLMs fail under distribution shift?"** → "Two sources of evidence: empirical results on SWE-bench partitions show large gaps between in-distribution and held-out performance, and the broader OOD-generalisation literature in ML — Geirhos et al., Arjovsky et al. on IRM — establishes this pattern consistently."

### Timing
90–100 seconds. **Do not skip the hypothesis sentence — it's the testable claim.**

---

## SLIDE 3 — Objectives & Scope (~60–70 seconds)

### Opening line
> "Six concrete objectives, with explicit scope boundaries."

### Core points

**Walk through objectives quickly (35s):**
> "Objective 1 — build the Causal Planning Layer using DoWhy for do-calculus action scoring.
> Objective 2 — construct the Structural Causal Model for the software debugging domain.
> Objective 3 — implement the Counterfactual Reflection Module for online belief update. *This is the original contribution.*
> Objective 4 — develop the Distribution Shift Evaluation Suite on SWE-bench.
> Objective 5 — show empirically that the causal agent beats ReAct, CoT, and a no-memory baseline on OOD task variants.
> Objective 6 — produce auditable causal traces and evaluate them as explanations."

**Scope statement (25s):**
> "Scope was tight by design. **In scope**: the four-component system in Python, baselines, ablations, the OOD evaluation suite, and the dissertation write-up.
> **Out of scope**: pre-training new LLMs, automatic causal discovery from data, domains beyond software debugging, and multi-agent variants. Each of these is interesting but would inflate this into a PhD-level project."

### Key concept to land
**Objective 3 is the original contribution.** Emphasize "original" verbally. Examiners need to know what to credit you for.

### Visual cues
When you say "this is the original contribution" on objective 3, briefly pause and look at the examiner.

### Transition
> "Let me get into the methodology — how this is actually built."

### If they interrupt
- **"Why is Counterfactual Reflection 'original'? Doesn't Reflexion do this already?"** → "Reflexion produces unstructured text reflections that condition the next prompt. The Counterfactual Reflection Module updates a structured causal model — specifically, the observational data feeding DoWhy. The next interventional query uses updated parameter estimates. It's a closed causal-learning loop, not just better prompting." *(This is FAQ Q19 — be ready.)*
- **"Why is multi-agent out of scope?"** → "Two reasons. First, it adds coordination problems that are themselves a research area. Second, the single-agent setting is sufficient to test the central hypothesis — does causal grounding improve generalisation. Adding multi-agent doesn't make the test sharper."

### Timing
60–70 seconds. Bullets are dense; don't dwell on each.

---

## SLIDE 4 — Methodology: System Architecture (~80–95 seconds)

### Opening line
> "The system has four components arranged in a closed decision-and-learning loop."

### Core points

**Per-component walkthrough (60s) — point at each box as you describe it:**

> "**Component 1 — Causal Graph Extractor.** Takes a task description, uses an LLM to propose candidate causal edges, returns a directed acyclic graph stored as a networkx structure. For the debugging domain the graph is largely hand-authored; the extractor is what lets the framework generalise to other domains later."

> "**Component 2 — Causal Planning Layer.** This is the heart of the system. For each candidate action, compute P(goal | do(action)) using DoWhy. The agent picks the argmax — the action with the highest interventional expected goal probability. This replaces the LLM's correlational scoring with an explicit interventional one."

> "**Component 3 — Counterfactual Reflection Module.** After each action, the agent queries the LLM: 'would an alternative action have produced a better outcome?' If yes — by a configurable margin θ — the new observation is added to history and the causal beliefs update. This is the online learning loop."

> "**Component 4 — Distribution Shift Evaluation Suite.** Partitions SWE-bench along four axes — language, bug type, framework, codebase size — so we can construct controlled out-of-distribution test conditions. The suite is itself a benchmark contribution applicable beyond this thesis."

**Closing observation (15s):**
> "Each component is independently testable, which matters for the ablation studies. Input is the task state plus candidate tools; output is the next action plus a causal trace explaining the choice."

### Key concept to land
**Closed loop.** The four components don't just sit beside each other — they feed each other. Decision → execution → reflection → updated beliefs → better next decision.

### Visual cues
Point at each numbered box in turn. When you reach Component 4, gesture across all four to indicate the loop.

### Transition
> "Component 2 — the causal planning layer — depends on having a usable causal graph. So let me show you the SCM for the debugging domain."

### If they interrupt
- **"What if the LLM-extracted graph from Component 1 is wrong?"** → "Manual validation is required for the debugging domain. For domain transfer, the extractor's output is a hypothesis, not ground truth. The Counterfactual Reflection Module provides an indirect signal — persistent disagreement between predicted and actual outcomes flags structural errors. DoWhy's refutation tests provide a more formal sensitivity check."
- **"How big is the action space?"** → "For software debugging, the tool set is 6–8 tools: code search, static analyzer, test runner, log inspector, documentation lookup, patch generator, regression checker. So the candidate action space at each step is small — under 10 — which is why exhaustive scoring is feasible."

### Timing
80–95 seconds. **This is the second-longest slide. Practice it.**

---

## SLIDE 5 — Methodology: The Causal Graph (SCM) (~110–130 seconds)

This is the most important slide. **Give it time. Do not rush.**

### Opening line
> "This is the structural causal model for the debugging domain — nine variables, eight directed edges."

### Core points

**Three inputs (15s):**
> "Three observational inputs. These are things the agent sees, not things it controls.
> *(Point to each.)* `error_message_type` — the kind of error encountered. `codebase_structure` — what the relevant repository looks like. `context_available` — what other information is in scope, like the active file or recent edits."

**The intervention point (40s — slow down here):**
> "The chain runs left to right. The two error-related inputs feed into `hypothesis_space` — the agent's current set of beliefs about what might be wrong.
> Hypothesis_space drives `tool_selected`. **And tool_selected is the intervention point.** This is the do() variable — the variable the agent controls.
> *(Pause. Let this land.)*
> When DoWhy computes do(tool_selected = X), it's not asking 'when in the past did tool X correlate with success'. It's asking 'what is the expected outcome if I force the choice to X right now, holding all other context fixed'. That distinction is the entire point of the framework."

**The rest of the chain (25s):**
> "tool_selected, together with context_available, produces `information_gained` — what the agent learns from the tool call. That feeds `root_cause_identified` — whether the agent now knows what's actually wrong. That feeds `patch_quality` — how good the proposed fix is. And patch_quality feeds `tests_passed` — the observable success signal that SWE-bench grades on."

**Provenance and validation (15s):**
> "The graph was hand-authored based on debugging workflow knowledge. It will be validated against real SWE-bench trajectories — comparing predicted causal effects to observed outcomes across many task instances. DoWhy's refutation tests give a more formal sensitivity check."

### Key concept to land
**`tool_selected` is the do() variable. That's where the agent intervenes.** Say this twice. The examiner should leave the slide understanding what "do-calculus on tool_selected" means concretely.

### Visual cues
- When introducing inputs, point to all three at the top
- When introducing tool_selected, point at the teal-highlighted node and the "★ do() intervention" label
- When walking the chain, sweep your finger left to right across the bottom row

### Transition
> "Now that you've seen the SCM, let me show you how the agent actually uses it — the decision loop, plus how we plan to evaluate it."

### If they interrupt
- **"Where does information_gained get its value from? Is that the LLM's judgement?"** → "It's a derived measurement from the tool call's output. For a test runner, did the test reveal new failure information? For a log inspector, did the logs contain a relevant trace? We use a small LLM-judge to score the information content of the tool output."
- **"Why is hypothesis_space a single node? Isn't it actually a distribution?"** → "Conceptually yes — hypothesis_space is a probability distribution over possible root causes. In the SCM it's represented as a categorical variable indicating which hypothesis cluster the agent's beliefs currently concentrate on. We coarsen for tractability; sensitivity analysis will check whether finer discretisation matters."
- **"Are you sure this graph is complete? What about test coverage, code review history, ..."** → "Almost certainly incomplete — every SCM is. The graph captures the variables that most directly mediate tool-choice success. Missing variables become unobserved confounders, which is exactly what DoWhy's refutation tests detect. If a missing variable matters, the back-door adjustment will be biased and the refutation will flag it."

### Timing
110–130 seconds. **The slide where you should be slowest and most deliberate.**

---

## SLIDE 6 — Decision Loop & Evaluation (~85–100 seconds)

### Opening line
> "The decision loop, then the evaluation plan."

### Core points

**Decision loop (40s) — gesture at the dark code block:**
> "For each candidate action, the agent computes score[a] = P(goal | do(a)) using DoWhy. It picks the argmax. It executes the chosen action and observes the actual outcome.
> Then — and this is the novel part — it runs counterfactual reflection. For each alternative action, the agent queries the LLM: 'estimate the outcome that would have resulted if that alternative had been chosen instead.' If any alternative looks materially better than what was actually observed — by a threshold θ — the agent treats that as new evidence and updates the observation history feeding DoWhy.
> Over many decisions, the causal model becomes increasingly well-calibrated to the actual task distribution."

**Evaluation plan (40s) — gesture at the right panel:**
> "Four baselines, including ours. No-memory establishes the LLM floor. ReAct is the canonical comparison and primary baseline. Chain-of-Thought captures pure prompt-based reasoning. Our agent — the Causal Planning Agent — is the proposed contribution.
> Six metrics, but **the primary is out-of-distribution task success**. That's the central empirical claim.
> Secondary metrics: in-distribution success — as a sanity check; recovery rate after a first tool failure — a proxy for adaptability; explanation quality — comparing our causal traces against baselines' CoT rationales, evaluated by both LLM-as-judge and a small human study.
> Two key ablations: with vs without counterfactual reflection, and do-calculus scoring vs LLM-only scoring. These isolate the contribution of each architectural choice."

**Distribution shift framing (15s):**
> "Distribution shift is operationalised along four axes — programming language, bug type, framework, codebase size. The Distribution Shift Evaluation Suite handles partitioning and held-out evaluation automatically."

### Key concept to land
**The ablations are designed to isolate the source of improvement.** This pre-empts the "are gains attributable to your method or to extra compute?" question.

### Visual cues
- Walk through the pseudocode top-to-bottom — your finger follows the lines
- For evaluation, point at the starred ★ primary metric explicitly

### Transition
> "Let me show you where I am right now in the work."

### If they interrupt
- **"What's θ — the threshold?"** → "A hyperparameter. Probably between 0.1 and 0.2 in normalized outcome units. It's tuned on a small held-out set. Too low and you update beliefs from LLM noise; too high and you never update."
- **"Why these baselines and not Reflexion or Voyager?"** → "ReAct is canonical. CoT is the pure-prompting comparison. No-memory is the floor. Reflexion is a strong candidate to add as a fourth baseline if time permits before mid-sem — it's the closest in spirit to my counterfactual reflection."
- **"How do you control for compute differences across agents?"** → "We log LLM token counts per task and report compute-matched results. The argument is that the gain must hold at matched compute — otherwise we haven't shown that causal structure helps."

### Timing
85–100 seconds.

---

## SLIDE 7 — Progress & Key Findings (~40–50 seconds)

### Opening line
> "Here's what's been done so far."

### Core points

**Quick checkmark walkthrough (25s):**
> "The literature survey is complete — ReAct, CLadder, Kıcıman et al., SWE-bench, DoWhy, Think-on-Graph. The SCM for the debugging domain is drafted — 9 nodes, 8 edges. The dissertation outline was submitted on May 15. The development environment is set up — Python 3.11, DoWhy 0.14, networkx, the Anthropic SDK."

**Key validation finding (15s) — emphasize this one:**
> "The most important validation so far: the DoWhy round-trip works end-to-end. I built a synthetic 3-node SCM with a known true average treatment effect of 2.0, ran observational data through DoWhy's Model → Identify → Estimate workflow, and recovered the ATE within 0.3 of the ground truth. This means the toolchain is wired correctly and we can proceed with the actual implementation."

**Current state (10s):**
> "Right now I'm mid-implementation on Components 1 and 2 — the graph extractor and the planning layer. Target is end-to-end working by May 29."

### Key concept to land
**The toolchain is validated.** This pre-empts skepticism that the project is too ambitious to start.

### Visual cues
Point at the teal "Currently" strip when you reach the current state.

### Transition
> "Which brings me to the plan for the rest of the semester."

### If they interrupt
- **"Why only a 3-node SCM for the validation?"** → "It's a minimal sanity check — does the toolchain run, does it produce numerically reasonable estimates. The full 9-node SCM will be exercised during baseline evaluation in early June. A larger synthetic test isn't more informative for the toolchain question; it's more informative for parameter calibration, which is the next milestone."

### Timing
40–50 seconds. This is a status slide — quick.

---

## SLIDE 8 — Future Work / Plan for Remaining Semester (~45–55 seconds)

### Opening line
> "The plan for the rest of the semester, organized around the two assessments."

### Core points

**Walk the table top to bottom (35s):**
> "Components 1 and 2 — the causal graph and DoWhy integration — by May 29.
> Baseline evaluation runs in parallel: ReAct, CoT, and no-memory on a SWE-bench subset by June 5. This gives me a comparison point early.
> Components 3 and 4 — the counterfactual module and the distribution-shift suite — by June 19, with an end-to-end demo showing the full causal agent against ReAct on a real debugging task.
> Mid-semester evaluation runs June 27 to July 3 — I'll present baseline-vs-causal results.
> Ablations and the distribution-shift experiments through July.
> Writing in late July through early August.
> Viva in mid-August."

**Closing emphasis (10s):**
> "The immediate next milestone is Components 1 and 2 working end-to-end by May 29 — that's the gating step for everything downstream."

### Key concept to land
**There's a clear, time-bounded, low-risk plan.** Examiners worry about students who don't have one.

### Visual cues
Move your finger down the rows of the table as you describe each phase.

### Transition
> "That's the plan. Happy to take questions."

### If they interrupt
- **"What if you slip on Components 1 and 2?"** → "The biggest risk to the timeline is DoWhy integration taking longer than expected. Mitigation: a smaller SCM (5–6 nodes) is enough to run the baseline comparison, so the timeline survives a partial implementation. The full 9-node graph is required only for the final dissertation, not for mid-sem results."
- **"You have parallel phases — is that realistic?"** → "Yes — baseline evaluation is largely independent of the causal agent's implementation. It can run on different hardware while I'm building the core system."

### Timing
45–55 seconds.

---

## SLIDE 9 — Q&A / Thank You (~10 seconds before Q&A)

### Opening line
> "That's the outline of my dissertation. I would welcome your questions."

### Core points
- Don't elaborate. The next phase is theirs.
- Slight pause for the examiner to take the floor.

### Visual cues
- Don't fidget. Stand still. Look at the examiner.
- If multiple examiners, look at the senior one first.

### If the room is silent for >5 seconds
Optional warm-up line: "I'm happy to go deeper on any of the components — the causal graph and the decision loop are the most important pieces."

### Timing
10 seconds, then yield to Q&A.

---

# PART 2 — COMPREHENSIVE FAQ BANK

> **Legend:** ★★ = certain to be asked. ★ = highly likely. (no star) = possible.
> Read all of them at least once. Memorize the ★★ answers.

## SECTION A — Foundational Causal Inference

### A1 ★★ — "Explain Pearl's ladder of causation."

> Three rungs.
> **Rung 1 — Association** (seeing). P(Y|X) — what we observe when X happens. All standard ML lives here.
> **Rung 2 — Intervention** (doing). P(Y|do(X)) — what happens if we *force* X to a value, ignoring its normal causes. This requires either an experiment or a causal model. My agent operates here at the planning layer.
> **Rung 3 — Counterfactual** (imagining). What would Y have been if X had been different, holding everything else identical to the actual world? This is what the Counterfactual Reflection Module asks the LLM to estimate.
>
> Critical detail: each rung strictly contains the one below. You can answer rung-1 questions from rung-2 information but not the reverse.

---

### A2 ★★ — "Define a Structural Causal Model formally."

> A Structural Causal Model is a tuple (U, V, F, P(U)) where:
> - **U** is a set of exogenous (unobserved) noise variables
> - **V** is a set of endogenous (observed) variables
> - **F** is a set of structural equations — for each V_i in V, a function V_i = f_i(parents(V_i), U_i)
> - **P(U)** is a probability distribution over the noise variables
>
> The SCM induces a directed graph where each V_i has an incoming edge from each of its parents. Interventions do(V_j = v) replace the structural equation for V_j with the constant v, severing incoming edges to V_j. This is the "surgical" view of intervention in Pearl's framework.
>
> My debugging SCM is Markovian — noise terms are independent — which simplifies identification. The 9 endogenous variables and 8 directed edges define the causal structure.

---

### A3 ★★ — "What's the difference between an SCM and a Bayesian network?"

> A Bayesian network is a graph plus a joint distribution that factorises according to the graph. It encodes conditional independence but says nothing about interventions — you can compute P(Y|X=x) but not P(Y|do(X=x)).
>
> An SCM is strictly stronger. It adds structural equations specifying *how* each variable is generated from its parents. This lets you answer interventional questions (rung 2) and, with enough information about the noise terms, counterfactual questions (rung 3).
>
> Every SCM induces a Bayesian network, but not every Bayesian network corresponds to a unique SCM — many SCMs can produce the same observational distribution while disagreeing about interventions.

---

### A4 ★ — "Explain the back-door criterion."

> A set of variables Z satisfies the back-door criterion relative to an ordered pair (X, Y) if:
> 1. Z blocks every path between X and Y that contains an arrow into X (a "back-door path"), and
> 2. Z contains no descendants of X.
>
> If such a Z exists, the causal effect P(Y|do(X)) is identifiable from observational data:
> P(Y|do(X=x)) = Σ_z P(Y|X=x, Z=z) P(Z=z)
>
> This is the back-door adjustment formula. DoWhy uses it as the default identification strategy when the graph admits it. In my SCM, when scoring do(tool_selected = X), the back-door set typically includes hypothesis_space and the inputs feeding it.

---

### A5 — "What's the front-door criterion? Would you ever need it?"

> The front-door criterion is used when there are unobserved confounders that prevent back-door adjustment, but there exists a mediator M such that:
> 1. M intercepts all directed paths from X to Y
> 2. There's no back-door path from X to M
> 3. All back-door paths from M to Y are blocked by X
>
> Then P(Y|do(X)) = Σ_m P(M=m|X=x) Σ_x' P(Y|M=m, X=x') P(X=x')
>
> Whether I'd need it: in the debugging SCM, if context_available isn't fully observed and acts as a hidden confounder between tool_selected and information_gained, front-door adjustment via the tool's intermediate output could identify the effect. It's a contingency, not a primary plan.

---

### A6 ★ — "State Pearl's three rules of do-calculus."

> These are the inference rules for transforming interventional expressions:
> **Rule 1** (insertion/deletion of observations): If a set Z blocks all paths from W to Y in the modified graph G_X̄ (with incoming edges to X removed), then P(Y|do(X), Z, W) = P(Y|do(X), Z).
> **Rule 2** (action/observation exchange): If Z d-separates W from Y in the graph G_X̄W̲ (incoming edges to X removed, outgoing from W removed), then P(Y|do(X), do(W), Z) = P(Y|do(X), W, Z).
> **Rule 3** (insertion/deletion of actions): If there are no causal paths from W to Y in G_X̄ that aren't blocked by Z, then P(Y|do(X), do(W), Z) = P(Y|do(X), Z).
>
> Together these are complete for identifiability — Shpitser & Pearl proved that if a causal effect is identifiable from a graph plus observational data, these three rules suffice to derive the formula.

---

### A7 — "What's identifiability and why does it matter?"

> A causal effect P(Y|do(X)) is *identifiable* from a graph G plus observational data if there's a unique value consistent with every distribution that factorizes according to G. Identifiability is a graph-theoretic property — you can determine it from the graph alone, without seeing data.
>
> Why it matters: if an effect isn't identifiable, no amount of data will give you the right answer. DoWhy's `identify_effect` call checks identifiability before estimation. If it fails, the framework reports that, and we fall back to LLM scoring rather than producing a confidently wrong causal estimate.

---

### A8 — "What's a Markovian vs non-Markovian SCM?"

> Markovian = all exogenous noise terms are independent. The graph alone fully captures the dependence structure. Identifiability is easier — every causal effect is identifiable from observational data plus the graph.
>
> Non-Markovian = noise terms may be correlated, which is equivalent to having unobserved confounders. Identifiability becomes harder; some effects are not identifiable, and we need additional assumptions or instrumental variables.
>
> My debugging SCM is treated as Markovian for the planning layer. The risk is that real debugging has unobserved confounders (developer skill, prior task exposure, model fine-tuning state). The Counterfactual Reflection Module and DoWhy refutation tests partially compensate.

---

## SECTION B — Implementation Specifics

### B1 ★★ — "Walk me through what happens, step by step, when the agent makes one decision."

> Step 1: The agent receives the current state — error message, codebase context, available tools.
> Step 2: The agent enumerates candidate actions — typically the 6–8 tools in its toolset.
> Step 3: For each candidate action `a`, the agent constructs a DoWhy `CausalModel` with the current observation history as data, the SCM as graph, treatment = tool_selected = a, outcome = tests_passed (or a proxy).
> Step 4: For each model, DoWhy runs `identify_effect()` to derive the back-door estimand, then `estimate_effect()` to compute P(goal | do(tool_selected = a)) numerically.
> Step 5: The agent picks `argmax_a score[a]` and executes that tool.
> Step 6: After execution, the agent observes the actual outcome.
> Step 7: For each alternative action, the agent queries the LLM for a counterfactual estimate.
> Step 8: If any alternative's estimate exceeds the actual outcome by more than θ, that observation is added to history.
> Step 9: The loop continues until the task succeeds or the step budget is exhausted.

---

### B2 ★ — "What estimator does DoWhy use by default? Why?"

> For back-door identification with a continuous outcome, DoWhy defaults to linear regression. For binary treatment with continuous outcome, propensity score matching is also available. The default is configurable per-call.
>
> Why linear regression by default: it's fast, well-understood, and works with the small data sizes typical at decision time. As observation history grows, we'll evaluate switching to gradient-boosted or doubly-robust estimators via DoWhy's econml backend.

---

### B3 — "How do you handle continuous variables in your SCM?"

> Most variables in the debugging SCM are categorical or ordinal. `error_message_type` is categorical (~10 classes). `hypothesis_space` is a categorical coarsening of the hypothesis distribution. `tool_selected` is categorical (one of 6–8 tools). `tests_passed` is binary.
>
> The genuinely continuous variables are scores: information_gained and patch_quality are real-valued in [0, 1]. For these, DoWhy uses standard continuous-outcome estimators (linear regression by default).
>
> Discretising continuous concepts has obvious downsides — we lose information. The sensitivity analysis will run several discretisation schemes and report whether results are robust.

---

### B4 — "What's the data structure for the observation history?"

> A pandas DataFrame with one row per observed agent step. Columns correspond to the SCM variables — error_message_type, codebase_structure, context_available, hypothesis_space, tool_selected, information_gained, root_cause_identified, patch_quality, tests_passed. Each row records the values at the time of one decision plus the resulting outcome.
>
> The DataFrame grows append-only as the agent acts. DoWhy queries it on every decision. For long runs we'll add windowing — only the last K observations — to keep DoWhy queries fast.

---

### B5 ★ — "How big does the observation history need to be for reliable estimates?"

> For the simple back-door adjustment with low-cardinality categorical variables, sample sizes of ~50–100 per cell of the adjustment set typically give reasonable estimates. Across the cells of our adjustment set (~6 cells), that's a few hundred observations total.
>
> Below 10 observations, DoWhy estimates are unreliable. That's the threshold for the cold-start fallback to LLM scoring. Between 10 and ~50, we run DoWhy but treat estimates as noisy — possibly with wider effective tolerance in the argmax decision.
>
> This is one of the tuning parameters that ablation studies will report.

---

### B6 — "What's the time complexity of one agent decision?"

> Per decision: O(|A| · T_DoWhy + T_LLM_counterfactual · |A|), where |A| is the number of candidate actions (~8) and T_DoWhy is the time for one identification + estimation call on the current history.
>
> T_DoWhy scales roughly linearly in history size for the linear regression estimator. For a history of 100 observations and 8 candidate actions, expect ~200–500ms per decision plus LLM call latency. LLM calls dominate.
>
> Optimisation potential: cache the identified estimand (it only depends on the graph, not the data) and reuse across calls. This is a low-effort optimisation we'll add early.

---

### B7 — "How do you handle missing data in observations?"

> Two cases. For variables that the agent didn't measure during a particular step (e.g., context_available wasn't recorded), DoWhy supports complete-case analysis — only rows with all required variables are used.
>
> For partially-observed scenarios where a variable is sometimes available and sometimes not, we use multiple imputation via DoWhy's interfaces. This adds a small bias but keeps sample sizes usable.
>
> The dissertation reports results both with complete-case-only and with imputation to demonstrate that conclusions are robust to the choice.

---

### B8 — "How do you bootstrap the observation history before the agent has acted?"

> Two strategies, used jointly. **(1)** The LLM is prompted to generate synthetic prior observations consistent with the SCM — given the graph, the LLM produces hypothetical task instances with realistic variable assignments. This seeds the history. **(2)** Below ~10 real observations, the agent falls back to LLM-only scoring; once enough real observations accumulate, DoWhy takes over.
>
> The synthetic prior is a known limitation — it's the LLM's prior, not the true data distribution. Sensitivity analysis will measure how much the synthetic prior affects final decisions, and the ablation comparing with and without synthetic priors quantifies its contribution.

---

## SECTION C — The Agent Loop & Decision Making

### C1 — "Why argmax? Why not sample from the score distribution?"

> Argmax is the deterministic exploitation strategy. In the current design, the LLM scoring fallback (during cold start) provides exploration; once DoWhy takes over, the policy is greedy w.r.t. the interventional expected utility.
>
> Sampling-based exploration — Thompson sampling, epsilon-greedy — is a natural extension and is mentioned in the future-work section. For the M.Tech scope, argmax is sufficient to test the central hypothesis (does causal scoring beat correlational scoring).

---

### C2 — "What if two actions have equal scores?"

> Ties are broken by LLM preference: among tied actions, the agent picks the one the LLM ranks highest. This rarely matters in practice — score equality is unusual with continuous estimates — but matters for reproducibility.
>
> Tie-breaking is logged so we can audit decisions.

---

### C3 ★ — "How does the agent know when to stop trying tools and emit a patch?"

> The same scoring mechanism. The patch_generator is itself an action in the candidate set. When its interventional score exceeds the score of further investigation tools, the agent generates the patch and stops.
>
> There's also a hard step budget — typically 12 tool calls per task — after which the agent must emit its best guess. This matches the SWE-bench evaluation protocol.

---

### C4 — "What happens if all candidate actions have low scores?"

> Two cases. **If history is small** — fewer than ~10 observations — the agent's DoWhy estimates are noisy and absolute scores aren't meaningful; the agent uses argmax of relative scores. **If history is large**, low absolute scores indicate either an out-of-distribution task or genuine difficulty. The agent emits the best-of-bad-options action and the counterfactual reflection step will flag this as low-confidence for downstream analysis.

---

### C5 — "Does the agent ever explore tools it hasn't tried before?"

> Yes — every tool starts with non-zero synthetic prior observations, so DoWhy can score it. As more real observations accumulate, scores for under-explored tools have higher variance, which the upper-confidence-bound-style scoring (a future extension) would account for.
>
> For the M.Tech scope, the synthetic prior is the only exploration mechanism. This is a deliberate simplification that the dissertation will note as a limitation.

---

## SECTION D — Counterfactual Reflection (the novel contribution)

### D1 ★★ — "How exactly does the Counterfactual Reflection Module work?"

> After the agent has taken action `a` and observed outcome `y`, the module:
> 1. Identifies the set of alternative actions A' that could have been taken instead.
> 2. For each alternative `a' ∈ A'`, constructs a prompt asking the LLM to estimate the expected outcome `y'` had `a'` been chosen, given the observed state.
> 3. Compares each `y'` against the actual `y`. If `y' - y > θ` (the threshold), the difference is treated as evidence that the SCM-implied scoring underestimated `a'`.
> 4. Adds a synthetic-like observation to the history representing this counterfactual evidence, weighted by a confidence factor.
> 5. The next DoWhy call uses the updated history.
>
> This is the "online belief update" loop. It's distinct from chain-of-thought self-reflection because the update lands in the SCM's data, not the prompt.

---

### D2 ★ — "How is this different from Reflexion (Shinn et al., 2023)?"

> Reflexion produces unstructured verbal self-criticism that conditions the next prompt. There's no persistent structure being updated — each reflection is a piece of text appended to context.
>
> The Counterfactual Reflection Module produces a structured update to the SCM's parameter estimates. Specifically, the new observation enters the DataFrame that DoWhy reads. The next interventional query uses updated data, producing a different score. The update is durable and compounds across many tasks.
>
> Also: Reflexion runs at task boundaries (after a full episode). Counterfactual Reflection runs at step boundaries (after every action). Higher resolution feedback.

---

### D3 — "What if the LLM's counterfactual estimates are systematically biased?"

> Real risk. LLMs are known to be over-confident and to agree with leading questions. Three mitigations:
> 1. **Threshold θ** suppresses small updates driven by LLM noise.
> 2. **Sample multiple counterfactual estimates per alternative** and use the median — reduces variance from LLM stochasticity.
> 3. **Ground-truth validation** against SWE-bench: whenever a SWE-bench task has known correct actions, the LLM's counterfactual estimates can be compared to actual outcomes. This calibration step quantifies bias.
>
> The dissertation will report calibration curves showing systematic bias if any exists.

---

### D4 — "Is the LLM's counterfactual estimate even meaningful? It hasn't run the alternative."

> The LLM is using its training distribution as a proxy for what would have happened. This is a strong assumption — it's only valid to the extent that the LLM has accurate intuitions about debugging outcomes.
>
> The empirical question is whether these LLM-derived counterfactual estimates, even if individually noisy, contain useful information in aggregate. Reflection-based methods like Reflexion have shown that they do, at least in some domains.
>
> If they don't — that's an important negative result and the dissertation reports it. The ablation comparing "with counterfactual reflection" vs "without" is the empirical test.

---

### D5 — "How often does the counterfactual reflection actually trigger a belief update?"

> Empirical question. Expected behaviour: early in an episode, when the agent has little data and the LLM is confident, updates trigger frequently. As the SCM gets calibrated, the agent's argmax aligns with the LLM's preferences and triggers become rare.
>
> The dissertation will report the rate of triggered updates over time per episode and across episodes.

---

## SECTION E — Distribution Shift Theory & Evaluation

### E1 ★★ — "Define 'distribution shift' formally in your evaluation."

> Let p_train(x, y) be the joint distribution of task instances and outcomes the agent was trained on or warm-started against, and p_test(x, y) be the distribution at test time. Distribution shift means p_train ≠ p_test.
>
> In my evaluation, the shift is along structured axes: programming language, framework, bug type, codebase size. The training partition holds tasks of one type (e.g., Python + Django); the test partition holds tasks of another type (e.g., Java + Spring). The marginal P(x) differs sharply across partitions.
>
> The hypothesis is that the SCM's causal mechanisms are invariant across these partitions even though the marginal P(x) changes — and that the agent's performance reflects this invariance.

---

### E2 ★★ — "How exactly do you construct OOD splits on SWE-bench?"

> SWE-bench has ~2,300 issues across 12 repositories. We partition along four axes:
> - **Language**: Python (the bulk) vs other languages where available. Within Python, framework subgroups.
> - **Framework**: Django, Flask, scikit-learn, sympy, etc. — each as a partition.
> - **Bug type**: type errors, logic errors, performance, race conditions — classified via static analysis of the fix patches.
> - **Codebase size**: small (<10k LoC), medium, large (>100k LoC).
>
> For each axis, we hold out one partition as the test set, warm-start the agent on the remaining partitions, evaluate on the held-out partition. We report in-distribution vs out-of-distribution performance gaps.
>
> The Distribution Shift Evaluation Suite automates this partitioning and the evaluation harness.

---

### E3 ★ — "Why would causal structure generalize when correlations don't?"

> Pearl's theoretical argument: causal mechanisms — the structural equations in an SCM — are properties of the underlying data-generating process. They don't depend on the marginal distribution of the inputs. If the mechanism `f(parents) → variable` is invariant across distributions, then so is P(Y|do(X)).
>
> By contrast, correlational quantities P(Y|X) depend on the joint distribution, which can change arbitrarily under shift. A correlation that holds in one distribution can flip in another.
>
> The empirical question is whether, in software debugging, the causal mechanisms are stable enough across the partition axes for this theoretical argument to deliver measurable gains. The thesis tests this.

---

### E4 — "What if the SCM itself isn't stable across distributions?"

> Then the framework's central claim weakens. The Counterfactual Reflection Module partially compensates — it updates beliefs from observed outcomes — but doesn't fix a structural mismatch.
>
> The dissertation reports a domain-shift ablation: train the SCM and parameters on one partition, evaluate on another, with and without belief updates from the test partition. If the SCM is unstable, the "no updates" condition should perform poorly while "with updates" does better.

---

### E5 — "Are SWE-bench partitions actually different distributions, or just different samples from the same distribution?"

> Empirically different. SWE-bench tasks from sklearn look very different from tasks from sympy — different code style, different dependency structures, different common bug patterns. The marginal distributions of variables like error_message_type and codebase_structure differ substantially across partitions.
>
> Whether the causal mechanisms (the structural equations) differ is the empirical question.

---

### E6 — "Is your distribution-shift framework equivalent to domain adaptation?"

> Related but distinct. Domain adaptation typically assumes a source and target domain and aims to learn a transformation between them. My setup is closer to **invariant prediction** (Peters et al., IRM, Arjovsky et al.) — find a representation or mechanism that's invariant across multiple training environments, and test whether that invariance holds on a held-out environment.
>
> The SCM is the candidate invariant structure. The do-calculus query is the prediction mechanism that should be invariant if the SCM is correctly specified.

---

## SECTION F — Software Debugging Domain

### F1 ★ — "Why software debugging specifically? The framework looks domain-agnostic."

> The framework *is* domain-agnostic — that's part of its value. Debugging was chosen for four reasons:
> 1. **SWE-bench exists**. Real GitHub issues, real fixes, real test harnesses. No benchmark to build.
> 2. **Tool use has genuine causal weight**. The order matters: running tests before reading logs versus after gives different information.
> 3. **Distribution shift is naturally well-defined** along language, framework, codebase axes.
> 4. **Industrial relevance**. Strong career signal.
>
> Alternative domains — scientific reasoning, automated theorem proving, robotic task planning — would all be natural follow-ups.

---

### F2 — "Has anyone applied causal reasoning to debugging before?"

> Not in the LLM-agent setting with do-calculus action selection that I've found. There's classical work on **fault localisation** using statistical methods (Tarantula, OchIAI), and some work on **causal slicing** for program understanding. There's also recent work on **counterfactual program reasoning** for explanation.
>
> But applying Pearl's interventional framework to an LLM agent's action selection at inference time, with online belief update via counterfactual reflection — that's the gap this dissertation fills, as far as my literature survey goes.

---

### F3 — "Aren't software bugs too variable to model with 9 causal variables?"

> The 9 variables aren't a complete model of debugging — they're the variables that mediate tool-choice success. Many sources of variation (developer skill, time pressure, codebase familiarity) are abstracted away because they don't directly affect which tool is the right next call.
>
> The framework would extend to richer SCMs straightforwardly — the planning layer is graph-agnostic. The 9-variable choice is a pragmatic starting point for the M.Tech timeline.

---

### F4 — "Why not formal verification or proof assistants? They're more 'causal' in a sense."

> Formal verification proves properties of code; it doesn't help an agent decide which tool to call when investigating a bug. The two are complementary, not alternatives. A causal-planning agent could, in principle, *use* a verifier as one of its tools — calling it as an action when its causal score is high.
>
> Adding formal tools to the agent's toolset is a natural extension. It's out of scope for this dissertation.

---

## SECTION G — SWE-bench & Evaluation

### G1 ★ — "SWE-bench has been reported to have train-test contamination with LLM training data. How do you address that?"

> Real concern. Three responses:
> 1. **SWE-bench-Verified** is a curated subset designed to mitigate contamination. We'll use it as a primary test bed.
> 2. **The contamination concern affects absolute scores, not relative comparisons.** All baselines and our agent use the same LLM and see the same potentially-contaminated tasks. Relative improvements remain meaningful.
> 3. **OOD partitions provide additional protection** — held-out language/framework subsets are less likely to be cleanly memorised even if seen in pretraining.

---

### G2 — "Why not also evaluate on HumanEval, MBPP, BugsInPy, etc.?"

> Time-limited. SWE-bench is the best fit because it tests multi-step tool use, which is where the causal planning argument applies. HumanEval and MBPP are single-shot code generation — there's no tool-selection decision to optimise.
>
> BugsInPy could be added as a secondary benchmark for cross-validation if time permits. The dissertation flags this as future work.

---

### G3 ★ — "What's your sample size? Is it statistically powered to detect realistic effect sizes?"

> Planned: 200–500 tasks per partition for the OOD evaluation, depending on partition size. Pre-registered effect sizes: a 5+ percentage-point improvement in OOD success rate is the meaningful-impact threshold. With ~300 paired comparisons (same task, different agent), this is detectable at p<0.05 with reasonable power for effect sizes ≥ 5pp.
>
> Power calculations will be included in the dissertation's methodology section. If power is insufficient, we'll either expand sample size or weaken the claim.

---

### G4 — "How do you control for the LLM's underlying capability?"

> All agents use the same LLM — same provider, same model version, same temperature. The only difference is the orchestration layer: ReAct's loop, CoT's prompting, our causal planning + reflection.
>
> Sensitivity analysis: rerun key comparisons with a different LLM (e.g., a smaller model) to verify that the conclusion holds across LLM capability levels.

---

### G5 — "How do you measure 'recovery rate after first tool failure'?"

> Filter to tasks where the agent's first tool call returned no useful information (empty result, irrelevant output, etc.). Compute the success rate on this filtered set per agent. Higher recovery rate means the agent adapts better when its initial plan doesn't pan out.
>
> Operational definition of "no useful information" is given by a fixed scoring rubric applied uniformly across agents — to avoid cherry-picking.

---

## SECTION H — Baselines & Comparison

### H1 — "Why ReAct? It's three years old at this point."

> ReAct is the canonical reasoning-and-acting baseline and remains the most-cited LLM agent architecture. More recent agents (Reflexion, Voyager, ToolFormer) are variants of the same core pattern — interleaved reasoning and action.
>
> The architectural argument I'm making is against the *correlational scoring* of the ReAct family, not against ReAct specifically. Beating ReAct is the necessary first step; beating later variants is a natural extension.

---

### H2 — "Will you also compare against Reflexion?"

> Yes, time permitting. Reflexion is the strongest extension to add because it's the closest in spirit to my Counterfactual Reflection Module — both involve self-reflection driving future behaviour. Showing causal reflection beats verbal reflection at matched compute is a strong differentiation result.
>
> If implementation slips, Reflexion is the first baseline to defer.

---

### H3 ★ — "How do you ensure baselines are fair? You designed your agent; you might prompt-engineer it more than the baselines."

> Three controls:
> 1. **Prompt parity**: the system prompt, task description format, and tool descriptions are identical across agents. Only the agent's internal logic differs.
> 2. **Compute parity**: total LLM tokens per task are logged. Comparisons are reported both head-to-head and compute-matched.
> 3. **Pre-registration**: the comparison protocol is fixed before running OOD experiments, so we can't tune the baselines down to make ours look better.
>
> The compute-matched comparison is especially important because our agent makes additional LLM calls in counterfactual reflection.

---

### H4 — "What if all baselines fail badly? Then your agent winning isn't impressive."

> Valid concern. We'll report absolute baseline performance per partition. If baselines achieve, say, 5% success on a partition and we achieve 8%, the absolute numbers are weak even if the *relative* gain is real. The dissertation will discuss both relative and absolute results.
>
> The OOD comparison is the cleanest test — even if absolute numbers are low, the *generalisation gap* (in-dist minus OOD) is meaningful: a smaller gap implies more robust generalisation.

---

## SECTION I — Novelty Defense (the critical questions)

### I1 ★★★ — "Recent work — Kıcıman et al., CLadder — shows LLMs already do causal reasoning when prompted. So what's new about your approach?"

> **This is the most important question. Memorize the answer.**
>
> Kıcıman et al. and CLadder demonstrate that LLMs can *answer* causal questions passively — given a prompt, they can identify confounders, perform counterfactual reasoning on text problems, and so on. The work establishes that LLMs have latent causal capabilities.
>
> What's novel in my approach is **active causal planning** — the agent uses a maintained Structural Causal Model to run interventional queries on its own decision process, in real time, at every action selection step. Three concrete differences:
>
> 1. **Architecturally**, the do-calculus computation happens in DoWhy, not in the LLM's forward pass. The LLM's causal reasoning is involved as a fallback during cold start, but the primary action selection is algorithmic.
> 2. **The reasoning is auditable** through the causal trace — examiners can inspect exactly which interventional query produced each action. The LLM's chain-of-thought isn't reliably faithful to its actual decision process.
> 3. **The system should produce verifiable improvements under distribution shift**, because the SCM's mechanisms are invariant across distributions in a way that prompting patterns are not. This is the central empirical test.
>
> The ablation comparing do-calculus scoring vs LLM-only scoring is specifically designed to settle this question. If LLM-only scoring matches or beats do-calculus scoring, the novelty claim fails and the dissertation reports that honestly.

---

### I2 ★★ — "Could you achieve the same effect with better prompting alone? CoT with a causal template, say?"

> Possibly partially, and that's why the ablation matters. The CoT baseline includes a causal-style prompt template — it's a strong correlational reasoner. The question is whether *structured* causal inference (do-calculus, identifiability checks, observation-driven estimation) adds value beyond a well-prompted LLM.
>
> Theoretical argument for value: prompting depends on the LLM's training distribution — what causal patterns it has seen. Do-calculus depends on the SCM's structure, which is independent of the LLM's training. Under distribution shift, the prompting-only approach should degrade more than the SCM-based approach.
>
> Empirical argument: the OOD evaluation directly tests this. If CoT-with-causal-prompt matches our agent on OOD, then prompting was sufficient and the structural machinery didn't earn its complexity. We'd report that result.

---

### I3 ★ — "Causal world models in RL (Bareinboim, Forney) already exist. Isn't your work a direct application of that?"

> Related but distinct. Bareinboim's work establishes causal foundations for RL — when can a policy learned in one MDP transfer to another, what's identifiable from off-policy data, etc. The contributions are theoretical: identifiability conditions, transfer learning bounds.
>
> My work uses LLMs as the agent (not a tabular or neural policy) and applies these ideas at *inference time* (not training). The specific novelty is the architecture: how to plug do-calculus scoring into an LLM agent's action selection loop and how to update the causal model from LLM-generated counterfactuals.
>
> Bareinboim's framework provides the theoretical justification for why the approach should work; the dissertation contributes the engineering and empirical evaluation in the LLM-agent setting.

---

### I4 — "Counterfactual reasoning + LLMs has been done. See, e.g., Cladder, CounterCurate. How is this different?"

> Cladder and CounterCurate test whether LLMs can *answer* counterfactual questions in static text settings. They're benchmarks, not architectures.
>
> The Counterfactual Reflection Module *uses* the LLM's counterfactual reasoning as input to a structured belief update over an SCM. The novelty is the integration: structured causal model + LLM counterfactual generator + online update loop. None of the cited work does all three together for an LLM agent.

---

### I5 — "Is this really 'planning'? Sounds like one-step greedy action selection."

> One-step greedy is what the current design does, yes. Multi-step planning — say, MCTS over interventional rollouts — is a natural extension and is mentioned in future work.
>
> The framing "causal planning" is justified because the do-calculus query at each step is interventional, not reactive. The agent is reasoning about the consequences of its actions, even if only one step ahead. Calling it "causal action selection" would be more precise; "causal planning" is the more recognisable term in the agentic AI literature.
>
> I'll be careful with this in the dissertation — the title may need a slight refinement after results are in.

---

## SECTION J — Ablations & Methodology

### J1 ★ — "What's the null hypothesis? When would you say your approach failed?"

> Null: the Causal Planning Agent's OOD success rate equals or under-performs the strongest baseline (likely ReAct) at matched compute.
>
> The dissertation rejects this null if the causal agent shows a statistically significant (p < 0.05) improvement of at least 5 percentage points on out-of-distribution task success rate, averaged across OOD partitions.
>
> Failure to reject is a real outcome. The dissertation reports it honestly and discusses whether the framework has narrower domains of applicability or whether the central hypothesis was wrong.

---

### J2 — "What confidence intervals do you report?"

> Bootstrap 95% confidence intervals on per-partition success rates, with N≥1000 bootstrap iterations. Paired bootstrap for direct agent-vs-agent comparisons (same task, different agent), which is more powerful than independent comparisons.
>
> Significance tests via paired permutation testing where applicable; Wilcoxon signed-rank as a sanity check.

---

### J3 ★ — "List all ablations you plan to run."

> Six ablations:
> 1. **With vs without Counterfactual Reflection Module** — quantifies the contribution of online belief update.
> 2. **Do-calculus scoring vs LLM-only scoring** — isolates the structured-causal-inference contribution.
> 3. **Cold-start threshold variation** — how does the LLM-to-DoWhy transition point affect results?
> 4. **Synthetic prior strength** — how much do the LLM-generated warm-start observations matter?
> 5. **SCM coarseness** — coarser vs finer discretisation of continuous variables.
> 6. **Counterfactual threshold θ** — how sensitive are results to the update-trigger threshold?
>
> Each ablation is run on the same OOD evaluation suite.

---

### J4 — "Aren't your ablations going to multiply your compute cost? Can you afford to run all of them?"

> Yes, ablations are expensive. Mitigation: each ablation runs on a smaller task subset (50–100 tasks) sufficient for detecting moderate effects, with the full evaluation only for the main comparison. Total ablation compute is estimated at ~3x the main comparison.
>
> Anthropic API costs and compute budget are tracked weekly. If costs exceed budget, the lowest-priority ablations (5 and 6 in the list above) get deferred.

---

### J5 — "How do you avoid p-hacking? You'll be running many comparisons."

> Pre-registration. The OOD comparison protocol — agents, partitions, metrics, statistical tests — is fixed before running the final experiments. Ablations are pre-specified.
>
> Multiple-comparison correction (Bonferroni or Benjamini-Hochberg) is applied across ablations.
>
> Any post-hoc analyses are clearly labeled as exploratory in the dissertation.

---

## SECTION K — Risks & Limitations

### K1 ★★ — "What's the biggest risk to this dissertation?"

> Honestly, two.
>
> **Risk 1: The SCM is too coarse.** Debugging is more chaotic than 9 variables can capture, and the agent might never accumulate enough observations within a single task trajectory for DoWhy to produce stable estimates. The synthetic-prior warm-start is the primary mitigation. If this fails, the dissertation discusses what richer SCMs would look like.
>
> **Risk 2: LLM counterfactual estimates are systematically biased.** LLMs tend toward over-confidence and toward agreeing with leading questions. If counterfactual estimates are biased, the belief update corrupts the SCM's parameters. Mitigations: threshold θ, multi-sample estimation, ground-truth validation against SWE-bench outcomes. The dissertation reports calibration curves explicitly.

---

### K2 — "What if you can't beat ReAct even in-distribution? Does that kill the thesis?"

> It would weaken the thesis but not kill it. The dissertation would honestly report negative results and discuss why the framework failed — was it the SCM, the counterfactual reflection, the LLM choice, or the domain? Negative results in a well-designed experiment are still a contribution, especially with thorough ablations.
>
> That said, in-distribution performance equivalent to ReAct is the minimum acceptable outcome. If we're meaningfully worse in-distribution, something's wrong with the implementation, not with the hypothesis.

---

### K3 — "Plan B if DoWhy is too slow at inference time?"

> Three fallback options ranked by preference:
> 1. **Cache identified estimands** — the back-door adjustment formula only depends on the graph, not the data. Compute once, reuse forever. ~10x speedup expected.
> 2. **Switch to a lighter estimator** — propensity-score-based estimation is often faster than full back-door adjustment for our setting.
> 3. **Approximate scoring** — compute exact P(goal | do(a)) for the top-3 candidate actions only, score the rest by LLM heuristics.
>
> Empirical decision once we're running the agent against real tasks.

---

### K4 — "What if you can't get API access / your compute budget runs out?"

> Anthropic API costs are tracked. Per-task cost estimated at ~$0.50–$2 with current models. Full evaluation budget is ~$3000–$5000. If costs exceed budget, three options:
> 1. Smaller SWE-bench subset for OOD experiments (50 tasks per partition instead of 200).
> 2. Use a cheaper model (e.g., Claude Haiku) for baselines and only the headline model for the causal agent.
> 3. Local open-source model (Llama 3.1 70B) for development, hosted model only for final evaluation.
>
> The dissertation reports cost per-experiment for reproducibility.

---

### K5 — "Your timeline assumes everything works. What's the realistic risk-adjusted timeline?"

> The plan has slack at two points: (1) baseline evaluation runs in parallel with implementation, so a slip on Components 1–2 doesn't block baselines; (2) the writing phase has ~2 weeks of buffer.
>
> Realistic slip scenarios: Components 1–2 take an extra week (push to June 5) — manageable because ablations only need full system from June 19. End-to-end demo slips by a week (to June 26) — uncomfortably close to mid-sem but feasible. Anything beyond that and the dissertation reports partial results at mid-sem and adjusts scope.

---

## SECTION L — LLM-Specific Concerns

### L1 — "Which LLM are you using? Why?"

> Primary: Claude (Anthropic API, latest available model at evaluation time). Reason: strong agentic-task performance, large context window for tool descriptions, established prompt-caching support for cost control.
>
> Sensitivity analysis: a smaller LLM (Claude Haiku or open-source equivalent) for one full re-run to verify that the causal framework's gains generalise across model capabilities.

---

### L2 — "LLM costs are notoriously volatile. How do you ensure reproducibility?"

> Three measures: (1) **prompt caching** is enabled throughout — system prompts and SCM descriptions are cached, dramatically reducing per-call cost; (2) **all LLM responses are logged** in a structured format with timestamps, model version, and input/output token counts; (3) **the dissertation reports per-experiment cost** so future researchers can budget their replication.
>
> Reproducibility is best-effort — LLM updates may change behaviour over time. The logged responses provide a snapshot of the model's behaviour at evaluation time.

---

### L3 — "What's your prompting strategy for the counterfactual queries?"

> A structured template:
> ```
> Given the following state: {state}
> The agent took action: {action_taken}
> Outcome: {outcome}
>
> Estimate the outcome that would have resulted from each of these alternative actions:
> {list of alternatives}
>
> Return JSON: [{"action": "...", "estimated_outcome": float in [0,1]}]
> ```
>
> Notes: outputs are constrained to JSON for parseability. Multiple samples (typically 3) are drawn per query to estimate variance. The exact prompt is in an appendix of the dissertation for reproducibility.

---

### L4 — "What if the LLM refuses to answer or returns malformed output?"

> Retry logic with up to 3 attempts, then fall back to "no counterfactual evidence available" — which is treated as no-update (same as if no alternative exceeded θ). Refusals are logged and reported in aggregate.
>
> In practice, structured-output prompts get very low refusal rates with Claude.

---

## SECTION M — Reproducibility & Open Science

### M1 — "Will the code be open-sourced?"

> Yes, on GitHub at the time of dissertation submission. The repository will include: full agent implementation, SCM specification, evaluation harness, baseline implementations, the Distribution Shift Evaluation Suite, and all configuration files needed to reproduce headline results.
>
> SWE-bench tasks are already public. The novelty is the agent and evaluation harness, not the benchmark.

---

### M2 — "How do you handle stochasticity? Will results be reproducible?"

> Two sources of stochasticity: LLM sampling and bootstrap-style estimation in DoWhy.
> - **LLM**: temperature is fixed (typically 0.2 or 0), random seed where the API supports it. Where it doesn't, we average over multiple runs and report variance.
> - **DoWhy estimation**: random seeds are set explicitly. The estimator is deterministic given fixed data and seed.
>
> Headline results are averaged over 3 random seeds per condition; variance is reported in confidence intervals.

---

## SECTION N — Future Work & Extensions

### N1 — "What's the natural next step after this dissertation?"

> Three near-term extensions:
> 1. **Multi-step interventional planning** — MCTS-style search over the SCM rather than one-step greedy.
> 2. **Automatic SCM discovery** — using observed data to refine or extend the hand-authored graph (partial overlap with Topic 7, federated learning, in my parked-topics list).
> 3. **Cross-domain transfer** — testing whether an SCM trained on one domain transfers to a related one (e.g., debugging → code review).

---

### N2 — "Would this work for non-LLM agents?"

> Yes — the framework is agnostic to what produces the candidate actions. An RL agent's policy could output candidates, and the do-calculus scoring would rank them. The LLM is currently the source of counterfactual estimates, but in domains with simulators, simulation-based counterfactual generation could replace it.
>
> Generalising beyond LLM agents is future work.

---

### N3 — "Could this approach reduce the need for fine-tuning LLM agents?"

> Possibly. If causal planning gives generalisation gains comparable to fine-tuning on OOD data, that's a meaningful efficiency win — fine-tuning is expensive and brittle. But this is a hypothesis; the dissertation doesn't directly test causal-planning vs fine-tuning.
>
> Direct comparison to fine-tuned agents is a strong follow-up paper.

---

## SECTION O — Publication & Career

### O1 — "What would make this work publishable?"

> Three things, ranked:
> 1. Clear quantitative result showing OOD generalisation improvement attributable to causal structure, with ablations isolating the mechanism.
> 2. The Distribution Shift Evaluation Suite as a standalone benchmark contribution — independently useful to the agent research community.
> 3. The explainability result — human evaluations showing causal traces are preferred over CoT rationales.
>
> Target venues: NeurIPS or ICLR workshops on agentic AI or causal ML, with a possible main-conference submission depending on result strength.

---

### O2 — "This sounds like an engineering project. Where's the science?"

> The science is in the empirical test of the hypothesis. The hypothesis — that causal structure produces more robust agent generalisation than correlational patterns — is non-obvious, falsifiable, and has theoretical grounding in causal inference. The Distribution Shift Evaluation Suite is the experimental apparatus.
>
> If the hypothesis is supported, that's a contribution to our understanding of when causal methods help AI agents. If it's not supported, that's a contribution too — a clear negative result is valuable for the field.

---

### O3 — "What's your specific intellectual contribution vs the contributions of DoWhy + ReAct + SWE-bench?"

> DoWhy contributes the inference engine. ReAct contributes the agent baseline. SWE-bench contributes the evaluation tasks.
>
> My contributions: (1) the **architecture** integrating do-calculus into LLM agent action selection at inference time; (2) the **Counterfactual Reflection Module**, which is a novel online causal-belief update mechanism; (3) the **Distribution Shift Evaluation Suite** as a benchmark contribution; (4) the **empirical investigation** of whether causal grounding helps LLM agents generalise — this is the scientific question, separate from the engineering.

---

# PART 3 — RECOVERY PHRASES

When you don't know an answer:

**Honesty (preferred):**
- "That's a good question. My current thinking is X, but I'd want to verify Y before committing — happy to follow up after."
- "I haven't worked through that case in detail. The framework would handle it via [closest related mechanism]. Let me think on it more carefully."
- "You're right, that's a gap in the current plan. I'll address it before mid-sem."

**Acknowledging limits:**
- "That's outside the scope of this dissertation — it would be a natural follow-up but I won't commit to addressing it here."
- "I haven't seen the paper you're referencing. Could you give me the citation, and I'll look into how it relates."

**Buying time:**
- "Let me make sure I understand the question." *(Repeat their question back in your own words. This buys 5–10 seconds.)*
- "Good question — let me think through this carefully." *(Then take 3 seconds of actual silence to think.)*

**Avoid:**
- "I don't know" with nothing after. Always pair with what you *can* say.
- Bluffing. Examiners can tell, and being caught is much worse than admitting a gap.
- "Yes, I'll add that to the dissertation." Don't promise scope you can't deliver.
- "It's complicated." Vague hedging suggests you don't understand.

---

# PART 4 — PRE-VIVA CHECKLIST

### Week before
- [ ] Read this document end-to-end at least twice
- [ ] Practice the deck out loud, timed, at least three times
- [ ] Record one practice run and listen back — catch verbal tics
- [ ] Ask a friend or colleague to challenge you with random questions from Part 2

### Day before
- [ ] Re-read Part 2, focusing on ★★ and ★★★ questions
- [ ] One final timed deck run
- [ ] Confirm the presentation file opens correctly on the actual machine you'll use
- [ ] Prepare a backup — PDF export of the deck on a USB drive
- [ ] Print this document for offline reference if examiners allow it

### Hour before
- [ ] Open `Outline_Viva_Arpan_Ghosh.pptx` and click through every slide
- [ ] Open `Dissertation_Abstract_Outline_Arpan_Ghosh.docx` in a second window for reference if asked
- [ ] Water within reach
- [ ] Phone silenced
- [ ] Breathe

### During the viva
- [ ] Make eye contact with examiners while answering
- [ ] If asked a question with two parts, answer the harder part first
- [ ] When uncertain, *pause* rather than fill silence with filler
- [ ] If interrupted, finish your current sentence then yield
- [ ] At the end, ask if there's anything else they'd like you to clarify

Good luck. The work is solid. Trust it.
