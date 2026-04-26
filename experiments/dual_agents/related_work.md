# Dual Agents Related Work

Last updated: 2026-04-25

## Scope

This note positions the `experiments/dual_agents/` "twin" protocol against prior research on:

- multi-agent debate
- iterative critique and refinement
- code-generation-specific multi-agent systems
- competitive-programming collaboration protocols

The goal is not to claim novelty for the general idea of multi-agent coding. The goal is to state clearly what is already established in the literature and what is specific to this repo's implementation.

## Closest Prior Work

### Multi-Agent Debate

**Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023)**  
https://arxiv.org/abs/2305.14325

Why it matters here:

- It is a clear precedent for the high-level pattern of multiple model instances producing candidate reasoning, challenging one another, and converging on a final answer.
- It supports the general claim that peer-style interaction can improve outcomes over a one-shot single-agent response.

Difference from this repo:

- That paper is a general debate framework for reasoning and factuality, not a benchmark harness specialized for code-generation evaluation.
- Our twin protocol is simpler and more rigid: two peers, one revision round, one consensus round, fixed six model calls per task.

### Iterative Refinement / Critique

**Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)**  
https://arxiv.org/abs/2303.11366

Why it matters here:

- It is strong prior art for draft -> feedback -> improved answer loops.
- It is relevant for coding because the paper reports coding gains from language feedback without weight updates.

Difference from this repo:

- Reflexion is primarily a self-improvement setup rather than a symmetric two-peer collaboration protocol.
- Our twin runner exchanges critiques across two agents instead of maintaining a single agent's episodic reflection memory.

## Code-Generation-Specific Multi-Agent Work

### AgentCoder

**Huang et al., "AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation" (2024)**  
https://arxiv.org/abs/2312.13010

Why it matters here:

- It is direct prior art for multi-agent code generation with iterative refinement.
- It uses specialized agents for programming, test generation, and execution feedback.

Difference from this repo:

- AgentCoder is role-specialized and testing-centric.
- Our twin protocol is intentionally symmetric rather than splitting the workflow into programmer / tester / executor roles.

### MapCoder

**Islam et al., "MapCoder: Multi-Agent Code Generation for Competitive Problem Solving" (2024)**  
https://arxiv.org/abs/2405.11403

Why it matters here:

- It is direct prior art for multi-agent code generation on competitive-programming-style tasks.
- It shows that decomposing code generation into multiple cooperating agents can improve pass@1 on difficult code benchmarks.

Difference from this repo:

- MapCoder uses a richer role decomposition such as recalling examples, planning, coding, and debugging.
- Our implementation is much smaller: draft, cross-revision, consensus, then selection.

### AdaCoder

**Zhu et al., "AdaCoder: An Adaptive Planning and Multi-Agent Framework for Function-Level Code Generation" (2025)**  
https://arxiv.org/abs/2504.04220

Why it matters here:

- It is more recent code-generation-specific prior art that explicitly studies the generalizability of multi-agent coding frameworks across models.
- It reinforces that code-generation multi-agent systems are already an active line of work, not an empty research space.

Difference from this repo:

- AdaCoder centers on adaptive planning, debugging, and a stronger script-based execution loop.
- Our twin system is deliberately a lightweight peer-collaboration baseline rather than an adaptive planner-debugger pipeline.

## Most Similar High-Level Framing

### Symmetric / Cross-Verification Code Collaboration

**Song and Azman, "Leveraging Symmetry in Multi-Agent Code Generation: A Cross-Verification Collaboration Protocol for Competitive Programming" (2025)**  
https://www.mdpi.com/2073-8994/17/10/1660

Why it matters here:

- This is the closest prior work in spirit to the repo's "twin" framing.
- It explicitly treats symmetric peer collaboration and cross-verification as central design ideas for code generation on competitive programming tasks.

Difference from this repo:

- Their framework is materially more elaborate, with multiple modules for review, symmetry handling, adversarial testing, and voting.
- Our twin protocol is smaller and easier to inspect: two peers, fixed stage order, no explicit symmetry detector, no adversarial test generator, no asynchronous voting module.

## Positioning For This Repo

The safest paper or report framing is:

1. The repo's dual-agent system is **not** the first multi-agent coding method.
2. The repo's contribution is a **small, benchmark-compatible twin baseline** built in the same evaluation stack as the single-agent prompt-optimization experiments.
3. The distinctive choices in this repo are mostly harness-level:
   - symmetric peers instead of role-specialized agents
   - fixed six-call protocol
   - shared benchmark loaders and hidden-test evaluator
   - grouped held-out splits aligned with the existing benchmark stack
   - optional `local_eval` reranking over all six candidates on the task's hidden tests

## Novelty Boundary

Reasonable claims:

- "We implement a lightweight symmetric twin-agent baseline for the existing local coding benchmark stack."
- "We evaluate a peer-collaboration protocol on the same held-out grouped splits used elsewhere in the repo."
- "We compare a small, inspectable collaboration harness against single-agent baselines in the same evaluation environment."

Claims to avoid:

- "This is the first multi-agent code-generation framework."
- "This is the first debate-style coding protocol."
- "This is the first symmetric peer-collaboration method for code generation."

The last claim is especially risky because the 2025 CVCP paper is already quite close in spirit, even though the concrete protocol differs.

## Methodological Note

If results from this repo are reported publicly, the selection strategy must be described carefully:

- `consensus_first` is the cleaner measure of collaboration alone.
- `local_eval` is better viewed as an oracle-style local reranking setup because it uses hidden tests to choose among multiple same-task candidates.

That distinction matters more than most prompt details. It changes what the results mean.

## Bottom Line

The right framing is not "new idea from scratch." The right framing is:

"a compact, symmetric, benchmark-aligned multi-agent coding baseline that is easy to run, inspect, and compare against the repo's existing single-agent experiments."
