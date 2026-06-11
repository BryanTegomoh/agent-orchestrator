# agent-orchestrator

[![CI](https://github.com/BryanTegomoh/agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/BryanTegomoh/agent-orchestrator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-blue.svg)](https://mypy.readthedocs.io/)

A small, typed library for orchestrating multi-agent LLM work: a dependency-aware task graph over task-routed sub-agents, with two-layer memory, prompt-injection screening, and fail-loud execution.

It is a reference implementation, not a turnkey platform. It ships the orchestration, routing, memory, and screening building blocks and stays out of the way; you wire them to your own LLM clients. Each block is a small, single-purpose module: strict-typed, tested, no required dependencies.

---

## The core idea

Most "agent frameworks" are a single model in a `while` loop. That shape fails in predictable ways once the work is real. This library is the set of patterns that address those failures, each kept small enough to read in one sitting.

| Failure mode of the naive approach | What this does instead |
|------------------------------------|------------------------|
| One model for everything, so you overpay on easy tasks and underperform on hard ones | Route by task type to the right model class, with a deterministic O(1) classifier (no model call) |
| Multi-step work runs out of order, or a step starts before its inputs exist | A task graph where dependencies are declared at creation and the scheduler gates on them |
| Failure is swallowed; a partial run reads as success | Fail-loud execution: unknown work, unregistered agents, and failed tasks raise rather than returning a half-result |
| An agent claims work it did not do | A self-report check rejects completion reports that reference tasks that do not exist |
| Flat memory that forgets across sessions | Two layers: keyword recall for known categories, vector recall for semantic queries across sessions |
| Untrusted input is treated as trusted | Screen external content before it reaches the model (as triage, not as a security boundary) |
| Autonomy is all-or-nothing | Permission grants and owner-decision gates: work stops at its authorized boundary with a decision-ready brief, while independent lanes continue |

---

## The looping model

The shift this library is built for: instead of prompting an agent one step at a time, you define a loop and let it run to a goal. Two shapes, one primitive for each.

**Single-agent loop (`run_goal`).** One worker produces, a judge scores the result against the goal, and the loop repeats with the judge's feedback until the goal is met or the budget is spent. The rule that decides whether this works: the judge must be independent of the worker. A worker reviewing its own output tends to approve it; an independent judge, or a deterministic check such as a test suite or a type checker, keeps the loop honest.

```
 goal + acceptance criteria
             │
             ▼
      worker produces ◄───────────────┐
             │                        │
             ▼               feedback │
       judge checks ── not done ──────┘
             │
     done ───┴─── budget spent
       │              │
       ▼              ▼
   "done"        "exhausted"
                 (never a false done)
```

**Fleet loop (`run_graph`).** An orchestrator owns the goal, declares the steps as a task graph, and delegates each step to a specialist. Independent steps run in parallel; dependent steps wait for their inputs; a final judge gate decides whether the result ships. A specialist can itself spawn a sub-graph, and the self-report gate checks any claim it makes about that work.

```
            orchestrator owns the goal
                       │
                       ▼
              declares a TaskGraph
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   specialist     specialist     specialist    independent: run in parallel
        └──────────────┼──────────────┘
                       ▼
                synthesis task                 gated: waits for all its parents
                       │
                       ▼
                  judge gate                   optional: run_goal on the result
```

**Closed, not open.** An open loop explores an unbounded space: it can find things you did not specify, it spends heavily to do so, and against a loose standard it produces low-quality output at scale. A closed loop is bounded: a human designs the path, every step has a check, and the budget is predictable. This library is deliberately closed. The path is the `TaskGraph` you declare, the checks are the judge and the fail-loud invariants, and the loop ends in an explicit verdict either way.

---

## Where this fits

For durable, distributed workflow execution, use a workflow engine such as Temporal; for a batteries-included agent framework, LangGraph or AutoGen. This library is for when you want the orchestration patterns themselves, in a few readable modules you can own and extend.

---

## The pieces

Nine small modules, strict-typed, no required dependencies. The orchestrator's single-task pipeline is screen → recall → route → delegate → persist; `run_graph` applies it per task across a gated graph.

| Module | Role |
|--------|------|
| [`taskgraph.py`](src/agent_orchestrator/taskgraph.py) | Dependency-gated DAG: parents declared at creation, ready waves, optional thread-pool execution, fail-loud invariants, self-report gate |
| [`authority.py`](src/agent_orchestrator/authority.py) | Bounded autonomy: permission grants, owner parks, decision-ready briefs |
| [`goal.py`](src/agent_orchestrator/goal.py) | Judge loop: worker, independent judge, feedback, explicit `done` or `exhausted` |
| [`orchestrator.py`](src/agent_orchestrator/orchestrator.py) | Ties it together: screening, memory recall, routing, delegation, graph execution |
| [`router.py`](src/agent_orchestrator/router.py) | Deterministic keyword classifier; task type to model, no model call |
| [`memory.py`](src/agent_orchestrator/memory.py) | File memory: pointer index, topic files, rolling 7-day active context |
| [`semantic_memory.py`](src/agent_orchestrator/semantic_memory.py) | Vector memory on LanceDB: extract, embed, idempotent upsert, cosine top-k |
| [`ledger.py`](src/agent_orchestrator/ledger.py) | Append-only audit trail of task lifecycle events |
| [`security.py`](src/agent_orchestrator/security.py) | Heuristic injection triage for untrusted content |

The walkthroughs live in [docs/guide.md](docs/guide.md): wiring a provider, task graphs, goal loops, bounded autonomy, routing, memory, and screening. Operational notes are in [docs/operations.md](docs/operations.md).

---

## Installation

Install from source (this is a reference implementation, not a published package):

```bash
git clone https://github.com/BryanTegomoh/agent-orchestrator
cd agent-orchestrator
pip install -e .
```

Add optional dependencies based on your stack:

```bash
# LLM providers (pick one or more)
pip install -e ".[litellm]"     # route to any provider
pip install -e ".[anthropic]"   # Claude only
pip install -e ".[openai]"      # OpenAI / compatible APIs

# Semantic memory (LanceDB vector store)
pip install -e ".[lancedb]"

# Everything
pip install -e ".[all]"
```

---

## Design decisions

The non-obvious choices, and the failure each one prevents.

**Dependencies are declared once, at creation.** A child is created with its parents (`add(..., parents=[...])`), never created first and linked later. Late linking opens a race where the scheduler claims a child before its inputs exist; declaring up front removes it. The graph is acyclic by construction, since a task can depend only on tasks that already exist.

**Unrunnable work fails loud.** An unknown parent raises at creation; an unregistered assignee raises at run time; a failed task leaves its descendants unreachable and `run` raises. A queue that silently drops impossible work is worse than one that stops and tells you why.

**Self-reported success is not evidence.** A task that reports the children it spawned has those ids checked against the graph; phantom ids are rejected. The same instinct that distrusts a model's claim about the world distrusts its claim about its own actions.

**A loop that cannot reach its goal says so.** `run_goal` returns `"exhausted"`, never a false `"done"`, and acceptance is decided by an explicit judge rather than the worker grading its own homework.

**The boundary of autonomy is explicit.** Tasks declare the grants they need, and grants do not imply one another: push is not merge, merge is not release. Work past its authority parks with a decision-ready brief (what, proof, recommendation, exact choices) instead of failing or proceeding, and lifecycle events land in an append-only ledger.

**No bundled benchmarks, no pinned model versions.** Model identifiers in the defaults are illustrative and overridable, and the routing logic depends on none of them. Nothing in the repo goes stale the day a new model ships, and no unverifiable performance claim is presented as fact.

---

## Limitations

Known edges, stated rather than discovered:

- **The router is a keyword heuristic.** Fast, free, and auditable, but not a semantic classifier; unusual phrasing can misroute. `delegate()` exists for when you already know where work should go.
- **Graph state is in-memory** for the duration of `run`. There is no persistence, resume, or distributed execution; a crashed process re-runs the graph. Re-running is safe on the memory side because ingestion is idempotent, and the ledger records what ran, but it does not restore state. For durable execution, use a workflow engine (see Where this fits).
- **The file-memory layer assumes one process** per memory directory. The orchestrator serializes its own writes; nothing arbitrates between processes.
- **Screening is regex triage.** It catches lazy injection attempts, not determined ones.
- **Transport is your client's job.** Beyond one fallback-model retry in `run`, there are no retries, backoff, or rate limits; bring them in your LLM caller.

---

## Running the examples

Every example runs with no API keys (stub agents and stub embeddings):

```bash
pip install -e ".[dev]"

python examples/full_pipeline.py     # the closed loop end to end: parallel fleet + judge gate
python examples/task_graph.py        # dependency-gated orchestration + fail-loud
python examples/goal_loop.py         # judge loop; converges, then exhausts
python examples/basic_routing.py     # task routing decisions
python examples/security_screening.py
python examples/memory_management.py # file memory and compaction
python examples/semantic_search.py   # vector recall (stub embeddings)
```

---

## Tests

```bash
pytest                              # full suite
pytest --cov=agent_orchestrator     # with coverage
```

The suite covers routing, screening, the task graph (gating, parallel waves, fail-loud, self-report), the goal loop, orchestrator execution, file memory, and semantic memory (cosine scores, idempotent re-ingest). CI runs `ruff`, `mypy --strict`, and `pytest` on Python 3.11 and 3.12.

---

## License

MIT
