# organisation

Personal organisation tools and experiments.

## Contents

### `daysheet.html`
Single-file daily planner for the teaching timetable at Philipp-Melanchthon-Gymnasium.
Open it in a browser — no build step, no dependencies.

- Shows the day's lessons, coaching blocks, and admin slots with times
- Automatic A/B week rotation (anchor: **Mon 24 Aug 2026 = A week**)
- Tick items off, add your own tasks, per-day progress bar
- Persists ticks per day via `window.storage` where available; otherwise a
  warning is shown and ticks reset on reload

### `graph_engine_v2_fixed.py`
Graph Engineering Framework v2.1 — a Python engine for building agentic AI
workflows as directed graphs, where nodes are LLM calls, tools, deterministic
functions, routers, verifiers, or convergent cycles, and edges are data
contracts rather than plain arrows.

- Implements 12 graph-engineering principles: node contracts, structured I/O,
  inspectable routing, verification gates, failure policies
  (RETRY / FALLBACK / SKIP / REPAIR / ESCALATE / STOP), checkpointing,
  cost-aware model tiers, cycle budgets with deduplication
- Async parallel fan-out / join; controlled cycles re-trigger nodes whose
  inputs went stale, capped by per-node budgets
- Runtime checklist validator that scores a graph against the 12 principles
- Three runnable demos: research & publishing pipeline, convergent bug
  discovery, and cost-aware request routing

Requires Python 3.11+; standard library only.

```bash
python3 graph_engine_v2_fixed.py
```

**v2.1 fixes over v2:** removed `asyncio.coroutine` (gone in Python 3.11+),
fixed ToolNode closure binding, controlled cycles no longer deadlock and
actually re-run (per-edge readiness + completion-sequence stale-input
detection), router decisions are deterministically enforced by the engine,
convergent cycles track consecutive dry rounds and can converge, the REPAIR
failure policy is handled (previously left nodes stuck), and the checklist
validator receives the graph instead of crashing on state.
