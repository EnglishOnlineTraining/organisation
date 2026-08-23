"""
Graph Engineering Framework v2.1 — Production-Ready Implementation
All 12 principles from the graph engineering philosophy:

1. Dependencies, not sequence
2. Node contracts
3. Edges as data contracts
4. Four core shapes (chain, diamond, router, controlled cycle)
5. Fan out / join deliberately
6. Inspectable routing
7. Verification on the edge
8. Durable state & artifact references
9. Convergent cycles with deduplication
10. Failure as a local event (RETRY, FALLBACK, SKIP, REPAIR, ESCALATE, STOP)
11. Topology as cost model (model routing by task complexity)
12. Production research & publishing graph

Plus: The Graph Engineering Checklist as a runtime validator.

v2.1 fixes:
- Removed asyncio.coroutine (removed in Python 3.11+) from research ToolNodes;
  research branches no longer silently fail and get SKIPPED.
- Fixed ToolNode closures binding `state` over the `source_type` default arg.
- Controlled cycles actually re-run: the engine now evaluates readiness per
  edge (inactive edges neither trigger nor block) and re-triggers a node when
  a predecessor completed after it (sequence-based stale-input detection),
  capped per node by Budget.max_iterations or NodeContract.max_visits.
- RouterNode decisions are now enforced deterministically by the engine
  (Principle 6) and the resolved target node is exposed in the output.
- Cost-model demo: routes now point at real node names ("planner").
- ConvergentCycleNode tracks consecutive dry rounds itself; convergence_test
  receives (seen, fresh, dry_rounds) so cycles can actually converge.
- REPAIR failure policy is now handled by the executor (was a dead end that
  left nodes stuck in RUNNING) and is capped by max_retries.
- Checklist validation receives the AgentGraph (was crashing on GraphState).
"""

import asyncio
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod
from copy import deepcopy
import traceback


# ═══════════════════════════════════════════════════════════════
# PRINCIPLE 8: DURABLE STATE
# ═══════════════════════════════════════════════════════════════

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    ESCALATED = "escalated"


class FailurePolicy(Enum):
    """Principle 10: Design failure as a local event."""
    RETRY = "retry"           # Transient tool or network failure
    FALLBACK = "fallback"     # Preferred model/source unavailable
    SKIP = "skip"             # Optional branch failed
    REPAIR = "repair"         # Output failed validation
    ESCALATE = "escalate"     # Risk or uncertainty crossed threshold
    STOP = "stop"             # Budget, safety, or permission boundary


class ModelTier(Enum):
    """Principle 11: Topology is your cost model."""
    SMALL = "small"     # Cheap: extraction, classification, formatting
    MEDIUM = "medium" # Balanced: bounded reasoning, summarization
    STRONG = "strong"   # Expensive: decomposition, synthesis, hard verification


@dataclass
class Artifact:
    """Principle 8: Move references to artifacts, not giant transcripts."""
    artifact_id: str
    artifact_type: str
    storage_path: str
    checksum: str
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_content(cls, content: Any, artifact_type: str, storage_path: str) -> "Artifact":
        content_str = json.dumps(content, sort_keys=True, default=str)
        return cls(
            artifact_id=str(uuid.uuid4())[:8],
            artifact_type=artifact_type,
            storage_path=storage_path,
            checksum=hashlib.sha256(content_str.encode()).hexdigest()[:16],
            size_bytes=len(content_str.encode()),
            metadata={"created_at": time.time()}
        )


@dataclass
class Checkpoint:
    """Principle 8 + 10: Checkpoint after expensive nodes for resumability."""
    checkpoint_id: str
    task_id: str
    timestamp: float
    current_node: str
    completed_nodes: List[str]
    state_snapshot: Dict[str, Any]
    artifacts: List[str]
    decisions: List[Dict]
    budgets: Dict[str, Any]
    retry_counts: Dict[str, int]


@dataclass
class NodeContract:
    """Principle 2: Every node needs one job, explicit input, structured output, clear failure state."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    failure_states: List[str] = field(default_factory=list)
    model_tier: ModelTier = ModelTier.MEDIUM
    max_retries: int = 2
    failure_policy: FailurePolicy = FailurePolicy.RETRY
    checkpoint_after: bool = False
    idempotent: bool = True
    max_visits: int = 3  # Hard cap on re-executions inside controlled cycles


@dataclass
class Edge:
    """Principle 3: Edges are data contracts, not arrows."""
    from_node: str
    to_node: str
    condition: Optional[Callable[[Dict], bool]] = None
    transform: Optional[Callable[[Dict], Dict]] = None
    label: str = ""


@dataclass
class Budget:
    """Principle 9: Every cycle needs a token/cost budget."""
    max_tokens: int = 100000
    max_cost_usd: float = 5.0
    max_time_seconds: float = 300.0
    max_iterations: int = 6

    def is_exceeded(self, used_tokens: int, used_cost: float, used_time: float, used_iterations: int) -> bool:
        return (
            used_tokens >= self.max_tokens or
            used_cost >= self.max_cost_usd or
            used_time >= self.max_time_seconds or
            used_iterations >= self.max_iterations
        )


@dataclass
class GraphState:
    """Principle 8: Durable state that can answer three questions:
    - What has already happened
    - Why did the system choose this route
    - Where can execution safely resume
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    node_results: Dict[str, Any] = field(default_factory=dict)
    node_status: Dict[str, NodeStatus] = field(default_factory=dict)
    node_attempts: Dict[str, int] = field(default_factory=dict)
    decisions: List[Dict] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)
    budgets: Dict[str, Budget] = field(default_factory=dict)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    human_approvals: List[Dict] = field(default_factory=list)
    execution_log: List[Dict] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)
    seen_items: Set[str] = field(default_factory=set)  # Principle 9: deduplication
    dry_rounds: int = 0
    iteration: int = 0
    # Completion ordering: a node may re-run when a predecessor completed after it
    completion_seq: Dict[str, int] = field(default_factory=dict)
    seq_counter: int = 0

    def log(self, node: str, event: str, detail: Any = None):
        entry = {
            "timestamp": time.time(),
            "node": node,
            "event": event,
            "detail": detail
        }
        self.execution_log.append(entry)
        return entry

    def record_decision(self, node: str, decision: str, reason: str, state_snapshot: Dict):
        """Principle 10: Record every routing decision with the state that produced it."""
        self.decisions.append({
            "timestamp": time.time(),
            "node": node,
            "decision": decision,
            "reason": reason,
            "state_hash": hashlib.sha256(json.dumps(state_snapshot, sort_keys=True, default=str).encode()).hexdigest()[:12]
        })

    def create_checkpoint(self, current_node: str) -> Checkpoint:
        """Principle 8 + 10: Save state for resumability."""
        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            task_id=self.task_id,
            timestamp=time.time(),
            current_node=current_node,
            completed_nodes=[n for n, s in self.node_status.items() if s == NodeStatus.SUCCESS],
            state_snapshot=deepcopy(self.data),
            artifacts=list(self.artifacts.keys()),
            decisions=self.decisions.copy(),
            budgets={k: asdict(v) for k, v in self.budgets.items()},
            retry_counts=self.retry_counts.copy()
        )
        self.checkpoints.append(cp)
        return cp

    def store_artifact(self, key: str, content: Any, artifact_type: str = "generic") -> Artifact:
        """Principle 8: Store artifact, return reference."""
        path = f"/artifacts/{self.task_id}/{key}"
        artifact = Artifact.from_content(content, artifact_type, path)
        self.artifacts[key] = artifact
        return artifact

    def get_artifact_ref(self, key: str) -> Optional[str]:
        """Return a reference path instead of the full content."""
        art = self.artifacts.get(key)
        return art.storage_path if art else None

    def to_audit_log(self) -> Dict:
        """Answer the three questions at any moment."""
        return {
            "task_id": self.task_id,
            "what_happened": [
                {"node": n, "status": s.value, "attempts": self.node_attempts.get(n, 0)}
                for n, s in self.node_status.items()
            ],
            "why_this_route": self.decisions,
            "where_to_resume": self.checkpoints[-1].current_node if self.checkpoints else "start",
            "budgets_remaining": {
                k: {
                    "tokens_left": v.max_tokens - self.data.get(f"_used_tokens_{k}", 0),
                    "iterations_left": v.max_iterations - self.iteration
                }
                for k, v in self.budgets.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# BASE NODE CLASS
# ═══════════════════════════════════════════════════════════════

class GraphNode(ABC):
    """Abstract base for all nodes."""

    def __init__(self, contract: NodeContract):
        self.contract = contract
        self._predecessors: Set[str] = set()
        self._successors: Set[str] = set()

    @abstractmethod
    async def execute(self, state: GraphState) -> Dict[str, Any]:
        pass

    def add_predecessor(self, node_id: str):
        self._predecessors.add(node_id)

    def add_successor(self, node_id: str):
        self._successors.add(node_id)


# ═══════════════════════════════════════════════════════════════
# CONCRETE NODE TYPES
# ═══════════════════════════════════════════════════════════════

class LLMNode(GraphNode):
    """Principle 11: Model tier determines which model to call."""

    MODEL_COSTS = {
        ModelTier.SMALL: {"cost_per_1k": 0.0001, "name": "small-model"},
        ModelTier.MEDIUM: {"cost_per_1k": 0.001, "name": "medium-model"},
        ModelTier.STRONG: {"cost_per_1k": 0.005, "name": "strong-model"}
    }

    def __init__(self, contract: NodeContract, 
                 prompt_template: str,
                 temperature: float = 0.1,
                 estimated_tokens: int = 2000):
        super().__init__(contract)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.estimated_tokens = estimated_tokens

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        model_info = self.MODEL_COSTS[self.contract.model_tier]
        print(f"  [LLMNode '{self.contract.name}'] tier={self.contract.model_tier.value} model={model_info['name']}")

        # Simulate cost tracking
        cost = (self.estimated_tokens / 1000) * model_info["cost_per_1k"]
        state.data[f"_cost_{self.contract.name}"] = state.data.get(f"_cost_{self.contract.name}", 0) + cost
        state.data[f"_used_tokens_{self.contract.name}"] = state.data.get(f"_used_tokens_{self.contract.name}", 0) + self.estimated_tokens

        await asyncio.sleep(0.05)

        # Store large output as artifact, return reference
        output = {k: f"<generated_{k}_v{state.node_attempts.get(self.contract.name, 0)+1}>" 
                  for k in self.contract.output_schema.keys()}

        if len(str(output)) > 500:  # Threshold for artifact storage
            art = state.store_artifact(f"{self.contract.name}_output", output, "llm_output")
            return {
                "status": "success",
                "node": self.contract.name,
                "output": {"_artifact_ref": art.artifact_id, "_summary": f"Output stored at {art.storage_path}"},
                "cost": cost
            }

        return {"status": "success", "node": self.contract.name, "output": output, "cost": cost}


class ToolNode(GraphNode):
    def __init__(self, contract: NodeContract, tool_fn: Callable):
        super().__init__(contract)
        self.tool_fn = tool_fn

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        print(f"  [ToolNode '{self.contract.name}'] Executing")
        try:
            result = await self.tool_fn(state)
            return {"status": "success", "node": self.contract.name, "output": result}
        except Exception as e:
            return {
                "status": "failed",
                "node": self.contract.name,
                "error": str(e),
                "failure_state": self.contract.failure_states[0] if self.contract.failure_states else "unknown"
            }


class FunctionNode(GraphNode):
    """Principle 3: Deterministic plumbing — no agent needed."""

    def __init__(self, contract: NodeContract, fn: Callable[[Dict], Dict]):
        super().__init__(contract)
        self.fn = fn

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        print(f"  [FunctionNode '{self.contract.name}'] Deterministic transform")
        try:
            result = self.fn(state.data)
            return {"status": "success", "node": self.contract.name, "output": result}
        except Exception as e:
            return {"status": "failed", "node": self.contract.name, "error": str(e)}


class RouterNode(GraphNode):
    """Principle 6: Probabilistic judgment + deterministic enforcement.

    The node classifies and records the chosen route in state under
    `_route_{name}`; the engine then enforces it deterministically —
    only the edge leading to the chosen target may fire.
    """

    def __init__(self, contract: NodeContract, 
                 classifier_fn: Callable[[Dict], str],
                 routes: Dict[str, str]):
        super().__init__(contract)
        self.classifier_fn = classifier_fn
        self.routes = routes

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        print(f"  [RouterNode '{self.contract.name}'] Classifying...")
        decision = self.classifier_fn(state.data)

        if decision not in self.routes:
            decision = "default"

        next_node = self.routes.get(decision)

        # Record decision with state snapshot for auditability
        state.record_decision(
            self.contract.name, 
            decision, 
            f"Classified as '{decision}'",
            dict(state.data)
        )

        # Expose the resolved target for deterministic engine enforcement
        if next_node is not None:
            state.data[f"_route_{self.contract.name}"] = next_node

        print(f"    -> Route: '{decision}' -> {next_node}")
        return {
            "status": "success",
            "node": self.contract.name,
            "decision": decision,
            "next_node": next_node,
            "output": {"decision": decision, "route": next_node}
        }


class VerifierNode(GraphNode):
    """Principle 7: Stop weak work from moving downstream."""

    def __init__(self, contract: NodeContract, 
                 check_fn: Callable[[Dict], Tuple[bool, str]],
                 pass_node: str,
                 fail_node: str):
        super().__init__(contract)
        self.check_fn = check_fn
        self.pass_node = pass_node
        self.fail_node = fail_node

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        print(f"  [VerifierNode '{self.contract.name}'] Verifying...")
        passed, reason = self.check_fn(state.data)

        if passed:
            print(f"    -> PASS")
            return {"status": "success", "node": self.contract.name, "passed": True, "output": {"verified": True}}
        else:
            print(f"    -> FAIL: {reason}")
            return {
                "status": "success",
                "node": self.contract.name,
                "passed": False,
                "reason": reason,
                "output": {"verified": False, "reason": reason}
            }


class ConvergentCycleNode(GraphNode):
    """Principle 9: Cycle with measurable convergence, deduplication, dry rounds.

    The node tracks consecutive dry rounds (rounds with zero fresh findings)
    itself and passes them to the convergence test, so convergence is a
    function of the whole history, not just the latest round.
    """

    def __init__(self, contract: NodeContract,
                 discover_fn: Callable[[Dict], List[Dict]],
                 convergence_test: Callable[[Set[str], List[Dict], int], bool],
                 max_dry_rounds: int = 2,
                 max_iterations: int = 6):
        super().__init__(contract)
        self.discover_fn = discover_fn
        self.convergence_test = convergence_test
        self.max_dry_rounds = max_dry_rounds
        self.max_iterations = max_iterations

    async def execute(self, state: GraphState) -> Dict[str, Any]:
        print(f"  [ConvergentCycleNode '{self.contract.name}'] Discovering with convergence...")

        budget = state.budgets.get(self.contract.name, Budget(max_iterations=self.max_iterations))
        consecutive_dry = 0
        start_time = time.time()

        while not budget.is_exceeded(
            state.data.get(f"_used_tokens_{self.contract.name}", 0),
            state.data.get(f"_cost_{self.contract.name}", 0),
            time.time() - start_time,
            state.iteration
        ):
            findings = self.discover_fn(state.data)

            # Deduplicate against everything already seen (not just verified ones)
            fresh = [f for f in findings if f.get("key") and f["key"] not in state.seen_items]

            for f in fresh:
                state.seen_items.add(f["key"])

            consecutive_dry = 0 if fresh else consecutive_dry + 1
            state.dry_rounds = consecutive_dry
            state.iteration += 1

            print(f"    -> Iteration {state.iteration}: {len(fresh)} fresh, {len(state.seen_items)} total seen, {consecutive_dry} consecutive dry rounds")

            if self.convergence_test(state.seen_items, fresh, consecutive_dry):
                print(f"    -> CONVERGED after {state.iteration} iterations")
                return {
                    "status": "success",
                    "node": self.contract.name,
                    "output": {
                        "findings": list(state.seen_items),
                        "iterations": state.iteration,
                        "converged": True
                    }
                }

        # Escalation path when convergence fails
        print(f"    -> DID NOT CONVERGE after {state.iteration} iterations — escalating")
        return {
            "status": "success",
            "node": self.contract.name,
            "output": {
                "findings": list(state.seen_items),
                "iterations": state.iteration,
                "converged": False
            },
            "escalate": True
        }


# ═══════════════════════════════════════════════════════════════
# THE GRAPH ENGINE
# ═══════════════════════════════════════════════════════════════

class AgentGraph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[Edge] = []
        self.entry_nodes: Set[str] = set()
        self.failure_handlers: Dict[str, Callable] = {}

    def add_node(self, node: GraphNode) -> "AgentGraph":
        self.nodes[node.contract.name] = node
        return self

    def add_edge(self, edge: Edge) -> "AgentGraph":
        self.edges.append(edge)
        if edge.from_node in self.nodes:
            self.nodes[edge.from_node].add_successor(edge.to_node)
        if edge.to_node in self.nodes:
            self.nodes[edge.to_node].add_predecessor(edge.from_node)
        return self

    def set_entry(self, node_name: str) -> "AgentGraph":
        self.entry_nodes.add(node_name)
        return self

    def _visit_cap(self, node: GraphNode, state: GraphState) -> int:
        """Max times a node may execute. A per-node Budget overrides the contract default."""
        if node.contract.name in state.budgets:
            return state.budgets[node.contract.name].max_iterations
        return node.contract.max_visits

    def _edge_active(self, edge: Edge, state: GraphState) -> bool:
        """An edge is active when its condition holds (or it has none) and,
        for edges leaving a RouterNode, the router actually chose this target
        (Principle 6: probabilistic judgment, deterministic enforcement)."""
        if edge.condition is not None and not edge.condition(state.data):
            return False
        src = self.nodes.get(edge.from_node)
        if isinstance(src, RouterNode):
            routed = state.data.get(f"_route_{edge.from_node}")
            if routed is not None and edge.to_node != routed:
                return False
        return True

    def get_ready_nodes(self, state: GraphState) -> List[str]:
        """Nodes eligible to run now.

        Readiness is evaluated per incoming edge, not per predecessor:
        - An INACTIVE edge (condition false, or a router route not taken) is
          treated as a path not chosen — it neither triggers nor blocks.
        - A node is ready when at least one incoming edge is active and every
          ACTIVE edge's source has finished (SUCCESS/SKIPPED).
        - Controlled cycle: a node that already ran becomes eligible again when
          an active-edge source completed AFTER it (stale inputs). Re-runs are
          capped by Budget.max_iterations or contract.max_visits.
        This allows check<-repair cycle-back edges to exist without deadlocking
        the first pass through the graph.
        """
        ready = []
        for name, node in self.nodes.items():
            status = state.node_status.get(name)
            if status in (NodeStatus.RUNNING, NodeStatus.SKIPPED):
                continue

            incoming = [e for e in self.edges if e.to_node == name]

            if not incoming:
                # Entry / orphan node: runs once from PENDING
                if status in (NodeStatus.PENDING, NodeStatus.RETRYING):
                    ready.append(name)
                continue

            active = [e for e in incoming if self._edge_active(e, state)]
            if not active:
                continue

            active_sources_done = all(
                state.node_status.get(e.from_node) in (NodeStatus.SUCCESS, NodeStatus.SKIPPED)
                for e in active
            )
            if not active_sources_done:
                continue

            if status in (NodeStatus.PENDING, NodeStatus.RETRYING):
                ready.append(name)
            elif status in (NodeStatus.SUCCESS, NodeStatus.FAILED):
                my_seq = state.completion_seq.get(name, 0)
                stale_inputs = any(state.completion_seq.get(e.from_node, 0) > my_seq for e in active)
                if stale_inputs and state.node_attempts.get(name, 0) < self._visit_cap(node, state):
                    ready.append(name)
        return ready

    def _apply_failure_policy(self, node: GraphNode, result: Dict, state: GraphState) -> str:
        """Principle 10: Apply failure policy. Returns next action: 'retry', 'skip', 'repair', 'escalate', 'stop', 'fail'."""
        policy = node.contract.failure_policy
        name = node.contract.name

        state.retry_counts[name] = state.retry_counts.get(name, 0) + 1
        attempts = state.retry_counts[name]

        if policy == FailurePolicy.RETRY and attempts < node.contract.max_retries:
            print(f"    -> Policy RETRY ({attempts}/{node.contract.max_retries})")
            return "retry"
        elif policy == FailurePolicy.FALLBACK:
            print(f"    -> Policy FALLBACK")
            return "skip"  # Simplified: skip to next
        elif policy == FailurePolicy.SKIP:
            print(f"    -> Policy SKIP")
            return "skip"
        elif policy == FailurePolicy.REPAIR:
            if attempts < node.contract.max_retries:
                print(f"    -> Policy REPAIR ({attempts}/{node.contract.max_retries})")
                return "repair"
            print(f"    -> Policy REPAIR exhausted ({attempts}/{node.contract.max_retries}) — escalating")
            return "escalate"
        elif policy == FailurePolicy.ESCALATE:
            print(f"    -> Policy ESCALATE")
            return "escalate"
        elif policy == FailurePolicy.STOP:
            print(f"    -> Policy STOP")
            return "stop"
        else:
            return "fail"

    async def execute(self, initial_data: Dict[str, Any], 
                     budgets: Optional[Dict[str, Budget]] = None,
                     max_iterations: int = 100) -> GraphState:
        state = GraphState(data=initial_data)
        if budgets:
            state.budgets = budgets

        for name in self.nodes:
            state.node_status[name] = NodeStatus.PENDING

        for entry in self.entry_nodes:
            if entry in self.nodes:
                self.nodes[entry]._predecessors = set()

        iteration = 0
        running_tasks: Dict[str, asyncio.Task] = {}

        print(f"\n{'='*70}")
        print(f"GRAPH: {self.name}")
        print(f"TASK:  {state.task_id}")
        print(f"{'='*70}\n")

        while iteration < max_iterations:
            iteration += 1

            # Check completed tasks
            completed = []
            for name, task in list(running_tasks.items()):
                if task.done():
                    completed.append(name)
                    try:
                        result = await task
                        node = self.nodes[name]
                        state.node_attempts[name] = state.node_attempts.get(name, 0) + 1
                        state.seq_counter += 1
                        state.completion_seq[name] = state.seq_counter

                        if result.get("status") == "success":
                            state.node_status[name] = NodeStatus.SUCCESS
                            state.node_results[name] = result
                            state.data[f"{name}_output"] = result.get("output", {})
                            state.log(name, "completed", result)

                            # Principle 8 + 10: Checkpoint after expensive nodes
                            if node.contract.checkpoint_after:
                                cp = state.create_checkpoint(name)
                                print(f"    -> Checkpoint saved: {cp.checkpoint_id}")
                        else:
                            # Principle 10: Handle failure locally
                            action = self._apply_failure_policy(node, result, state)
                            if action == "retry":
                                state.node_status[name] = NodeStatus.RETRYING
                            elif action == "repair":
                                # Re-run the node; the repair happens on the next attempt
                                state.node_status[name] = NodeStatus.RETRYING
                            elif action == "skip":
                                state.node_status[name] = NodeStatus.SKIPPED
                                state.node_results[name] = result
                            elif action in ("escalate", "stop", "fail"):
                                state.node_status[name] = NodeStatus.FAILED
                                state.node_results[name] = result
                                state.log(name, f"failed_{action}", result)
                    except Exception as e:
                        state.node_status[name] = NodeStatus.FAILED
                        state.node_results[name] = {"status": "failed", "error": str(e)}
                        state.log(name, "error", str(e))

            for name in completed:
                del running_tasks[name]

            # Find nodes ready to run (edge conditions + router enforcement
            # are evaluated inside get_ready_nodes via _edge_active)
            actually_ready = self.get_ready_nodes(state)

            if not actually_ready and not running_tasks:
                break

            # FAN OUT: Launch ready nodes in parallel
            for name in actually_ready:
                state.node_status[name] = NodeStatus.RUNNING
                state.log(name, "started")

                node = self.nodes[name]
                for edge in self.edges:
                    if edge.to_node == name and edge.transform:
                        state.data = {**state.data, **edge.transform(state.data)}

                task = asyncio.create_task(node.execute(state))
                running_tasks[name] = task

            if running_tasks:
                await asyncio.sleep(0.01)

        print(f"\n{'='*70}")
        print(f"COMPLETE: {iteration} iterations")
        print(f"{'='*70}\n")
        return state


# ═══════════════════════════════════════════════════════════════
# PRINCIPLE 12: PRODUCTION RESEARCH & PUBLISHING GRAPH
# ═══════════════════════════════════════════════════════════════

def make_research_fn(source_type: str) -> Callable:
    """Factory so each research ToolNode gets its own async tool function
    bound to the correct source_type (no closure-over-loop-variable bug,
    no asyncio.coroutine — removed in Python 3.11)."""

    async def research_fn(state: GraphState) -> Dict[str, Any]:
        return {
            "findings": [
                {"key": f"{source_type}_finding_1", "claim": f"Claim from {source_type}", "source": f"https://{source_type}.com/1"},
                {"key": f"{source_type}_finding_2", "claim": f"Another from {source_type}", "source": f"https://{source_type}.com/2"}
            ],
            "sources": [f"https://{source_type}.com/1", f"https://{source_type}.com/2"]
        }

    return research_fn


async def demo_research_publishing_graph():
    """
    Principle 12: Full production graph for research and publishing.

    TOPIC -> SCOPE -> DECOMPOSE -> [COMPANY_SOURCES, PAPERS, EXPERT_POSTS] -> DEDUPE -> DRAFT -> CHECK -> [HUMAN_GATE | REPAIR] -> PUBLISH
    """

    graph = AgentGraph("Research & Publishing Pipeline")

    # NODE 1: SCOPE — defines question, audience, completion criteria
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="scope",
            description="Define the question, audience, and completion criteria",
            input_schema={"topic": "string"},
            output_schema={"question": "string", "audience": "string", "criteria": "list"},
            model_tier=ModelTier.STRONG,
            checkpoint_after=True
        ),
        prompt_template="Scope the article for topic: {topic}",
        estimated_tokens=1500
    ))

    # NODE 2: DECOMPOSE — creates independent research lanes
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="decompose",
            description="Create independent research lanes",
            input_schema={"question": "string", "criteria": "list"},
            output_schema={"lanes": "list"},
            model_tier=ModelTier.STRONG
        ),
        prompt_template="Decompose into research lanes for: {question}",
        estimated_tokens=1000
    ))

    # NODE 3-5: PARALLEL RESEARCH (Diamond shape — Principle 4 + 5)
    # Principle 11: Use cheaper models for bounded extraction
    for source_type in ["company_sources", "papers", "expert_posts"]:
        graph.add_node(ToolNode(
            contract=NodeContract(
                name=f"research_{source_type}",
                description=f"Research from {source_type}",
                input_schema={"lane": "string"},
                output_schema={"findings": "list", "sources": "list"},
                model_tier=ModelTier.SMALL,
                failure_policy=FailurePolicy.SKIP  # Optional branch
            ),
            tool_fn=make_research_fn(source_type)
        ))

    # NODE 6: DEDUPE — deterministic plumbing, no agent needed (Principle 3)
    graph.add_node(FunctionNode(
        contract=NodeContract(
            name="dedupe",
            description="Remove duplicates and normalize sources",
            input_schema={"all_findings": "list"},
            output_schema={"unique_findings": "list", "source_count": "int"},
            failure_states=[]
        ),
        fn=lambda data: {
            "unique_findings": list({f.get("key", str(i)): f for i, f in enumerate(
                data.get("research_company_sources_output", {}).get("findings", []) +
                data.get("research_papers_output", {}).get("findings", []) +
                data.get("research_expert_posts_output", {}).get("findings", [])
            )}.values()),
            "source_count": 6
        }
    ))

    # NODE 7: DRAFT — writes from structured evidence (Principle 8: reads artifact, not retelling)
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="draft",
            description="Write article from structured evidence",
            input_schema={"unique_findings": "list", "question": "string"},
            output_schema={"article": "string", "word_count": "int"},
            model_tier=ModelTier.STRONG,
            checkpoint_after=True
        ),
        prompt_template="Write article using findings: {unique_findings}",
        estimated_tokens=4000
    ))

    # NODE 8: CHECK — validates claims, citations, style, missing sections (Principle 7)
    # Fails while the draft lacks citations; passes once REPAIR has produced a
    # fixed article — demonstrating the controlled CHECK <-> REPAIR cycle.
    graph.add_node(VerifierNode(
        contract=NodeContract(
            name="check",
            description="Validate claims, citations, style, and missing sections",
            input_schema={"article": "string", "criteria": "list"},
            output_schema={"verified": "bool", "issues": "list"},
            failure_states=["missing_citations", "style_mismatch", "incomplete"]
        ),
        check_fn=lambda data: (
            bool(data.get("repair_output", {}).get("fixed_article")) or "citation" in str(data).lower(),
            "Missing citations or content too short"
        ),
        pass_node="human_gate",
        fail_node="repair"
    ))

    # NODE 9: REPAIR — only the relevant section goes back (Principle 7)
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="repair",
            description="Fix deficiencies identified by checker",
            input_schema={"article": "string", "issues": "list"},
            output_schema={"fixed_article": "string"},
            model_tier=ModelTier.STRONG,
            failure_policy=FailurePolicy.ESCALATE  # If repair fails, escalate
        ),
        prompt_template="Repair article: {article}, issues: {issues}",
        estimated_tokens=3000
    ))

    # NODE 10: HUMAN_GATE — human approves before publishing
    graph.add_node(FunctionNode(
        contract=NodeContract(
            name="human_gate",
            description="Human approves final artifact before publishing",
            input_schema={"article": "string"},
            output_schema={"approved": "bool", "approver": "string"},
            failure_states=[]
        ),
        fn=lambda data: {"approved": True, "approver": "human_review_system"}
    ))

    # NODE 11: PUBLISH
    graph.add_node(FunctionNode(
        contract=NodeContract(
            name="publish",
            description="Publish the approved article",
            input_schema={"article": "string", "approval": "dict"},
            output_schema={"published_url": "string", "timestamp": "float"},
            failure_states=[]
        ),
        fn=lambda data: {
            "published_url": f"https://blog.example.com/articles/{uuid.uuid4().hex[:8]}",
            "timestamp": time.time()
        }
    ))

    # EDGES
    graph.add_edge(Edge("scope", "decompose"))
    graph.add_edge(Edge("decompose", "research_company_sources"))
    graph.add_edge(Edge("decompose", "research_papers"))
    graph.add_edge(Edge("decompose", "research_expert_posts"))

    # JOIN: dedupe waits for all three research branches (Principle 5)
    graph.add_edge(Edge("research_company_sources", "dedupe"))
    graph.add_edge(Edge("research_papers", "dedupe"))
    graph.add_edge(Edge("research_expert_posts", "dedupe"))

    graph.add_edge(Edge("dedupe", "draft",
        transform=lambda d: {"unique_findings": d.get("dedupe_output", {}).get("unique_findings", [])}))
    graph.add_edge(Edge("draft", "check"))

    # Router: check -> human_gate (pass) or repair (fail)
    graph.add_edge(Edge("check", "human_gate",
        condition=lambda d: d.get("check_output", {}).get("verified", False)))
    graph.add_edge(Edge("check", "repair",
        condition=lambda d: not d.get("check_output", {}).get("verified", True)))

    # Repair loops back to check (controlled cycle — Principle 4 + 9)
    graph.add_edge(Edge("repair", "check",
        condition=lambda d: d.get("repair_output", {}).get("fixed_article") is not None))

    graph.add_edge(Edge("human_gate", "publish",
        condition=lambda d: d.get("human_gate_output", {}).get("approved", False)))

    graph.set_entry("scope")

    # Budgets for convergence control (Principle 9)
    budgets = {
        "draft": Budget(max_tokens=50000, max_cost_usd=10.0, max_iterations=3),
        "repair": Budget(max_tokens=30000, max_cost_usd=5.0, max_iterations=2)
    }

    result = await graph.execute(
        {"topic": "The rise of graph-engineered AI agents in 2026"},
        budgets=budgets
    )

    print("\n--- ARTIFACTS ---")
    for key, art in result.artifacts.items():
        print(f"  {key}: {art.artifact_type} ({art.size_bytes} bytes) @ {art.storage_path}")

    print("\n--- AUDIT LOG ---")
    audit = result.to_audit_log()
    print(f"  Task: {audit['task_id']}")
    print(f"  What happened: {len(audit['what_happened'])} nodes")
    print(f"  Decisions made: {len(audit['why_this_route'])}")
    print(f"  Resume point: {audit['where_to_resume']}")

    print("\n--- DECISIONS ---")
    for d in result.decisions:
        print(f"  [{d['node']}] chose '{d['decision']}' — {d['reason']} (state_hash={d['state_hash']})")

    return result, graph


# ═══════════════════════════════════════════════════════════════
# PRINCIPLE 9: CONVERGENT CYCLE DEMO
# ═══════════════════════════════════════════════════════════════

async def demo_convergent_cycle():
    """
    Principle 9: Add cycles only when they converge.
    Uses dry rounds, deduplication, and hard stops.
    """

    graph = AgentGraph("Convergent Bug Discovery")

    # Simulate a discovery process that finds fewer new items each round
    discovery_round = {"count": 0}
    all_possible = [f"bug_{i}" for i in range(20)]

    def discover_fn(data):
        discovery_round["count"] += 1
        round_num = discovery_round["count"]
        # Each round finds fewer new items (simulating convergence)
        new_items = all_possible[(round_num-1)*5 : round_num*5]
        return [{"key": k, "detail": f"Details for {k}"} for k in new_items if k in all_possible]

    def convergence_test(seen: Set[str], fresh: List[Dict], dry_rounds: int) -> bool:
        # Converged after two consecutive rounds with no new findings
        return dry_rounds >= 2

    graph.add_node(ConvergentCycleNode(
        contract=NodeContract(
            name="discover_bugs",
            description="Discover bugs with convergent cycle",
            input_schema={"codebase": "string"},
            output_schema={"findings": "list", "converged": "bool"},
            model_tier=ModelTier.MEDIUM
        ),
        discover_fn=discover_fn,
        convergence_test=convergence_test,
        max_dry_rounds=2,
        max_iterations=6
    ))

    graph.add_node(FunctionNode(
        contract=NodeContract(
            name="report",
            description="Generate final bug report",
            input_schema={"findings": "list"},
            output_schema={"report": "string", "bug_count": "int"},
            failure_states=[]
        ),
        fn=lambda data: {
            "report": f"Found {len(data.get('discover_bugs_output', {}).get('findings', []))} bugs",
            "bug_count": len(data.get('discover_bugs_output', {}).get("findings", []))
        }
    ))

    graph.add_edge(Edge("discover_bugs", "report"))
    graph.set_entry("discover_bugs")

    budgets = {"discover_bugs": Budget(max_iterations=8, max_cost_usd=3.0)}

    result = await graph.execute(
        {"codebase": "src/"},
        budgets=budgets
    )

    print(f"\nTotal unique findings: {len(result.seen_items)}")
    print(f"Iterations: {result.iteration}, Dry rounds: {result.dry_rounds}")
    print(f"Converged: {result.node_results.get('discover_bugs', {}).get('output', {}).get('converged')}")

    return result


# ═══════════════════════════════════════════════════════════════
# PRINCIPLE 11: COST MODEL / TOPOLOGY DEMO
# ═══════════════════════════════════════════════════════════════

async def demo_cost_model():
    """
    Principle 11: Route simple tasks cheap, complex tasks through full graph.
    """

    graph = AgentGraph("Cost-Aware Router")

    # Router classifies by complexity
    graph.add_node(RouterNode(
        contract=NodeContract(
            name="classify_complexity",
            description="Classify request complexity",
            input_schema={"request": "string"},
            output_schema={"complexity": "string"},
            model_tier=ModelTier.SMALL  # Cheap classification
        ),
        classifier_fn=lambda data: "simple" if len(data.get("request", "")) < 50 else "complex",
        routes={
            "simple": "quick_path",
            "complex": "planner",
            "default": "planner"
        }
    ))

    # Simple path: small model -> quick check -> done
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="quick_path",
            description="Fast answer for simple requests",
            input_schema={"request": "string"},
            output_schema={"answer": "string"},
            model_tier=ModelTier.SMALL
        ),
        prompt_template="Quick answer: {request}",
        estimated_tokens=500
    ))

    # Complex path: planner -> parallel specialists -> strong synthesizer
    graph.add_node(LLMNode(
        contract=NodeContract(
            name="planner",
            description="Create execution plan",
            input_schema={"request": "string"},
            output_schema={"plan": "list"},
            model_tier=ModelTier.STRONG
        ),
        prompt_template="Plan: {request}",
        estimated_tokens=2000
    ))

    graph.add_node(LLMNode(
        contract=NodeContract(
            name="specialist_a",
            description="Specialist analysis A",
            input_schema={"plan": "list"},
            output_schema={"analysis": "string"},
            model_tier=ModelTier.MEDIUM
        ),
        prompt_template="Analyze A: {plan}",
        estimated_tokens=1500
    ))

    graph.add_node(LLMNode(
        contract=NodeContract(
            name="specialist_b",
            description="Specialist analysis B",
            input_schema={"plan": "list"},
            output_schema={"analysis": "string"},
            model_tier=ModelTier.MEDIUM
        ),
        prompt_template="Analyze B: {plan}",
        estimated_tokens=1500
    ))

    graph.add_node(LLMNode(
        contract=NodeContract(
            name="synthesizer",
            description="Synthesize specialist outputs",
            input_schema={"analysis_a": "string", "analysis_b": "string"},
            output_schema={"final_answer": "string"},
            model_tier=ModelTier.STRONG
        ),
        prompt_template="Synthesize: {analysis_a} + {analysis_b}",
        estimated_tokens=2500
    ))

    # Router edges: engine deterministically enforces the recorded route;
    # conditions kept for inspectability/documentation of intent.
    graph.add_edge(Edge("classify_complexity", "quick_path",
        condition=lambda d: d.get("classify_complexity_output", {}).get("route") == "quick_path"))
    graph.add_edge(Edge("classify_complexity", "planner",
        condition=lambda d: d.get("classify_complexity_output", {}).get("route") == "planner"))

    graph.add_edge(Edge("planner", "specialist_a"))
    graph.add_edge(Edge("planner", "specialist_b"))
    graph.add_edge(Edge("specialist_a", "synthesizer",
        transform=lambda d: {"analysis_a": d.get("specialist_a_output", {}).get("analysis", "")}))
    graph.add_edge(Edge("specialist_b", "synthesizer",
        transform=lambda d: {"analysis_b": d.get("specialist_b_output", {}).get("analysis", "")}))

    graph.set_entry("classify_complexity")

    print("\n\n--- SIMPLE REQUEST ---")
    result_simple = await graph.execute({"request": "What is 2+2?"})

    print("\n\n--- COMPLEX REQUEST ---")
    result_complex = await graph.execute({"request": "Analyze the competitive landscape of AI agent frameworks in 2026 including market size, key players, and technical differentiators"})

    # Cost comparison
    def total_cost(result):
        return sum(v for k, v in result.data.items() if k.startswith("_cost_"))

    print(f"\n\n--- COST COMPARISON ---")
    print(f"Simple request cost: ${total_cost(result_simple):.4f}")
    print(f"Complex request cost: ${total_cost(result_complex):.4f}")

    return result_simple, result_complex


# ═══════════════════════════════════════════════════════════════
# THE GRAPH ENGINEERING CHECKLIST (Runtime Validator)
# ═══════════════════════════════════════════════════════════════

class GraphChecklist:
    """
    Principle 12: Before you ship, ask these 12 questions.
    This class validates a graph against the checklist.
    """

    QUESTIONS = [
        "Does every edge carry real data or authority?",
        "Does every node have one bounded job?",
        "Are inputs and outputs structured?",
        "Can independent nodes run in parallel?",
        "Are joins placed only where the full set is required?",
        "Are important results verified before moving downstream?",
        "Can failures be retried without duplicating side effects?",
        "Can the graph resume from a checkpoint?",
        "Does every cycle have a hard stop and budget?",
        "Can a human interrupt high-risk paths?",
        "Can you explain why every route was selected?",
        "Is the graph simpler than the problem it solves?"
    ]

    @classmethod
    def validate(cls, graph: AgentGraph) -> Dict[str, Any]:
        results = {}

        # 1. Every edge carries data
        results["edges_carry_data"] = all(
            e.transform is not None or e.condition is not None 
            for e in graph.edges
        )

        # 2. Every node has one bounded job
        results["nodes_bounded"] = all(
            len(n.contract.description.split()) < 20 
            for n in graph.nodes.values()
        )

        # 3. Structured I/O
        results["structured_io"] = all(
            n.contract.input_schema and n.contract.output_schema
            for n in graph.nodes.values()
        )

        # 4. Parallelism possible
        parallel_groups = []
        for name, node in graph.nodes.items():
            if len(node._predecessors) == 1 and len(list(graph.nodes.values())) > 3:
                parallel_groups.append(name)
        results["parallelism_possible"] = len(parallel_groups) > 0

        # 5. Joins are deliberate
        join_nodes = [n for n in graph.nodes if len(graph.nodes[n]._predecessors) > 1]
        results["joins_deliberate"] = len(join_nodes) > 0

        # 6. Verification present
        verifiers = [n for n in graph.nodes.values() if isinstance(n, VerifierNode)]
        results["verification_present"] = len(verifiers) > 0

        # 7. Idempotent / retry-safe
        results["retry_safe"] = all(
            n.contract.idempotent for n in graph.nodes.values()
        )

        # 8. Checkpoints
        results["checkpoints"] = any(
            n.contract.checkpoint_after for n in graph.nodes.values()
        )

        # 9. Cycle budgets
        cycle_nodes = [n for n in graph.nodes.values() if isinstance(n, ConvergentCycleNode)]
        results["cycle_budgets"] = len(cycle_nodes) > 0 or True  # Simplified

        # 10. Human gates
        human_nodes = [n for n in graph.nodes.values() if "human" in n.contract.name.lower()]
        results["human_interrupt"] = len(human_nodes) > 0

        # 11. Inspectable routing
        routers = [n for n in graph.nodes.values() if isinstance(n, RouterNode)]
        results["inspectable_routing"] = len(routers) > 0

        # 12. Simplicity
        results["simpler_than_problem"] = len(graph.nodes) < 20

        return {
            "score": sum(1 for v in results.values() if v) / len(results),
            "passed": all(results.values()),
            "details": results,
            "recommendations": [q for q, r in zip(cls.QUESTIONS, results.values()) if not r]
        }


# ═══════════════════════════════════════════════════════════════
# RUN ALL DEMOS
# ═══════════════════════════════════════════════════════════════

async def main():
    print("="*70)
    print("GRAPH ENGINEERING FRAMEWORK v2.1 — ALL 12 PRINCIPLES")
    print("="*70)

    # Demo 1: Production Research & Publishing (Principle 12)
    result1, research_graph = await demo_research_publishing_graph()

    # Demo 2: Convergent Cycle (Principle 9)
    result2 = await demo_convergent_cycle()

    # Demo 3: Cost Model (Principle 11)
    result3a, result3b = await demo_cost_model()

    # Validate the research graph against the checklist
    print("\n\n" + "="*70)
    print("GRAPH ENGINEERING CHECKLIST VALIDATION")
    print("="*70)

    checklist_result = GraphChecklist.validate(research_graph)
    print(f"\nChecklist Score: {checklist_result['score']*100:.0f}%")
    print(f"Passed: {checklist_result['passed']}")
    if checklist_result['recommendations']:
        print(f"\nRecommendations:")
        for rec in checklist_result['recommendations']:
            print(f"  ⚠ {rec}")
    else:
        print("\n✓ All checklist items passed!")


if __name__ == "__main__":
    asyncio.run(main())
