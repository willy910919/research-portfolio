from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}_{canonical_hash(value)[:16]}"


@dataclass(frozen=True)
class TypedResearchTaskSpec:
    question: str
    goal: str = ""
    task_family: str = ""
    outcomes: List[str] = field(default_factory=list)
    exposures: List[str] = field(default_factory=list)
    predictors: List[str] = field(default_factory=list)
    candidate_features: List[str] = field(default_factory=list)
    candidate_scope: str = ""
    forced_covariates: List[str] = field(default_factory=list)
    excluded_features: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    requested_methods: List[str] = field(default_factory=list)
    selected_methods: List[str] = field(default_factory=list)
    method_selection_authority: str = ""
    required_outputs: List[str] = field(default_factory=list)
    top_k: int = 0
    needs_clarification: bool = False
    clarification_question: str = ""
    explicit_facts: Dict[str, Any] = field(default_factory=dict)
    reconciliation_notes: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    spec_version: str = "v0.9.6"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["spec_id"] = _stable_id("SPEC", payload)
        payload["spec_hash"] = canonical_hash(payload)
        return payload


@dataclass(frozen=True)
class LeadDecision:
    """The single Lead Agent decision for one user turn.

    This record points to the immutable UserRequirement and describes how the
    turn will be executed. It is not another requirement contract: workers may
    consume it, but they cannot change the referenced requirement.
    """

    question: str
    relation: str
    requirement_hash: str
    task_id: str
    complexity_level: int = 0
    complexity_label: str = "low"
    execution_mode: str = "direct_execution"
    tool_ids: List[str] = field(default_factory=list)
    branches: List[str] = field(default_factory=list)
    worker_roles: List[str] = field(default_factory=list)
    needs_literature: bool = False
    needs_clarification: bool = False
    clarification_question: str = ""
    decision_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["decision_id"] = _stable_id("LEAD", payload)
        payload["decision_hash"] = canonical_hash(payload)
        payload["authority"] = "lead_agent"
        return payload


@dataclass(frozen=True)
class AnalysisNode:
    node_id: str
    capability: str
    request: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    input_contract: Dict[str, Any] = field(default_factory=dict)
    output_contract: List[str] = field(default_factory=list)
    composition_role: str = "atomic_operation"
    required: bool = True
    branch: str = "analysis"
    criticality: str = "core"
    timeout_sec: int = 60
    retry: int = 0
    queue_class: str = "statistics"
    resource_profile: Dict[str, Any] = field(default_factory=dict)
    error_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskNode(AnalysisNode):
    """V2 DAG node with branch-local execution and recovery policy."""

    node_version: str = "2.0"


@dataclass(frozen=True)
class AnalysisGraph:
    spec_hash: str
    nodes: List[AnalysisNode] = field(default_factory=list)
    required_outputs: List[str] = field(default_factory=list)
    output_bindings: Dict[str, List[str]] = field(default_factory=dict)
    recomposition_policy: str = "all_required_outputs_from_completed_nodes"
    graph_version: str = "v2"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "spec_hash": self.spec_hash,
            "nodes": [node.to_dict() for node in self.nodes],
            "required_outputs": list(self.required_outputs),
            "output_bindings": {
                str(name): list(node_ids)
                for name, node_ids in self.output_bindings.items()
            },
            "recomposition_policy": self.recomposition_policy,
            "graph_version": self.graph_version,
        }
        payload["graph_id"] = _stable_id("GRAPH", payload)
        payload["graph_hash"] = canonical_hash(payload)
        return payload


@dataclass
class ExecutionStep:
    node_id: str
    capability: str
    request: Dict[str, Any]
    status: str = "pending"
    started_at: str = ""
    completed_at: str = ""
    evidence_manifest: Dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""
    attempt_count: int = 0
    repair_records: List[Dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        self.status = "running"
        if not self.started_at:
            self.started_at = utc_now()
        self.attempt_count += 1

    def complete(self, evidence_manifest: Mapping[str, Any]) -> None:
        self.status = "completed"
        self.completed_at = utc_now()
        self.evidence_manifest = dict(evidence_manifest or {})
        self.error_type = ""
        self.error_message = ""

    def fail(self, exc: BaseException) -> None:
        self.status = "failed"
        self.completed_at = utc_now()
        self.error_type = type(exc).__name__
        self.error_message = str(exc)

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.completed_at = utc_now()
        self.error_type = "DependencyUnavailable"
        self.error_message = str(reason)

    def record_repair(self, repair: Mapping[str, Any]) -> None:
        self.repair_records.append(dict(repair or {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionManifest:
    spec_hash: str
    graph_hash: str
    requirement_hash: str = ""
    expected_outputs: List[str] = field(default_factory=list)
    requested_methods: List[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: f"EXEC_{uuid.uuid4().hex[:16]}")
    started_at: str = field(default_factory=utc_now)
    completed_at: str = ""
    steps: List[ExecutionStep] = field(default_factory=list)
    completed_outputs: List[str] = field(default_factory=list)
    missing_outputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "running"
    manifest_version: str = "v2"

    def finish(self) -> None:
        self.completed_at = utc_now()
        completed = {
            str(output)
            for step in self.steps
            if step.status == "completed"
            for output in list(step.evidence_manifest.get("declared_outputs") or [])
            if str(output)
        }
        expected = {str(item) for item in self.expected_outputs if str(item)}
        self.completed_outputs = sorted(completed)
        self.missing_outputs = sorted(expected - completed)
        completed_steps = sum(step.status == "completed" for step in self.steps)
        failed_steps = sum(
            step.status in {"failed", "skipped", "cancelled"}
            for step in self.steps
        )
        if completed_steps and (failed_steps or self.missing_outputs):
            self.status = "partial"
        elif completed_steps or not self.steps:
            self.status = "completed"
        else:
            self.status = "failed"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "spec_hash": self.spec_hash,
            "graph_hash": self.graph_hash,
            "requirement_hash": self.requirement_hash,
            "expected_outputs": list(self.expected_outputs),
            "requested_methods": list(self.requested_methods),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [step.to_dict() for step in self.steps],
            "completed_outputs": list(self.completed_outputs),
            "missing_outputs": list(self.missing_outputs),
            "warnings": list(self.warnings),
            "status": self.status,
            "manifest_version": self.manifest_version,
        }
        payload["manifest_hash"] = canonical_hash(payload)
        return payload
