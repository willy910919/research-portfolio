from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .contracts import LeadDecision, canonical_hash
from .reconciliation import (
    assess_capability_fit,
    build_analysis_graph,
    extract_deterministic_facts,
    validate_analysis_graph,
)
from .tool_registry import get_tool_manifest, resolve_tool_requests
from .user_requirements import (
    build_user_requirement,
    project_requirement_to_task_spec,
    verify_user_requirement,
)


_OUTPUT_ALIASES = {
    "counts": "count",
    "frequencies": "count",
    "frequency": "count",
    "percents": "percentage",
    "percent": "percentage",
    "percentages": "percentage",
    "proportions": "percentage",
    "proportion": "percentage",
    "odds_ratios": "odds_ratio",
    "confidence_intervals": "confidence_interval",
    "p_values": "p_value",
    "coefficients": "coefficient",
}


def _unique(values: Sequence[Any]) -> list[Any]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _normalize_output(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _OUTPUT_ALIASES.get(text, text)
    if "_by_" in normalized:
        base = _OUTPUT_ALIASES.get(normalized.split("_by_", 1)[0], "")
        if base in {"count", "percentage"}:
            return base
    return normalized


def _overlay_explicit_facts(
    task_spec: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Overlay only facts that can be located in the current user utterance."""

    spec = dict(task_spec or {})
    for field in (
        "outcomes",
        "exposures",
        "predictors",
        "candidate_features",
        "forced_covariates",
        "group_by",
        "requested_methods",
    ):
        values = list(facts.get(field) or [])
        if values:
            spec[field] = values
    if list(facts.get("filters") or []):
        spec["filters"] = list(facts.get("filters") or [])
    if str(facts.get("candidate_scope") or ""):
        spec["candidate_scope"] = str(facts.get("candidate_scope") or "")
    if int(facts.get("top_k") or 0):
        spec["top_k"] = int(facts.get("top_k") or 0)
    if str(facts.get("method_selection_authority") or ""):
        spec["method_selection_authority"] = str(
            facts.get("method_selection_authority") or ""
        )

    requested_outputs = _unique(
        [
            _normalize_output(item)
            for item in (
                list(spec.get("required_outputs") or [])
                + list(facts.get("required_outputs") or [])
            )
            if str(item).strip()
        ]
    )
    spec["required_outputs"] = requested_outputs
    spec["explicit_facts"] = dict(facts)
    return spec


def _clear_resolved_clarification(task_spec: Mapping[str, Any]) -> Dict[str, Any]:
    spec = dict(task_spec or {})
    question = str(spec.get("clarification_question") or "").strip().lower()
    if not question:
        return spec
    resolved = False
    if any(cue in question for cue in ("outcome", "target", "結果變項", "依變項")):
        resolved = len(list(spec.get("outcomes") or [])) == 1
    elif any(cue in question for cue in ("至少兩個", "納入分析的欄位", "predictor")):
        resolved = len(list(spec.get("predictors") or [])) >= 2
    elif any(cue in question for cue in ("分組欄位", "group by")):
        resolved = bool(spec.get("group_by"))
    elif any(cue in question for cue in ("分析方法", "指定方法", "method")):
        resolved = bool(spec.get("requested_methods") or spec.get("selected_methods"))
    if resolved:
        spec["needs_clarification"] = False
        spec["clarification_question"] = ""
    return spec


def preserve_descriptive_requests(
    task_spec: Mapping[str, Any],
    current_requests: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Retain the Lead Agent's exact descriptive columns during minimal repair.

    A generic summary request cannot reconstruct column roles from filters.
    Reuse registered requests instead of guessing columns or dropping a valid
    describe step. Their relative order and explicit data source stay intact.
    """
    outputs = set(task_spec.get("required_outputs") or [])
    if not outputs.intersection({"summary_statistics", "mean", "count"}):
        return []
    return [dict(request) for request in current_requests
            if str(request.get("tool") or "") == "describe"
            and isinstance(request.get("columns"), (list, tuple))
            and request.get("columns")]


def compile_lead_plan(
    question: str,
    plan: Mapping[str, Any],
    columns: Sequence[str],
    *,
    existing_requirement: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compile the Lead Agent plan without creating a second decision-maker.

    The Lead Agent owns semantic planning. Deterministic extraction may only
    restore explicit fields, numbers and operators from the current utterance.
    Tool Broker resolution and the legacy graph validation are evidence about
    executability; neither may rewrite the immutable UserRequirement.
    """

    result = dict(plan or {})
    facts = extract_deterministic_facts(question, columns)
    task_spec_seed = dict(result.get("task_spec") or {})
    if bool(result.get("needs_clarification")):
        task_spec_seed.setdefault("needs_clarification", True)
        task_spec_seed.setdefault(
            "clarification_question",
            str(result.get("clarification_question") or ""),
        )
    task_spec = _clear_resolved_clarification(_overlay_explicit_facts(
        task_spec_seed,
        facts,
    ))
    requirement = build_user_requirement(
        question,
        task_spec,
        facts,
        existing=(
            dict(existing_requirement or {})
            if verify_user_requirement(dict(existing_requirement or {}))
            else None
        ),
    )
    task_spec = project_requirement_to_task_spec(requirement, task_spec)
    task_spec = _clear_resolved_clarification(task_spec)
    task_spec.pop("spec_id", None)
    task_spec.pop("spec_hash", None)
    task_spec["spec_id"] = "SPEC_" + canonical_hash(task_spec)[:16]
    task_spec["spec_hash"] = canonical_hash(task_spec)

    raw_requests = [
        dict(item)
        for item in list(result.get("tool_requests") or [])
        if isinstance(item, Mapping)
    ]
    broker = resolve_tool_requests(raw_requests)
    accepted = list(broker.get("accepted") or [])
    graph = build_analysis_graph(task_spec, accepted)
    validation = validate_analysis_graph(task_spec, graph, columns)
    capability_fit = assess_capability_fit(task_spec, graph, facts)

    result["user_requirement"] = requirement
    result["task_spec"] = task_spec
    result["tool_requests"] = accepted
    result["tool_broker_resolution"] = broker
    result["analysis_graph"] = graph
    result["contract_validation"] = {
        **validation,
        "authority": "advisory_only",
    }
    result["capability_fit"] = capability_fit
    if not bool(task_spec.get("needs_clarification")):
        result["needs_clarification"] = False
        result["clarification_question"] = ""
    result["lead_compilation"] = {
        "authority": "lead_agent_plus_explicit_facts",
        "requirement_hash": requirement.get("requirement_hash"),
        "broker_complete": bool(broker.get("complete")),
        "legacy_validation_authority": "advisory_only",
    }

    resources = dict(facts.get("explicit_resource_requests") or {})
    if resources.get("literature"):
        result["needs_literature"] = True
    if resources.get("external"):
        result["needs_external_information"] = True
    if resources.get("custom_code"):
        result["needs_code"] = True
    if capability_fit.get("minimum_sufficient_resources"):
        result["needs_literature"] = False
        result["literature_queries"] = []
        result["needs_external_information"] = False
        result["external_information_queries"] = []
        result["needs_code"] = False
        result["delegations"] = []
        result["resource_pruning"] = {
            "applied": True,
            "reason": "Verified built-in tools fully cover this request.",
        }
    return result


def build_lead_decision(
    question: str,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create the one public decision record consumed by all workers."""

    requirement = dict(plan.get("user_requirement") or {})
    complexity = dict(plan.get("complexity_spec") or {})
    resource_plan = dict(plan.get("resource_plan") or {})
    requests = [
        dict(item)
        for item in list(plan.get("tool_requests") or [])
        if isinstance(item, Mapping)
    ]
    tool_ids = _unique(
        [str(item.get("tool") or "") for item in requests if item.get("tool")]
    )
    branches = _unique(
        [
            str(
                getattr(get_tool_manifest(tool_id), "branch", "analysis")
                or "analysis"
            )
            for tool_id in tool_ids
        ]
    )
    if bool(plan.get("needs_literature")):
        branches.append("evidence")
    if bool(plan.get("needs_external_information")):
        branches.append("external_resources")
    hypothesis_workflow = dict(plan.get("hypothesis_workflow") or {})
    if bool(hypothesis_workflow.get("enabled")):
        branches.append("hypothesis")
    branches = _unique(branches)
    workers = _unique(
        [
            str(item)
            for item in list(resource_plan.get("selected_roles") or [])
            if str(item)
        ]
    )
    if bool(hypothesis_workflow.get("enabled")):
        workers = _unique(
            workers
            + [
                str(item.get("role") or "")
                for item in list(hypothesis_workflow.get("nodes") or [])
                if isinstance(item, Mapping) and str(item.get("role") or "")
            ]
        )
    relation = str(
        plan.get("task_relation")
        or dict(plan.get("memory_resolution") or {}).get("relation")
        or "independent"
    ).strip().lower()
    if relation not in {"independent", "continuation", "revision"}:
        relation = "independent"
    decision = LeadDecision(
        question=str(question or ""),
        relation=relation,
        requirement_hash=str(requirement.get("requirement_hash") or ""),
        task_id=str(requirement.get("task_id") or ""),
        complexity_level=int(complexity.get("level") or 0),
        complexity_label=str(complexity.get("label") or "low"),
        execution_mode=str(
            resource_plan.get("execution_mode")
            or plan.get("execution_mode")
            or "direct_execution"
        ),
        tool_ids=tool_ids,
        branches=branches,
        worker_roles=workers,
        needs_literature=bool(plan.get("needs_literature")),
        needs_clarification=bool(
            requirement.get("needs_clarification")
            or plan.get("needs_clarification")
        ),
        clarification_question=str(
            requirement.get("clarification_question")
            or plan.get("clarification_question")
            or ""
        ),
    )
    return decision.to_dict()
