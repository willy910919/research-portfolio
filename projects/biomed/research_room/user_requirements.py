from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .contracts import canonical_hash
from .filtering import normalize_filter_conditions
from .resource_intent import LITERATURE_OUTPUTS, parse_resource_intents, tool_resource


def _unique(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )


def _json_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _decode_json_value(value: str) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return value


def _requirement_hash_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return stable requirement semantics without runtime lineage ids."""

    result = dict(payload)
    result.pop("requirement_hash", None)
    result.pop("task_id", None)
    result.pop("parent_task_id", None)
    return result


def _find_span(question: str, value: Any) -> str:
    text = str(question or "")
    candidates = [str(value or "").strip()]
    normalized = str(value or "").strip().lower()
    aliases = {
        "logistic_regression": [
            "logistic regression",
            "logistic",
            "邏輯斯迴歸",
        ],
        "linear_regression": ["linear regression", "線性迴歸"],
        "lasso": ["lasso"],
        "pca": ["pca", "主成分分析"],
        "kmeans": ["k-means", "kmeans", "分群"],
        "odds_ratio": ["odds ratio", "or"],
        "confidence_interval": ["95% ci", "95％ ci", "confidence interval"],
        "p_value": ["p-value", "p value", "p值"],
        "selected_features": ["特徵選擇", "候選特徵", "重要特徵"],
        "literature_evidence": ["文獻", "pubmed", "pmid", "研究支持"],
        "pmid_list": ["pubmed", "pmid", "文獻"],
        "figure": ["圖", "plot", "chart", "heatmap", "熱力圖"],
    }
    candidates.extend(aliases.get(normalized, []))
    for candidate in sorted(
        {item for item in candidates if item},
        key=len,
        reverse=True,
    ):
        match = re.search(re.escape(candidate), text, flags=re.IGNORECASE)
        if match:
            return text[match.start() : match.end()]
    return ""


@dataclass(frozen=True)
class RequirementProvenance:
    field: str
    value_json: str
    source: str
    span: str = ""
    certainty: str = "explicit"

    @property
    def value(self) -> Any:
        return _decode_json_value(self.value_json)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["value"] = self.value
        payload.pop("value_json", None)
        return payload


@dataclass(frozen=True)
class RequirementFilter:
    field: str
    operator: str
    value_json: str
    logic: str = "and"

    @property
    def value(self) -> Any:
        return _decode_json_value(self.value_json)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "logic": self.logic,
        }


@dataclass(frozen=True)
class UserRequirement:
    """The immutable, user-facing source of truth for one research task.

    Planner output may add implementation defaults, but it cannot mutate these
    fields. Binding decisions are identified by provenance with
    certainty="explicit".
    """

    task_id: str
    parent_task_id: str = ""
    dataset_version: str = ""
    task: str = ""
    outcomes: Tuple[str, ...] = field(default_factory=tuple)
    exposures: Tuple[str, ...] = field(default_factory=tuple)
    predictors: Tuple[str, ...] = field(default_factory=tuple)
    candidate_scope: str = ""
    candidate_features: Tuple[str, ...] = field(default_factory=tuple)
    forced_covariates: Tuple[str, ...] = field(default_factory=tuple)
    excluded_features: Tuple[str, ...] = field(default_factory=tuple)
    filters: Tuple[RequirementFilter, ...] = field(default_factory=tuple)
    group_by: Tuple[str, ...] = field(default_factory=tuple)
    requested_methods: Tuple[str, ...] = field(default_factory=tuple)
    requested_outputs: Tuple[str, ...] = field(default_factory=tuple)
    top_k: int = 0
    evidence_requested: bool = False
    prohibited_resources: Tuple[str, ...] = field(default_factory=tuple)
    needs_clarification: bool = False
    clarification_question: str = ""
    provenance: Tuple[RequirementProvenance, ...] = field(default_factory=tuple)
    contract_version: str = "v1"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "dataset_version": self.dataset_version,
            "task": self.task,
            "outcomes": list(self.outcomes),
            "exposures": list(self.exposures),
            "predictors": list(self.predictors),
            "candidate_scope": self.candidate_scope,
            "candidate_features": list(self.candidate_features),
            "forced_covariates": list(self.forced_covariates),
            "excluded_features": list(self.excluded_features),
            "filters": [item.to_dict() for item in self.filters],
            "group_by": list(self.group_by),
            "requested_methods": list(self.requested_methods),
            "requested_outputs": list(self.requested_outputs),
            "top_k": int(self.top_k),
            "evidence_requested": bool(self.evidence_requested),
            "prohibited_resources": list(self.prohibited_resources),
            "needs_clarification": bool(self.needs_clarification),
            "clarification_question": self.clarification_question,
            "provenance": [item.to_dict() for item in self.provenance],
            "contract_version": self.contract_version,
            "immutable": True,
        }
        payload["binding_fields"] = sorted(
            {
                item.field
                for item in self.provenance
                if item.certainty == "explicit"
            }
        )
        payload["requirement_hash"] = canonical_hash(
            _requirement_hash_payload(payload)
        )
        return payload


def _explicit_values(facts: Mapping[str, Any], field: str) -> set[str]:
    aliases = {
        "outcomes": "outcomes",
        "exposures": "exposures",
        "predictors": "predictors",
        "candidate_features": "candidate_features",
        "forced_covariates": "forced_covariates",
        "group_by": "group_by",
        "requested_methods": "requested_methods",
        "requested_outputs": "required_outputs",
    }
    key = aliases.get(field, field)
    return {
        str(item)
        for item in list(facts.get(key) or [])
        if str(item)
    }


def _provenance_for_values(
    question: str,
    facts: Mapping[str, Any],
    field: str,
    values: Sequence[Any],
) -> List[RequirementProvenance]:
    explicit_values = _explicit_values(facts, field)
    records: List[RequirementProvenance] = []
    for value in values:
        span = _find_span(question, value)
        if field == "requested_outputs":
            span = str(dict(facts.get("required_output_spans") or {}).get(str(value)) or span)
        # Mentioning a column is not proof of its role.  For example,
        # "BMI 與 HYPERTENSION_SELF 的關聯" contains both names, but only the
        # semantic plan determines which is outcome and which is exposure.
        # A value is binding only when the deterministic extractor found an
        # explicit role cue for this field.
        explicit = str(value) in explicit_values
        records.append(
            RequirementProvenance(
                field=field,
                value_json=_json_value(value),
                source=(
                    "user_original_text"
                    if explicit
                    else "research_agent_semantic_inference"
                ),
                span=span,
                certainty="explicit" if explicit else "inferred",
            )
        )
    return records


def _normalize_inferred_role_overlaps(
    fields: Dict[str, Tuple[str, ...]],
    facts: Mapping[str, Any],
) -> Dict[str, Tuple[str, ...]]:
    """Remove compiler-created role overlap without hiding user conflicts.

    Explicitly assigning the same column to incompatible roles remains a hard
    error.  If only one side is explicit, that assignment wins.  If neither
    side is explicit, the Lead Agent's outcome role has precedence and the
    duplicate inferred role is dropped before execution.
    """

    normalized = dict(fields)
    explicit_outcomes = _explicit_values(facts, "outcomes")
    outcome_values = list(normalized.get("outcomes") or ())

    for other_field in (
        "exposures",
        "forced_covariates",
        "candidate_features",
    ):
        other_values = list(normalized.get(other_field) or ())
        explicit_other = _explicit_values(facts, other_field)
        overlap = set(outcome_values).intersection(other_values)
        for column in overlap:
            outcome_explicit = column in explicit_outcomes
            other_explicit = column in explicit_other
            if outcome_explicit and other_explicit:
                continue
            if other_explicit and not outcome_explicit:
                outcome_values = [item for item in outcome_values if item != column]
            else:
                other_values = [item for item in other_values if item != column]
        normalized[other_field] = tuple(other_values)

    normalized["outcomes"] = tuple(outcome_values)
    return normalized


def _requirement_from_mapping(value: Mapping[str, Any]) -> UserRequirement:
    normalized_filters = normalize_filter_conditions(value.get("filters") or [])
    filters = tuple(
        RequirementFilter(
            field=str(item.get("field") or ""),
            operator=str(item.get("operator") or "=="),
            value_json=_json_value(item.get("value")),
            logic=str(item.get("logic") or "and"),
        )
        for item in normalized_filters
        if isinstance(item, Mapping) and str(item.get("field") or "")
    )
    provenance = tuple(
        RequirementProvenance(
            field=str(item.get("field") or ""),
            value_json=_json_value(item.get("value")),
            source=str(item.get("source") or ""),
            span=str(item.get("span") or ""),
            certainty=str(item.get("certainty") or "inferred"),
        )
        for item in list(value.get("provenance") or [])
        if isinstance(item, Mapping)
    )
    return UserRequirement(
        task_id=str(value.get("task_id") or f"TASK_{uuid.uuid4().hex[:16]}"),
        parent_task_id=str(value.get("parent_task_id") or ""),
        dataset_version=str(value.get("dataset_version") or ""),
        task=str(value.get("task") or ""),
        outcomes=_unique(value.get("outcomes") or []),
        exposures=_unique(value.get("exposures") or []),
        predictors=_unique(value.get("predictors") or []),
        candidate_scope=str(value.get("candidate_scope") or ""),
        candidate_features=_unique(value.get("candidate_features") or []),
        forced_covariates=_unique(value.get("forced_covariates") or []),
        excluded_features=_unique(value.get("excluded_features") or []),
        filters=filters,
        group_by=_unique(value.get("group_by") or []),
        requested_methods=_unique(value.get("requested_methods") or []),
        requested_outputs=_unique(value.get("requested_outputs") or []),
        top_k=int(value.get("top_k") or 0),
        evidence_requested=bool(value.get("evidence_requested")),
        prohibited_resources=_unique(value.get("prohibited_resources") or []),
        needs_clarification=bool(value.get("needs_clarification")),
        clarification_question=str(value.get("clarification_question") or ""),
        provenance=provenance,
        contract_version=str(value.get("contract_version") or "v1"),
    )


def verify_user_requirement(value: Mapping[str, Any]) -> bool:
    if not value or not bool(value.get("immutable")):
        return False
    expected = str(value.get("requirement_hash") or "")
    return bool(expected) and canonical_hash(
        _requirement_hash_payload(value)
    ) == expected


def build_user_requirement(
    question: str,
    task_spec: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    existing: Mapping[str, Any] | None = None,
    dataset_version: str = "",
    task_id: str = "",
    parent_task_id: str = "",
) -> Dict[str, Any]:
    """Create or preserve the immutable requirement for the current task."""

    if existing and verify_user_requirement(existing):
        return dict(existing)

    spec = dict(task_spec or {})
    fact_map = dict(facts or {})
    resource_intents = parse_resource_intents(question)
    resource_requests = {key: value["state"] == "requested" for key, value in resource_intents.items()}
    prohibited_resources = tuple(key for key, value in resource_intents.items() if value["state"] == "prohibited")
    normalized_filters = normalize_filter_conditions(spec.get("filters") or [])
    filters = tuple(
        RequirementFilter(
            field=str(item.get("field") or ""),
            operator=str(item.get("operator") or "=="),
            value_json=_json_value(item.get("value")),
            logic=str(item.get("logic") or "and"),
        )
        for item in normalized_filters
        if isinstance(item, Mapping) and str(item.get("field") or "")
    )
    fields: Dict[str, Tuple[str, ...]] = {
        "outcomes": _unique(spec.get("outcomes") or []),
        "exposures": _unique(spec.get("exposures") or []),
        "predictors": _unique(spec.get("predictors") or []),
        "candidate_features": _unique(spec.get("candidate_features") or []),
        "forced_covariates": _unique(spec.get("forced_covariates") or []),
        "excluded_features": _unique(spec.get("excluded_features") or []),
        "group_by": _unique(spec.get("group_by") or []),
        "requested_methods": _unique(spec.get("requested_methods") or []),
        "requested_outputs": _unique(spec.get("required_outputs") or []),
    }
    fields = _normalize_inferred_role_overlaps(fields, fact_map)
    if "literature" in prohibited_resources:
        fields["requested_outputs"] = tuple(value for value in fields["requested_outputs"] if value not in LITERATURE_OUTPUTS)
    provenance: List[RequirementProvenance] = []
    for field_name, values in fields.items():
        provenance.extend(
            _provenance_for_values(
                question,
                fact_map,
                field_name,
                values,
            )
        )
    for item in filters:
        span = _find_span(question, item.field)
        provenance.append(
            RequirementProvenance(
                field="filters",
                value_json=_json_value(item.to_dict()),
                source="user_original_text",
                span=span,
                certainty="explicit",
            )
        )
    if int(spec.get("top_k") or 0):
        explicit_top_k = int(fact_map.get("top_k") or 0)
        provenance.append(
            RequirementProvenance(
                field="top_k",
                value_json=_json_value(int(spec.get("top_k") or 0)),
                source=(
                    "user_original_text"
                    if explicit_top_k
                    else "research_agent_semantic_inference"
                ),
                span=(
                    _find_span(question, explicit_top_k)
                    if explicit_top_k
                    else ""
                ),
                certainty="explicit" if explicit_top_k else "inferred",
            )
        )
    evidence_requested = bool(resource_requests.get("literature"))
    for resource in prohibited_resources:
        provenance.append(RequirementProvenance(
            field="prohibited_resources", value_json=_json_value(resource), source="user_original_text",
            span=resource_intents[resource]["spans"][-1]["text"], certainty="explicit",
        ))
    if evidence_requested:
        provenance.append(
            RequirementProvenance(
                field="evidence_requested",
                value_json="true",
                source="user_original_text",
                span=(
                    _find_span(question, "PubMed")
                    or _find_span(question, "文獻")
                ),
                certainty="explicit",
            )
        )
    requirement = UserRequirement(
        task_id=task_id or f"TASK_{uuid.uuid4().hex[:16]}",
        parent_task_id=parent_task_id,
        dataset_version=dataset_version,
        task=str(spec.get("goal") or spec.get("task_family") or ""),
        outcomes=fields["outcomes"],
        exposures=fields["exposures"],
        predictors=fields["predictors"],
        candidate_scope=str(spec.get("candidate_scope") or ""),
        candidate_features=fields["candidate_features"],
        forced_covariates=fields["forced_covariates"],
        excluded_features=fields["excluded_features"],
        filters=filters,
        group_by=fields["group_by"],
        requested_methods=fields["requested_methods"],
        requested_outputs=fields["requested_outputs"],
        top_k=int(spec.get("top_k") or 0),
        evidence_requested=evidence_requested,
        prohibited_resources=prohibited_resources,
        needs_clarification=bool(spec.get("needs_clarification")),
        clarification_question=str(spec.get("clarification_question") or ""),
        provenance=tuple(provenance),
    )
    return requirement.to_dict()


def bind_user_requirement_runtime(
    requirement: Mapping[str, Any],
    *,
    task_id: str,
    parent_task_id: str = "",
    dataset_version: str = "",
) -> Dict[str, Any]:
    current = _requirement_from_mapping(requirement)
    return replace(
        current,
        task_id=str(task_id or current.task_id),
        parent_task_id=str(parent_task_id or ""),
        dataset_version=str(dataset_version or current.dataset_version),
    ).to_dict()


def project_requirement_to_task_spec(
    requirement: Mapping[str, Any],
    task_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project the sole requirement source into the legacy executor schema."""

    result = dict(task_spec or {})
    mapping = {
        "outcomes": "outcomes",
        "exposures": "exposures",
        "predictors": "predictors",
        "candidate_features": "candidate_features",
        "forced_covariates": "forced_covariates",
        "excluded_features": "excluded_features",
        "group_by": "group_by",
        "requested_methods": "requested_methods",
        "requested_outputs": "required_outputs",
    }
    for source, target in mapping.items():
        if source in requirement:
            result[target] = list(requirement.get(source) or [])
    for key in (
        "candidate_scope",
        "filters",
        "top_k",
        "needs_clarification",
        "clarification_question",
    ):
        if key in requirement:
            result[key] = requirement.get(key)
    if requirement.get("task"):
        result["goal"] = str(requirement.get("task") or "")
    result["requirement_hash"] = str(
        requirement.get("requirement_hash") or ""
    )
    result["requirement_projection"] = True
    return result


def binding_values(
    requirement: Mapping[str, Any],
    field: str,
) -> List[Any]:
    return [
        item.get("value")
        for item in list(requirement.get("provenance") or [])
        if isinstance(item, Mapping)
        and str(item.get("field") or "") == field
        and str(item.get("certainty") or "") == "explicit"
    ]


def requirement_execution_gate(
    requirement: Mapping[str, Any],
    tool_requests: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Dict[str, Any]:
    """Enforce only the hard safety and explicit-requirement boundaries."""

    allowed_columns = {str(item) for item in columns}
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []
    explicit_field_groups = (
        "outcomes",
        "exposures",
        "predictors",
        "candidate_features",
        "forced_covariates",
        "excluded_features",
        "group_by",
    )
    explicit_columns = {
        str(value)
        for field_name in explicit_field_groups
        for value in binding_values(requirement, field_name)
        if str(value)
    }
    for value in binding_values(requirement, "filters"):
        if isinstance(value, Mapping) and value.get("field"):
            explicit_columns.add(str(value["field"]))
    missing_columns = sorted(explicit_columns - allowed_columns)
    if missing_columns:
        errors.append(
            {
                "code": "explicit_column_unavailable",
                "message": (
                    "Explicitly requested columns are unavailable: "
                    + ", ".join(missing_columns)
                ),
                "columns": missing_columns,
            }
        )

    outcomes = set(str(item) for item in requirement.get("outcomes") or [])
    exposures = set(str(item) for item in requirement.get("exposures") or [])
    forced = set(
        str(item) for item in requirement.get("forced_covariates") or []
    )
    candidates = set(
        str(item) for item in requirement.get("candidate_features") or []
    )
    contradictory = sorted(outcomes.intersection(exposures.union(forced)))
    explicit_outcomes = {
        str(value) for value in binding_values(requirement, "outcomes")
    }
    explicit_exposures = {
        str(value) for value in binding_values(requirement, "exposures")
    }
    explicit_forced = {
        str(value) for value in binding_values(requirement, "forced_covariates")
    }
    hard_contradictory = sorted(
        column
        for column in contradictory
        if column in explicit_outcomes
        and column in explicit_exposures.union(explicit_forced)
    )
    if hard_contradictory:
        errors.append(
            {
                "code": "explicit_role_conflict",
                "message": (
                    "A requested outcome also has an incompatible model role: "
                    + ", ".join(hard_contradictory)
                ),
                "columns": hard_contradictory,
            }
        )
    inferred_contradictory = sorted(
        set(contradictory) - set(hard_contradictory)
    )
    if inferred_contradictory:
        warnings.append(
            "Ignored non-binding inferred role overlap: "
            + ", ".join(inferred_contradictory)
        )
    leaked_candidates = sorted(outcomes.intersection(candidates))
    if leaked_candidates:
        errors.append(
            {
                "code": "explicit_target_leakage",
                "message": (
                    "The outcome cannot be a candidate feature: "
                    + ", ".join(leaked_candidates)
                ),
                "columns": leaked_candidates,
            }
        )

    method_aliases = {
        "lasso_logistic": "lasso",
        "logistic": "logistic_regression",
        "adjusted_logistic_association": "logistic_regression",
        "binary_model_comparison": "logistic_regression",
        "kmeans_clustering": "kmeans",
    }
    actual_methods: set[str] = set()
    unsafe_keys = {
        "shell",
        "command",
        "system_command",
        "subprocess",
        "remote_code",
    }
    for request in tool_requests:
        capability = str(request.get("tool") or "").strip().lower()
        from .tool_registry import get_tool_manifest
        manifest = get_tool_manifest(capability)
        resource = tool_resource(capability, bool(manifest and manifest.network_access))
        if resource and resource in set(requirement.get("prohibited_resources") or []):
            errors.append({"code": "explicit_resource_prohibited", "tool": capability,
                           "resource": resource, "message": "The user explicitly prohibited this resource: " + resource})
        if capability:
            actual_methods.add(method_aliases.get(capability, capability))
        for method in list(request.get("methods") or []):
            normalized = str(method).strip().lower()
            if normalized:
                actual_methods.add(method_aliases.get(normalized, normalized))
        dangerous = sorted(
            key
            for key in unsafe_keys
            if request.get(key) not in (None, "", False, [])
        )
        if dangerous:
            errors.append(
                {
                    "code": "unsafe_execution_request",
                    "message": (
                        "The execution plan requests prohibited capabilities: "
                        + ", ".join(dangerous)
                    ),
                    "keys": dangerous,
                }
            )
    explicit_methods = {
        str(value).strip().lower()
        for value in binding_values(requirement, "requested_methods")
        if str(value).strip()
    }
    normalized_explicit_methods = {
        method_aliases.get(method, method) for method in explicit_methods
    }
    missing_methods = sorted(normalized_explicit_methods - actual_methods)
    if missing_methods:
        errors.append(
            {
                "code": "explicit_method_not_executed",
                "message": (
                    "The execution plan does not contain the explicitly "
                    "requested method: "
                    + ", ".join(missing_methods)
                ),
                "methods": missing_methods,
            }
        )

    if not tool_requests and (
        requirement.get("outcomes")
        or requirement.get("filters")
        or requirement.get("requested_methods")
    ):
        warnings.append(
            "No executable node was produced; rebuild the minimum plan from "
            "UserRequirement before asking the user."
        )
    return {
        "allow_execution": not errors,
        "decision": "block" if errors else "allow",
        "hard_errors": errors,
        "warnings": warnings,
        "explicit_columns": sorted(explicit_columns),
        "explicit_methods": sorted(normalized_explicit_methods),
        "actual_methods": sorted(actual_methods),
        "authority": "minimal_hard_safety_gate",
        "policy": [
            "explicit_column_and_role_consistency",
            "operation_type_compatibility_at_executor",
            "target_leakage_prevention",
            "dangerous_permission_prevention",
            "explicit_method_fidelity",
        ],
    }
