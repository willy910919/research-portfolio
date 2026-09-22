from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from packaging.requirements import InvalidRequirement, Requirement

from .filtering import FILTER_OPERATORS


@dataclass(frozen=True)
class ToolManifest:
    """V2 authoritative description of one broker-visible tool.

    Compatibility fields remain available because resource allocation and old
    artifacts still read them, but the nested schemas below are the canonical
    tool-boundary contract.
    """

    id: str
    outputs: Tuple[str, ...]
    version: str = "1.0.0"
    methods: Tuple[str, ...] = ()
    complexity_level: int = 0
    queue_class: str = "statistics"
    branch: str = "analysis"
    risk: str = "low"
    guards: Tuple[str, ...] = ()
    timeout_sec: int = 60
    verification: str = "end_to_end_verified"
    executable: bool = True
    accepts: Tuple[str, ...] = ("tabular_data",)
    requires: Tuple[str, ...] = ()
    input_fields: Tuple[str, ...] = ()
    cpu_cores: int = 1
    memory_gb: float = 1.0
    network_access: bool = False
    data_scope: str = "session_dataset"
    manifest_version: str = "2.0"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    preconditions: Tuple[str, ...] = ()
    permissions: Dict[str, Any] = field(default_factory=dict)
    resource_profile_v2: Dict[str, Any] = field(default_factory=dict)
    error_policy: Dict[str, Any] = field(default_factory=dict)
    executor: str = ""
    capability_pack: str = "classical_statistics"
    package_requirements: Tuple[str, ...] = ()
    smoke_test: str = "manifest_contract"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["input_schema"] = dict(self.input_schema)
        result["output_schema"] = dict(self.output_schema)
        result["permissions"] = dict(self.permissions)
        result["resource_profile"] = dict(self.resource_profile_v2)
        result.pop("resource_profile_v2", None)
        result["error_policy"] = dict(self.error_policy)
        result["input_contract"] = {
            **dict(self.input_schema),
            "accepts": list(self.accepts),
            "requires": list(self.requires),
            "fields": list(self.input_fields),
        }
        result["output_contract"] = {
            **dict(self.output_schema),
            "outputs": list(self.outputs),
        }
        result["security"] = dict(self.permissions)
        return result


def _capability_pack(tool_id: str, branch: str) -> str:
    if tool_id in {
        "cox_survival",
        "kaplan_meier",
        "gee_longitudinal",
        "mixed_effects",
    }:
        return "survival_longitudinal"
    if tool_id in {
        "propensity_score_iptw",
        "propensity_score_matching",
        "propensity_balance_diagnostics",
    }:
        return "causal_inference"
    if tool_id in {
        "omics_normalization",
        "differential_expression",
        "multiple_testing_correction",
    }:
        return "omics"
    if tool_id in {"pubmed_connector_search", "europe_pmc_connector_search"}:
        return "evidence_research"
    if tool_id in {
        "binary_model_comparison",
        "regression_model_comparison",
        "multiclass_model_comparison",
        "pca",
        "kmeans_clustering",
    }:
        return "tabular_ml"
    if branch == "system":
        return "system_observability"
    if branch == "data":
        return "data_management"
    return "classical_statistics"


def _manifest(
    tool_id: str,
    outputs: Tuple[str, ...],
    *,
    methods: Tuple[str, ...] = (),
    level: int = 0,
    queue: str = "statistics",
    branch: str = "analysis",
    risk: str = "low",
    guards: Tuple[str, ...] = (),
    timeout: int = 60,
    accepts: Optional[Tuple[str, ...]] = None,
    requires: Tuple[str, ...] = (),
    input_fields: Tuple[str, ...] = (),
    cpu: int = 1,
    memory_gb: float = 1.0,
    network_access: bool = False,
    data_scope: str = "session_dataset",
    preconditions: Tuple[str, ...] = (),
    packages: Tuple[str, ...] = (),
    max_retries: int = 1,
    capability_pack: str = "",
) -> ToolManifest:
    if accepts is None:
        accepts = () if branch == "system" else ("tabular_data",)
    input_schema = {
        "type": "object",
        "accepts": list(accepts),
        "required_capabilities": list(requires),
        "properties": {
            field_name: {"type": "field_or_parameter"}
            for field_name in input_fields
        },
        "data_scope": data_scope,
        "additionalProperties": True,
    }
    if tool_id == "filter_group":
        input_schema["required"] = ["filters"]
        input_schema["properties"] = {
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["field", "operator"],
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {"enum": list(FILTER_OPERATORS)},
                        "value": {},
                    },
                    "additionalProperties": False,
                },
            },
            "logic": {"enum": ["and", "or"]},
            "group_by": {"type": "array", "items": {"type": "string"}},
        }
    output_schema = {
        "type": "object",
        "required": list(outputs),
        "properties": {
            output_name: {"type": "artifact_or_evidence"}
            for output_name in outputs
        },
    }
    permissions = {
        "network_access": bool(network_access),
        "filesystem": "artifact_store_only",
        "subprocess": False,
        "data_scope": data_scope,
        "risk": risk,
    }
    resource_profile = {
        "cpu_cores": max(1, int(cpu)),
        "memory_gb": max(0.1, float(memory_gb)),
        "timeout_sec": max(1, int(timeout)),
        "queue_class": queue,
    }
    error_policy = {
        "max_retries": max(0, int(max_retries)),
        "dependency_failure": "skip_node",
        "non_core_failure": "partial_completion",
        "core_failure": "fail_branch_preserve_artifacts",
    }
    return ToolManifest(
        id=tool_id,
        outputs=outputs,
        methods=methods,
        complexity_level=level,
        queue_class=queue,
        branch=branch,
        risk=risk,
        guards=guards,
        timeout_sec=timeout,
        accepts=accepts,
        requires=requires,
        input_fields=input_fields,
        cpu_cores=max(1, int(cpu)),
        memory_gb=max(0.1, float(memory_gb)),
        network_access=bool(network_access),
        data_scope=data_scope,
        input_schema=input_schema,
        output_schema=output_schema,
        preconditions=tuple(preconditions or requires),
        permissions=permissions,
        resource_profile_v2=resource_profile,
        error_policy=error_policy,
        executor=f"broker:{tool_id}",
        capability_pack=capability_pack or _capability_pack(tool_id, branch),
        package_requirements=packages,
    )


_MANIFESTS = (
    _manifest("model_status", ("table",), branch="system"),
    _manifest("meeting_status", ("table",), branch="system"),
    _manifest("package_status", ("table",), branch="system"),
    _manifest("dataset_summary", ("summary_statistics", "table"), branch="data"),
    _manifest("dataset_profile", ("summary_statistics", "table"), branch="data"),
    _manifest("describe", ("summary_statistics", "count", "mean", "table"), branch="data"),
    _manifest("frequency", ("count", "percentage", "table"), branch="data"),
    _manifest(
        "filter_group",
        ("count", "percentage", "table", "download"),
        branch="data",
        guards=("typed_filter_operators", "group_role_separation"),
    ),
    _manifest("scatter", ("figure",), branch="visualization"),
    _manifest("histogram", ("figure",), branch="visualization"),
    _manifest("heatmap", ("figure",), branch="visualization"),
    _manifest("visualization", ("figure",), branch="visualization"),
    _manifest("correlation", ("correlation", "table"), branch="analysis"),
    _manifest(
        "linear_regression",
        ("coefficient", "confidence_interval", "p_value", "r_squared", "sample_size", "intercept", "table"),
        methods=("linear_regression",),
        level=1,
        risk="standard",
        guards=("numeric_outcome", "explicit_role_binding"),
    ),
    _manifest(
        "logistic_regression",
        ("coefficient", "confidence_interval", "p_value", "odds_ratio", "sample_size", "intercept", "table"),
        methods=("logistic_regression",),
        level=1,
        risk="standard",
        guards=("binary_outcome", "explicit_role_binding"),
    ),
    _manifest(
        "adjusted_logistic_association",
        ("coefficient", "confidence_interval", "p_value", "odds_ratio", "table", "download"),
        methods=("logistic_regression",),
        level=1,
        risk="standard",
        guards=("binary_outcome", "exposure_covariate_separation"),
    ),
    _manifest(
        "chi_square",
        ("p_value", "effect_size", "table"),
        methods=("chi_square",),
        level=1,
        risk="standard",
        guards=("categorical_inputs",),
    ),
    _manifest(
        "pca",
        ("components", "explained_variance", "loadings", "figure", "table"),
        methods=("pca",),
        level=1,
        queue="ml",
        risk="standard",
        guards=("numeric_matrix",),
        timeout=180,
    ),
    _manifest(
        "binary_model_comparison",
        ("roc_auc", "pr_auc", "calibration", "selected_features", "table", "figure", "download"),
        methods=("lasso", "logistic_regression", "random_forest", "gradient_boosting"),
        level=2,
        queue="ml",
        risk="high",
        guards=(
            "train_test_split",
            "selection_inside_cv",
            "no_target_leakage",
            "untouched_holdout",
        ),
        requires=("binary_outcome", "train_split", "numeric_or_encoded_matrix"),
        input_fields=("outcome", "features", "methods", "top_k"),
        cpu=4,
        memory_gb=8.0,
        timeout=600,
    ),
    _manifest(
        "regression_model_comparison",
        ("rmse", "mae", "r_squared", "table", "figure", "download"),
        methods=("linear_regression", "ridge", "random_forest", "gradient_boosting"),
        level=2,
        queue="ml",
        risk="high",
        guards=("train_test_split", "no_target_leakage", "untouched_holdout"),
        requires=("continuous_outcome", "train_split", "numeric_or_encoded_matrix"),
        input_fields=("outcome", "features", "methods"),
        cpu=4,
        memory_gb=8.0,
        timeout=600,
    ),
    _manifest(
        "multiclass_model_comparison",
        ("accuracy", "f1", "table", "figure", "download"),
        methods=("logistic_regression", "random_forest", "gradient_boosting"),
        level=2,
        queue="ml",
        risk="high",
        guards=("train_test_split", "no_target_leakage", "untouched_holdout"),
        requires=("multiclass_outcome", "train_split", "numeric_or_encoded_matrix"),
        input_fields=("outcome", "features", "methods"),
        cpu=4,
        memory_gb=8.0,
        timeout=600,
    ),
    _manifest(
        "kmeans_clustering",
        (
            "clusters",
            "cluster_assignments",
            "group_descriptive_stats",
            "group_means",
            "group_distributions",
            "silhouette",
            "table",
            "figure",
            "download",
        ),
        methods=("kmeans", "describe"),
        level=1,
        queue="ml",
        risk="standard",
        guards=("numeric_matrix",),
        timeout=300,
    ),
    _manifest(
        "cox_survival",
        ("hazard_ratio", "confidence_interval", "p_value", "table", "download"),
        methods=("cox",),
        level=1,
        queue="ml",
        risk="standard",
        guards=("duration_event_contract", "explicit_role_binding"),
        timeout=300,
    ),
    _manifest(
        "kaplan_meier",
        ("survival_curve", "median_survival", "table"),
        methods=("kaplan_meier",),
        level=1,
        queue="statistics",
        risk="standard",
        guards=("duration_event_contract", "nonnegative_duration"),
        requires=("duration", "binary_event"),
        input_fields=("duration", "event", "group"),
        packages=("numpy", "pandas"),
        timeout=180,
        capability_pack="survival_longitudinal",
    ),
    _manifest(
        "gee_longitudinal",
        ("coefficient", "confidence_interval", "p_value", "table"),
        methods=("gee",),
        level=2,
        queue="statistics",
        risk="high",
        guards=("longitudinal_subject_contract", "explicit_role_binding"),
        requires=("repeated_measurements", "subject_identifier"),
        input_fields=("outcome", "subject", "predictors", "family", "covariance"),
        packages=("numpy", "pandas", "statsmodels"),
        cpu=2,
        memory_gb=4.0,
        timeout=300,
        capability_pack="survival_longitudinal",
    ),
    _manifest(
        "mixed_effects",
        ("coefficient", "confidence_interval", "p_value", "table"),
        methods=("mixed_effects", "mixed_model"),
        level=2,
        queue="statistics",
        risk="high",
        guards=("longitudinal_subject_contract", "explicit_role_binding"),
        requires=("continuous_outcome", "subject_identifier"),
        input_fields=("outcome", "subject", "predictors", "reml"),
        packages=("numpy", "pandas", "statsmodels"),
        cpu=2,
        memory_gb=4.0,
        timeout=300,
        capability_pack="survival_longitudinal",
    ),
    _manifest(
        "propensity_score_iptw",
        ("effect_estimate", "balance_diagnostics", "weights", "table"),
        methods=("iptw", "propensity_score"),
        level=2,
        queue="ml",
        risk="high",
        guards=("binary_treatment", "positivity_check", "explicit_estimand"),
        requires=("binary_treatment", "outcome", "baseline_covariates"),
        input_fields=("treatment", "outcome", "covariates", "estimand", "trim"),
        packages=("numpy", "pandas", "scikit-learn"),
        cpu=2,
        memory_gb=4.0,
        timeout=300,
        capability_pack="causal_inference",
    ),
    _manifest(
        "propensity_score_matching",
        ("effect_estimate", "matched_pairs", "balance_diagnostics", "table"),
        methods=("propensity_score_matching", "matching"),
        level=2,
        queue="ml",
        risk="high",
        guards=("binary_treatment", "common_support", "explicit_estimand"),
        requires=("binary_treatment", "outcome", "baseline_covariates"),
        input_fields=("treatment", "outcome", "covariates", "caliper"),
        packages=("numpy", "pandas", "scikit-learn"),
        cpu=2,
        memory_gb=4.0,
        timeout=300,
        capability_pack="causal_inference",
    ),
    _manifest(
        "propensity_balance_diagnostics",
        ("balance_diagnostics", "weights", "table"),
        methods=("standardized_mean_difference", "iptw"),
        level=2,
        queue="ml",
        risk="high",
        guards=("binary_treatment", "baseline_covariates", "positivity_check"),
        requires=("binary_treatment", "baseline_covariates"),
        input_fields=("treatment", "covariates"),
        packages=("numpy", "pandas", "scikit-learn"),
        cpu=2,
        memory_gb=4.0,
        timeout=300,
        capability_pack="causal_inference",
    ),
    _manifest(
        "omics_normalization",
        ("normalized_data", "quality_summary", "table", "download"),
        methods=("log2_cpm", "zscore", "median_center"),
        level=2,
        queue="ml",
        risk="standard",
        guards=("numeric_matrix", "nonnegative_counts_for_cpm"),
        requires=("omics_feature_matrix",),
        input_fields=("features", "method", "pseudocount"),
        packages=("numpy", "pandas"),
        cpu=2,
        memory_gb=8.0,
        timeout=300,
        capability_pack="omics",
    ),
    _manifest(
        "differential_expression",
        ("effect_size", "p_value", "fdr", "table"),
        methods=("welch_t_test", "benjamini_hochberg"),
        level=2,
        queue="ml",
        risk="high",
        guards=("binary_group", "multiple_testing_correction"),
        requires=("omics_feature_matrix", "binary_group"),
        input_fields=("features", "group", "alpha"),
        packages=("numpy", "pandas", "scipy"),
        cpu=2,
        memory_gb=8.0,
        timeout=300,
        capability_pack="omics",
    ),
    _manifest(
        "multiple_testing_correction",
        ("adjusted_p_value", "rejection", "table"),
        methods=("benjamini_hochberg",),
        level=1,
        queue="statistics",
        risk="standard",
        guards=("valid_probability_range",),
        requires=("p_values",),
        input_fields=("p_value_column", "p_values", "alpha"),
        packages=("numpy", "pandas"),
        timeout=120,
        capability_pack="omics",
    ),
    _manifest(
        "pubmed_connector_search",
        ("citations", "query_record", "table"),
        methods=("pubmed",),
        level=1,
        queue="research",
        branch="evidence",
        risk="standard",
        guards=("sanitized_query", "citation_database_response"),
        accepts=(),
        requires=("deidentified_query",),
        input_fields=("query", "limit"),
        packages=("requests", "pandas"),
        network_access=True,
        data_scope="no_patient_data",
        timeout=60,
        capability_pack="evidence_research",
    ),
    _manifest(
        "europe_pmc_connector_search",
        ("citations", "query_record", "table"),
        methods=("europe_pmc",),
        level=1,
        queue="research",
        branch="evidence",
        risk="standard",
        guards=("sanitized_query", "citation_database_response"),
        accepts=(),
        requires=("deidentified_query",),
        input_fields=("query", "limit"),
        packages=("requests", "pandas"),
        network_access=True,
        data_scope="no_patient_data",
        timeout=60,
        capability_pack="evidence_research",
    ),
)

TOOL_REGISTRY: Dict[str, ToolManifest] = {
    manifest.id: manifest for manifest in _MANIFESTS
}

_TOOL_AVAILABILITY_PROVIDER: Optional[Callable[[str], bool]] = None


def set_tool_availability_provider(
    provider: Optional[Callable[[str], bool]],
) -> None:
    """Bind runtime plugin state without coupling the registry to persistence."""

    global _TOOL_AVAILABILITY_PROVIDER
    _TOOL_AVAILABILITY_PROVIDER = provider


def tool_runtime_available(tool_id: str) -> bool:
    if _TOOL_AVAILABILITY_PROVIDER is None:
        return True
    try:
        return bool(_TOOL_AVAILABILITY_PROVIDER(str(tool_id or "")))
    except Exception:
        return False


def validate_tool_registry(
    *,
    executable_bindings: Optional[Sequence[str]] = None,
    check_packages: bool = False,
) -> Dict[str, Any]:
    """Validate manifests before a planner is allowed to advertise tools."""

    errors = []
    warnings = []
    bindings = set(executable_bindings or ())
    seen = set()
    for manifest in TOOL_REGISTRY.values():
        if manifest.id in seen:
            errors.append({"tool": manifest.id, "code": "duplicate_tool_id"})
        seen.add(manifest.id)
        if manifest.manifest_version != "2.0":
            errors.append({"tool": manifest.id, "code": "manifest_not_v2"})
        if not manifest.executor.startswith("broker:"):
            errors.append({"tool": manifest.id, "code": "invalid_executor"})
        if not manifest.input_schema or not manifest.output_schema:
            errors.append({"tool": manifest.id, "code": "missing_schema"})
        if not manifest.resource_profile_v2 or not manifest.error_policy:
            errors.append({"tool": manifest.id, "code": "missing_runtime_policy"})
        if bindings and manifest.executable and manifest.id not in bindings:
            errors.append({"tool": manifest.id, "code": "executor_binding_missing"})
        if check_packages:
            for package in manifest.package_requirements:
                try:
                    package_name = Requirement(package).name
                except InvalidRequirement:
                    package_name = str(package).split("=", 1)[0].strip()
                module_name = {
                    "scikit-learn": "sklearn",
                }.get(package_name.lower(), package_name.replace("-", "_"))
                if importlib.util.find_spec(module_name) is None:
                    errors.append(
                        {
                            "tool": manifest.id,
                            "code": "required_package_missing",
                            "package": package,
                        }
                    )
        if manifest.verification != "end_to_end_verified":
            warnings.append(
                {
                    "tool": manifest.id,
                    "code": "not_end_to_end_verified",
                    "verification": manifest.verification,
                }
            )
    return {
        "status": "ready" if not errors else "invalid",
        "manifest_version": "2.0",
        "tool_count": len(TOOL_REGISTRY),
        "errors": errors,
        "warnings": warnings,
        "capability_packs": sorted(
            {manifest.capability_pack for manifest in TOOL_REGISTRY.values()}
        ),
    }


def capability_gap(tool_id: str) -> Dict[str, Any]:
    manifest = get_tool_manifest(tool_id)
    if manifest is None:
        return {
            "status": "capability_gap",
            "tool_id": str(tool_id or ""),
            "reason": "tool_not_registered",
            "may_use_external_discovery": True,
        }
    if not manifest.executable:
        return {
            "status": "capability_gap",
            "tool_id": manifest.id,
            "reason": "tool_not_executable",
            "verification": manifest.verification,
            "may_use_external_discovery": True,
        }
    if not tool_runtime_available(manifest.id):
        return {
            "status": "capability_gap",
            "tool_id": manifest.id,
            "reason": "plugin_disabled_or_dependencies_missing",
            "capability_pack": manifest.capability_pack,
            "may_use_external_discovery": False,
        }
    return {"status": "available", "tool_id": manifest.id}


def get_tool_manifest(tool_id: str) -> Optional[ToolManifest]:
    return TOOL_REGISTRY.get(str(tool_id or "").strip())


def tool_manifest_dict(tool_id: str) -> Dict[str, Any]:
    manifest = get_tool_manifest(tool_id)
    return manifest.to_dict() if manifest else {}


def tool_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    return {
        tool_id: manifest.to_dict()
        for tool_id, manifest in sorted(TOOL_REGISTRY.items())
    }


def executable_tool_ids() -> set[str]:
    return {
        tool_id
        for tool_id, manifest in TOOL_REGISTRY.items()
        if manifest.executable
    }


def low_risk_tool_ids() -> set[str]:
    return {
        tool_id
        for tool_id, manifest in TOOL_REGISTRY.items()
        if manifest.executable
        and manifest.complexity_level == 0
        and manifest.risk == "low"
    }


def tool_outputs() -> Dict[str, set[str]]:
    return {
        tool_id: set(manifest.outputs)
        for tool_id, manifest in TOOL_REGISTRY.items()
    }


def tool_methods() -> Dict[str, set[str]]:
    return {
        tool_id: set(manifest.methods)
        for tool_id, manifest in TOOL_REGISTRY.items()
        if manifest.methods
    }


def manifest_for_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable broker contract for a normalized request."""

    return tool_manifest_dict(str(request.get("tool") or ""))


def resolve_tool_requests(
    requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve requests against the executable registry without rewriting them.

    The broker is deliberately narrow: it confirms whether a requested tool is
    registered and executable. Research intent and parameter choices remain the
    Lead Agent's responsibility and are preserved verbatim for the executor.
    """

    accepted = []
    rejected = []
    manifests: Dict[str, Dict[str, Any]] = {}
    for index, raw_request in enumerate(requests):
        if not isinstance(raw_request, Mapping):
            rejected.append(
                {
                    "index": index,
                    "tool": "",
                    "reason": "invalid_request_shape",
                }
            )
            continue
        request = dict(raw_request)
        tool_id = str(request.get("tool") or "").strip()
        manifest = get_tool_manifest(tool_id)
        if manifest is None:
            rejected.append(
                {
                    "index": index,
                    "tool": tool_id,
                    "reason": "tool_not_registered",
                    "capability_gap": capability_gap(tool_id),
                }
            )
            continue
        if not manifest.executable:
            rejected.append(
                {
                    "index": index,
                    "tool": tool_id,
                    "reason": "tool_not_executable",
                    "capability_gap": capability_gap(tool_id),
                }
            )
            continue
        if not tool_runtime_available(tool_id):
            rejected.append(
                {
                    "index": index,
                    "tool": tool_id,
                    "reason": "plugin_disabled_or_dependencies_missing",
                    "capability_gap": capability_gap(tool_id),
                }
            )
            continue
        accepted.append(request)
        manifests[tool_id] = manifest.to_dict()
    return {
        "accepted": accepted,
        "rejected": rejected,
        "complete": not rejected,
        "manifests": manifests,
        "authority": "tool_broker_v2",
    }
