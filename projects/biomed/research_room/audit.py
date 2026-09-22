from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .reconciliation import validate_analysis_graph
from .user_requirements import (
    binding_values,
    requirement_execution_gate,
    verify_user_requirement,
)


NON_CORE_OUTPUTS = {
    "figure",
    "download",
    "pmid_list",
    "literature_evidence",
    "evidence_limitation_summary",
    "methodology_summary",
    "external_sources",
}


def deterministic_requirement_audit(
    user_requirement: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    columns: Sequence[str],
) -> Dict[str, Any]:
    """Audit actual execution against the only user-facing requirement truth."""

    steps = [
        dict(item)
        for item in list(execution_manifest.get("steps") or [])
        if isinstance(item, Mapping)
    ]
    requests = [dict(step.get("request") or {}) for step in steps]
    gate = requirement_execution_gate(
        user_requirement,
        requests,
        columns,
    )
    hard_errors = list(gate.get("hard_errors") or [])
    warnings: List[str] = []
    if not verify_user_requirement(user_requirement):
        hard_errors.append(
            {
                "code": "invalid_user_requirement_hash",
                "message": "The immutable UserRequirement hash is invalid.",
            }
        )
    expected_hash = str(user_requirement.get("requirement_hash") or "")
    manifest_hash = str(execution_manifest.get("requirement_hash") or "")
    if manifest_hash and manifest_hash != expected_hash:
        hard_errors.append(
            {
                "code": "execution_requirement_mismatch",
                "message": (
                    "ExecutionManifest is linked to a different "
                    "UserRequirement."
                ),
            }
        )

    completed_outputs = {
        str(item)
        for item in list(execution_manifest.get("completed_outputs") or [])
        if str(item)
    }
    if not completed_outputs:
        completed_outputs = {
            str(item)
            for step in steps
            if str(step.get("status") or "") == "completed"
            for item in list(
                dict(step.get("evidence_manifest") or {}).get(
                    "declared_outputs"
                )
                or []
            )
            if str(item)
        }
    explicit_outputs = {
        str(item)
        for item in binding_values(
            user_requirement,
            "requested_outputs",
        )
        if str(item)
    }
    evidence_requested = bool(user_requirement.get("evidence_requested"))
    analytic_expected = explicit_outputs - NON_CORE_OUTPUTS
    missing_core_outputs = sorted(analytic_expected - completed_outputs)
    missing_non_core_outputs = sorted(
        (explicit_outputs.intersection(NON_CORE_OUTPUTS)) - completed_outputs
    )
    if missing_core_outputs:
        warnings.append(
            "Core requested outputs are incomplete: "
            + ", ".join(missing_core_outputs)
        )
    if missing_non_core_outputs:
        warnings.append(
            "Non-core outputs are incomplete and must be reported as a "
            "branch-local limitation: "
            + ", ".join(missing_non_core_outputs)
        )
    if evidence_requested:
        warnings.append(
            "Literature evidence is audited by the independent evidence "
            "branch and is not a prerequisite for analytic execution."
        )

    selected_features = {
        str(item)
        for step in steps
        if str(step.get("status") or "") == "completed"
        for item in list(
            dict(step.get("evidence_manifest") or {}).get("result_terms")
            or []
        )
        if str(item)
    }
    outcomes = {
        str(item) for item in user_requirement.get("outcomes") or []
    }
    leaked = sorted(outcomes.intersection(selected_features))
    if leaked:
        hard_errors.append(
            {
                "code": "executed_target_leakage",
                "message": (
                    "The outcome was returned as a selected feature: "
                    + ", ".join(leaked)
                ),
                "columns": leaked,
            }
        )

    completed_steps = [
        step for step in steps if str(step.get("status") or "") == "completed"
    ]
    failed_steps = [
        step for step in steps if str(step.get("status") or "") == "failed"
    ]
    if failed_steps and completed_steps:
        warnings.append(
            "Some execution branches failed; completed branch artifacts remain "
            "valid and must not be discarded."
        )
    elif failed_steps and not completed_steps:
        warnings.append("No analytic execution branch completed.")

    decision = (
        "block"
        if hard_errors
        else ("warning" if warnings else "pass")
    )
    completion = (
        "unsafe"
        if hard_errors
        else (
            "failed" if failed_steps and not completed_steps
            else ("partial" if missing_core_outputs or failed_steps else "completed")
        )
    )
    return {
        "decision": decision,
        "completion": completion,
        "public_note": (
            "執行證據符合使用者需求。"
            if decision == "pass"
            else (
                (
                    "已產生的輸出予以保留，但部分分支或要求尚未完成。"
                    if completed_steps
                    else "本次沒有完成可用的分析結果；請依執行錯誤修正資料或需求後再執行。"
                )
                if decision == "warning"
                else "執行觸犯硬性安全或明確需求底線。"
            )
        ),
        "completed_requirements": sorted(completed_outputs),
        "missing_requirements": missing_core_outputs,
        "missing_non_core_outputs": missing_non_core_outputs,
        "methodological_warnings": warnings,
        "hard_errors": hard_errors,
        "execution_gate": gate,
        "authority": "immutable_requirement_execution_auditor",
        "requirement_hash": expected_hash,
        "manifest_hash": str(
            execution_manifest.get("manifest_hash") or ""
        ),
        "preserve_completed_artifacts": True,
    }


def deterministic_contract_audit(
    task_spec: Mapping[str, Any],
    graph: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    columns: Sequence[str],
) -> Dict[str, Any]:
    validation = validate_analysis_graph(task_spec, graph, columns)
    errors = list(validation.get("errors") or [])
    warnings = list(validation.get("warnings") or [])

    if str(execution_manifest.get("spec_hash") or "") != str(
        task_spec.get("spec_hash") or ""
    ):
        errors.append("ExecutionManifest 的 spec hash 與研究契約不一致。")
    if str(execution_manifest.get("graph_hash") or "") != str(
        graph.get("graph_hash") or ""
    ):
        errors.append("ExecutionManifest 的 graph hash 與執行圖不一致。")

    graph_nodes = {
        str(node.get("node_id") or ""): dict(node)
        for node in list(graph.get("nodes") or [])
    }
    manifest_steps = {
        str(step.get("node_id") or ""): dict(step)
        for step in list(execution_manifest.get("steps") or [])
    }

    for node_id, node in graph_nodes.items():
        if node_id not in manifest_steps:
            errors.append(f"必要分析節點未執行：{node_id}")
            continue
        step = manifest_steps[node_id]
        if step.get("status") != "completed":
            errors.append(
                f"分析節點未成功完成：{node_id} ({step.get('status')})"
            )
        if json.dumps(
            dict(step.get("request") or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ) != json.dumps(
            dict(node.get("request") or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ):
            errors.append(f"分析節點實際參數與核准契約不同：{node_id}")
        if step.get("status") == "completed" and not step.get("evidence_manifest"):
            warnings.append(f"分析節點缺少證據 manifest：{node_id}")
        if step.get("status") == "completed":
            declared_outputs = set(
                str(item)
                for item in list(
                    dict(step.get("evidence_manifest") or {}).get(
                        "declared_outputs"
                    )
                    or []
                )
            )
            required_node_outputs = set(
                str(item) for item in list(node.get("output_contract") or [])
            )
            if not required_node_outputs.issubset(declared_outputs):
                errors.append(f"分析節點未證明其輸出型別：{node_id}")

    required_exposures = set(
        str(item) for item in task_spec.get("exposures") or []
    )
    if required_exposures:
        actual_columns = {
            str(item)
            for step in manifest_steps.values()
            if step.get("status") == "completed"
            for item in list(
                dict(
                    dict(step.get("evidence_manifest") or {}).get(
                        "actual_input_contract"
                    )
                    or {}
                ).get("columns")
                or []
            )
        }
        missing_actual_exposures = sorted(required_exposures - actual_columns)
        if missing_actual_exposures:
            errors.append(
                "ExecutionManifest has no tool evidence that the requested "
                "exposure entered the fitted model: "
                + ", ".join(missing_actual_exposures)
            )

        inferential_outputs = {
            "odds_ratio",
            "confidence_interval",
            "p_value",
            "coefficient",
        }
        if inferential_outputs.intersection(
            str(item) for item in task_spec.get("required_outputs") or []
        ):
            result_terms = {
                str(item)
                for step in manifest_steps.values()
                if step.get("status") == "completed"
                for item in list(
                    dict(step.get("evidence_manifest") or {}).get(
                        "result_terms"
                    )
                    or []
                )
            }
            missing_result_terms = sorted(required_exposures - result_terms)
            if missing_result_terms:
                errors.append(
                    "The fitted result does not contain an estimate for the "
                    "requested exposure: "
                    + ", ".join(missing_result_terms)
                )

    if str(task_spec.get("goal") or "") == "feature_discovery":
        selected_features = [
            str(item)
            for step in manifest_steps.values()
            if step.get("status") == "completed"
            for item in list(
                dict(step.get("evidence_manifest") or {}).get(
                    "result_terms"
                )
                or []
            )
            if str(item)
        ]
        selected_features = list(dict.fromkeys(selected_features))
        forced_covariates = {
            str(item)
            for item in task_spec.get("forced_covariates") or []
            if str(item)
        }
        outcomes = {
            str(item)
            for item in task_spec.get("outcomes") or []
            if str(item)
        }
        invalid_selected = sorted(
            set(selected_features).intersection(
                forced_covariates.union(outcomes)
            )
        )
        if not selected_features:
            errors.append(
                "The feature-selection result contains no auditable selected "
                "features."
            )
        if invalid_selected:
            errors.append(
                "Forced covariates or outcomes were incorrectly reported as "
                "selected candidate features: "
                + ", ".join(invalid_selected)
            )
        requested_top_k = int(task_spec.get("top_k") or 0)
        if requested_top_k and len(selected_features) > requested_top_k:
            errors.append(
                "The feature-selection result exceeds the requested top-k "
                f"limit ({requested_top_k}): {len(selected_features)}."
            )
        if (
            requested_top_k
            and selected_features
            and len(selected_features) < requested_top_k
        ):
            warnings.append(
                "The validated selector returned fewer auditable features "
                f"than requested ({len(selected_features)}/{requested_top_k})."
            )

    unexpected = sorted(set(manifest_steps) - set(graph_nodes))
    if unexpected:
        errors.append("執行了未經 AnalysisGraph 核准的節點：" + ", ".join(unexpected))

    for output, providers in dict(graph.get("output_bindings") or {}).items():
        completed_providers = [
            str(node_id)
            for node_id in list(providers or [])
            if manifest_steps.get(str(node_id), {}).get("status") == "completed"
            and str(output)
            in set(
                str(item)
                for item in list(
                    dict(
                        manifest_steps.get(str(node_id), {}).get(
                            "evidence_manifest"
                        )
                        or {}
                    ).get("declared_outputs")
                    or []
                )
            )
        ]
        if not completed_providers:
            errors.append(f"必要輸出未由成功節點提供：{output}")

    decision = "block" if errors else ("warning" if warnings else "pass")
    return {
        "decision": decision,
        "public_note": (
            "實際執行與研究契約一致。"
            if decision == "pass"
            else (
                "執行結果存在可說明的限制，請連同警告解讀。"
                if decision == "warning"
                else "實際執行未完整符合研究契約，本次不得產生正式研究結論。"
            )
        ),
        "completed_requirements": (
            [
                "typed_task_spec",
                "analysis_graph",
                "execution_manifest",
                "typed_output_recomposition",
            ]
            if decision != "block"
            else []
        ),
        "missing_requirements": errors,
        "methodological_warnings": warnings,
        "authority": "deterministic_contract_auditor",
        "spec_hash": str(task_spec.get("spec_hash") or ""),
        "graph_hash": str(graph.get("graph_hash") or ""),
        "manifest_hash": str(execution_manifest.get("manifest_hash") or ""),
    }
