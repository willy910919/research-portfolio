from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .contracts import canonical_hash, utc_now
from .reconciliation import extract_deterministic_facts


_FIELD_CUES = {
    "outcomes": ("outcome", "target", "結果變項", "依變項", "結局"),
    "exposures": ("exposure", "暴露", "主要變項"),
    "forced_covariates": ("covariate", "共變項", "調整變項", "校正變項"),
    "predictors": ("predictor", "預測變項", "納入分析的欄位", "至少兩個"),
    "candidate_features": ("候選特徵", "候選變項", "candidate feature"),
    "group_by": ("group by", "分組欄位", "分組變項", "依哪個欄位分組"),
    "requested_methods": ("分析方法", "指定方法", "method"),
}


def infer_requested_fields(clarification_question: str) -> list[str]:
    text = str(clarification_question or "").strip().lower()
    return [
        field
        for field, cues in _FIELD_CUES.items()
        if any(cue.lower() in text for cue in cues)
    ]


def build_pending_clarification(
    *,
    run_id: str,
    original_question: str,
    clarification_question: str,
    plan: Mapping[str, Any],
    phase: str,
    event_id: int = 0,
) -> Dict[str, Any]:
    task_spec = dict(plan.get("task_spec") or {})
    requirement = dict(plan.get("user_requirement") or {})
    requested_fields = infer_requested_fields(clarification_question)
    if not requested_fields:
        if task_spec.get("goal") == "feature_discovery" and not task_spec.get(
            "outcomes"
        ):
            requested_fields = ["outcomes"]
        elif not task_spec.get("outcomes") and task_spec.get("task_family") in {
            "classification",
            "regression",
            "survival",
        }:
            requested_fields = ["outcomes"]

    payload = {
        "version": "1.0",
        "status": "waiting_input",
        "run_id": str(run_id or ""),
        "event_id": int(event_id or 0),
        "original_question": str(original_question or "").strip(),
        "clarification_question": str(clarification_question or "").strip(),
        "requested_fields": requested_fields,
        "user_requirement": requirement,
        "task_spec": task_spec,
        "analysis_graph": dict(plan.get("analysis_graph") or {}),
        "resume_phase": str(phase or "waiting_input"),
        "created_at": utc_now(),
    }
    payload["pending_id"] = "CLARIFY_" + canonical_hash(payload)[:16]
    return payload


def _resolved_values(
    answer: str,
    pending: Mapping[str, Any],
    columns: Sequence[str],
) -> Dict[str, list[Any]]:
    facts = extract_deterministic_facts(answer, columns)
    mentioned = list(facts.get("mentioned_columns") or [])
    resolved: Dict[str, list[Any]] = {}
    for field in list(pending.get("requested_fields") or []):
        values = list(facts.get(field) or [])
        if not values and field in {
            "outcomes",
            "exposures",
            "forced_covariates",
            "predictors",
            "candidate_features",
            "group_by",
        }:
            # The pending question supplies the role; a short answer may only
            # contain the concrete column name.
            values = mentioned
        if values:
            resolved[field] = values
    return resolved


def build_clarification_resume_text(
    pending: Mapping[str, Any],
    answer: str,
    columns: Sequence[str],
    *,
    fallback_original_question: str = "",
) -> str:
    original = str(
        pending.get("original_question") or fallback_original_question or ""
    ).strip()
    clarification = str(pending.get("clarification_question") or "").strip()
    answer_text = str(answer or "").strip()
    resolved = _resolved_values(answer_text, pending, columns)

    canonical_lines = []
    labels = {
        "outcomes": "已確認 outcome 欄位",
        "exposures": "已確認 exposure 欄位",
        "forced_covariates": "已確認調整變項",
        "predictors": "已確認預測欄位",
        "candidate_features": "已確認候選特徵",
        "group_by": "已確認分組欄位",
        "requested_methods": "已確認分析方法",
    }
    for field, values in resolved.items():
        if field == "outcomes" and len(values) == 1:
            canonical_lines.append(f"以 {values[0]} 為 outcome。")
        else:
            canonical_lines.append(
                f"{labels.get(field, field)}：{', '.join(map(str, values))}。"
            )

    sections = [original] if original else []
    if clarification:
        sections.append(f"待澄清事項：{clarification}")
    sections.append(f"使用者對上述待澄清事項的回答：{answer_text}")
    if canonical_lines:
        sections.append("結構化確認：" + " ".join(canonical_lines))
    return "\n\n".join(item for item in sections if item)


def resolved_pending_record(
    pending: Mapping[str, Any],
    *,
    answer: str,
    resolved_run_id: str,
) -> Dict[str, Any]:
    return {
        **dict(pending or {}),
        "status": "resolved",
        "answer": str(answer or "").strip(),
        "resolved_run_id": str(resolved_run_id or ""),
        "resolved_at": utc_now(),
    }
