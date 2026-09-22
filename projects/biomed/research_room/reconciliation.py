from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .contracts import AnalysisGraph, TaskNode, TypedResearchTaskSpec, canonical_hash
from .filtering import (
    extract_missing_filter_intents,
    normalize_filter_condition,
    normalize_filter_conditions,
    protected_missing_fields,
)
from .user_requirements import (
    build_user_requirement,
    project_requirement_to_task_spec,
)
from .resource_intent import LITERATURE_OUTPUTS, enforce_resource_constraints, parse_resource_intents
from .tool_registry import (
    executable_tool_ids,
    get_tool_manifest,
    tool_methods,
    tool_outputs,
)


ANALYTIC_TOOLS = executable_tool_ids().difference(
    {"model_status", "meeting_status", "package_status"}
)

METHOD_ALIASES = {
    "restricted cubic spline logistic regression": "restricted_cubic_spline_logistic",
    "restricted cubic spline": "restricted_cubic_spline_logistic",
    "lasso": "lasso",
    "lasso_logistic": "lasso",
    "logistic": "logistic_regression",
    "logistic regression": "logistic_regression",
    "邏輯斯迴歸": "logistic_regression",
    "邏輯迴歸": "logistic_regression",
    "線性迴歸": "linear_regression",
    "linear regression": "linear_regression",
    "random forest": "random_forest",
    "random_forest": "random_forest",
    "隨機森林": "random_forest",
    "gradient boosting": "gradient_boosting",
    "xgboost": "xgboost",
    "pca": "pca",
    "主成分分析": "pca",
    "k-means": "kmeans",
    "kmeans": "kmeans",
    "kmeans_clustering": "kmeans",
    "cox": "cox",
    "shap": "shap",
    "smote": "smote",
}

OUTPUT_ALIASES = {
    "counts": "count",
    "frequencies": "count",
    "frequency": "count",
    "percents": "percentage",
    "percent": "percentage",
    "percentages": "percentage",
    "proportions": "percentage",
    "proportion": "percentage",
    "cluster_assignment": "cluster_assignments",
    "group_descriptive_statistics": "group_descriptive_stats",
    "group_statistics": "group_descriptive_stats",
    "group_profiles": "group_descriptive_stats",
    "explained_variances": "explained_variance",
    "feature_loadings": "loadings",
    "odds_ratios": "odds_ratio",
    "confidence_intervals": "confidence_interval",
    "p_values": "p_value",
    "coefficients": "coefficient",
}

TOOL_METHODS = tool_methods()
TOOL_OUTPUTS = tool_outputs()

CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

ENGLISH_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _unique(values: Iterable[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def _valid_columns(values: Iterable[Any], columns: Sequence[str]) -> List[str]:
    allowed = set(str(column) for column in columns)
    return _unique(str(value) for value in values if str(value) in allowed)


def _normalize_method(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return METHOD_ALIASES.get(text, text)


def _normalize_output(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = OUTPUT_ALIASES.get(text, text)
    if "_by_" in normalized:
        base = normalized.split("_by_", 1)[0]
        grouped_output = OUTPUT_ALIASES.get(base, base)
        if grouped_output in {"count", "percentage"}:
            return grouped_output
    return normalized


RESEARCH_SYNTHESIS_OUTPUTS = {
    "implementation_steps",
    "methodology_overview",
    "recommended_software_packages",
    "methodology_evidence",
    "implementation_sources",
    "source_assessment",
    "pmid_list",
    "citation_list",
    "literature_evidence",
    "evidence_limitation_summary",
}


VERIFIABLE_CONTEXT_MARKER = "[可驗證的對話脈絡]"


def _authoritative_user_text(question: str) -> str:
    """Return only the current utterance for deterministic fact locking.

    Reply excerpts and prior results remain available to the Research Agent for
    reference resolution, but they are not authoritative current instructions.
    """

    return str(question or "").split(VERIFIABLE_CONTEXT_MARKER, 1)[0].strip()


def _research_only_contract(task_spec: Mapping[str, Any]) -> bool:
    facts = dict(task_spec.get("explicit_facts") or {})
    resources = dict(facts.get("explicit_resource_requests") or {})
    outputs = {
        _normalize_output(item)
        for item in list(task_spec.get("required_outputs") or [])
        if str(item).strip()
    }
    return bool(
        any(bool(value) for value in resources.values())
        and outputs
        and outputs.issubset(RESEARCH_SYNTHESIS_OUTPUTS)
    )


def _execution_required_outputs(task_spec: Mapping[str, Any]) -> List[str]:
    outputs = _unique(
        _normalize_output(item)
        for item in list(task_spec.get("required_outputs") or [])
        if str(item)
    )
    # Literature, citation and evidence-synthesis outputs are produced by the
    # evidence workflow, not by dataframe AnalysisGraph nodes. Mixed requests
    # such as "run LASSO and compare the selected features with PubMed" must
    # therefore validate the analytic and evidence branches independently.
    return [
        output
        for output in outputs
        if output not in RESEARCH_SYNTHESIS_OUTPUTS
    ]


def _mentioned_columns(question: str, columns: Sequence[str]) -> List[str]:
    text = str(question or "")
    mentioned: List[Tuple[int, int, str]] = []
    for column in sorted((str(item) for item in columns), key=len, reverse=True):
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            mentioned.append((match.start(), -len(column), column))
    return [item[2] for item in sorted(mentioned)]


def _columns_after_cue(
    question: str,
    columns: Sequence[str],
    cues: Sequence[str],
    *,
    max_chars: int = 100,
) -> List[str]:
    text = str(question or "")
    found: List[str] = []
    for cue in cues:
        for match in re.finditer(cue, text, flags=re.IGNORECASE):
            fragment = text[match.end() : match.end() + max_chars]
            fragment = re.split(
                r"[。；;\n]"
                r"|(?:之後|後再|並且|然後)"
                r"|後(?=\s*(?:，|,|以|再|檢驗|評估|分析|看|建立|預測))",
                fragment,
                maxsplit=1,
            )[0]
            found.extend(_mentioned_columns(fragment, columns))
    return _unique(found)


def _columns_before_cue(
    question: str,
    columns: Sequence[str],
    cues: Sequence[str],
    *,
    max_chars: int = 100,
) -> List[str]:
    text = str(question or "")
    found: List[str] = []
    for cue in cues:
        for match in re.finditer(cue, text, flags=re.IGNORECASE):
            fragment = text[max(0, match.start() - max_chars) : match.start()]
            fragment = re.split(r"[。；;，,\n]", fragment)[-1]
            found.extend(_mentioned_columns(fragment, columns))
    return _unique(found)


def _parse_top_k(question: str) -> int:
    english_number = "|".join(ENGLISH_NUMBERS)
    english_patterns = [
        rf"\btop\s+(\d{{1,3}}|{english_number})\b",
        rf"\b(\d{{1,3}}|{english_number})\b"
        rf"(?:\s+[A-Za-z-]+){{0,5}}\s+"
        rf"(?:candidate\s+)?(?:features?|variables?|predictors?)\b",
    ]
    for pattern in english_patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            return int(token) if token.isdigit() else ENGLISH_NUMBERS[token]
    patterns = [
        r"(?:前|top\s*)(\d{1,3})\s*(?:個|項)?(?:穩定|重要|候選|預測)*\s*(?:特徵|變項|變量)",
        r"(\d{1,3})\s*(?:個|項)(?:穩定|重要|候選|預測)*\s*(?:特徵|變項|變量)",
        r"([一二兩三四五六七八九十])\s*(?:個|項)(?:穩定|重要|候選|預測)*\s*(?:特徵|變項|變量)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            token = match.group(1)
            return int(token) if token.isdigit() else CHINESE_NUMBERS.get(token, 0)
    return 0


def _extract_filters(question: str, columns: Sequence[str]) -> List[Dict[str, Any]]:
    located_filters: List[Tuple[int, Dict[str, Any]]] = []
    for column in sorted((str(item) for item in columns), key=len, reverse=True):
        pattern = (
            rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])\s*"
            r"(>=|<=|==|=|!=|>|<)\s*"
            r"([-+]?\d+(?:\.\d+)?|true|false|yes|no)"
        )
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            raw_value = match.group(2)
            if re.fullmatch(r"[-+]?\d+", raw_value):
                value: Any = int(raw_value)
            elif re.fullmatch(r"[-+]?\d+\.\d+", raw_value):
                value = float(raw_value)
            else:
                value = raw_value.lower()
            located_filters.append(
                (
                    match.start(),
                    {
                        "field": column,
                        "operator": "==" if match.group(1) == "=" else match.group(1),
                        "value": value,
                    },
                )
            )
    ordinary_filters = [
        item[1] for item in sorted(located_filters, key=lambda item: item[0])
    ]
    missing_filters = extract_missing_filter_intents(question, columns)
    missing_fields = {str(item.get("field") or "") for item in missing_filters}
    return _unique(
        missing_filters
        + [
            item
            for item in ordinary_filters
            if str(item.get("field") or "") not in missing_fields
        ]
    )


def _extract_explicit_group_by(
    question: str,
    columns: Sequence[str],
) -> List[str]:
    """Extract only group roles stated next to a real dataframe column name.

    Longer-range cue matching is useful as a hint for the LLM, but it is not a
    verifiable fact and must never override the Research Agent's semantic role.
    """

    text = str(question or "")
    located: List[Tuple[int, str]] = []
    for column in sorted((str(item) for item in columns), key=len, reverse=True):
        escaped = re.escape(column)
        patterns = (
            rf"(?:按|依|按照|根據)\s*(?:欄位|變項)?\s*"
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])\s*"
            rf"(?:分組|分層|分類|區分)",
            rf"(?:分組|分層)\s*(?:欄位|變項)?\s*(?:為|是|:|：)?\s*"
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
            rf"(?:分別)?(?:去)?看\s*"
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                located.append((match.start(), column))
                break
    return _unique(item[1] for item in sorted(located))


def extract_deterministic_facts(
    question: str,
    columns: Sequence[str],
) -> Dict[str, Any]:
    text = _authoritative_user_text(question)
    resource_intents = parse_resource_intents(text)
    mentioned = _mentioned_columns(text, columns)
    heatmap_requested = bool(
        re.search(r"(?:熱力圖|heat\s*map|heatmap)", text, flags=re.IGNORECASE)
    )
    visualization_type = ""
    visualization_patterns = (
        ("heatmap", r"(?:熱力圖|heat\s*map|heatmap)"),
        ("boxplot", r"(?:箱型圖|盒鬚圖|box\s*plot|boxplot)"),
        ("violin", r"(?:小提琴圖|violin\s*plot|violin)"),
        ("scatter", r"(?:散佈圖|散點圖|scatter\s*plot|scatterplot)"),
        ("histogram", r"(?:直方圖|histogram)"),
        ("bar", r"(?:長條圖|柱狀圖|bar\s*chart)"),
        ("line", r"(?:折線圖|趨勢圖|line\s*plot)"),
    )
    for candidate, pattern in visualization_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            visualization_type = candidate
            break
    correlation_heatmap = heatmap_requested and bool(
        re.search(r"(?:相關(?:係數|矩陣)?|correlation)", text, flags=re.IGNORECASE)
    )
    visualization_only = bool(visualization_type) and not bool(
        re.search(
            r"(?:平均|中位數|標準差|人數|比例|檢定|迴歸|預測|"
            r"p[\s_-]*value|勝算比|信賴區間|模型比較|特徵選擇)",
            text,
            flags=re.IGNORECASE,
        )
    )
    outcomes: List[str] = []
    for column in mentioned:
        escaped = re.escape(column)
        outcome_patterns = [
            rf"(?:以|將)\s*{escaped}\s*(?:作為|當作|為)\s*(?:outcome|target|結果變項|依變項)",
            rf"{escaped}\s*(?:作為|當作|為)\s*(?:outcome|target|結果變項|依變項)",
            rf"(?:預測|预测)\s*{escaped}(?![A-Za-z0-9_])",
            rf"(?:outcome|target|結果變項|依變項|結局)(?:\s*欄位)?\s*"
            rf"(?:(?:[:：=]|(?:幫我)?(?:使用|用|設為|指定為|選擇|改成|改用|是|為))\s*)*"
            rf"{escaped}(?![A-Za-z0-9_])",
        ]
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in outcome_patterns):
            outcomes.append(column)
    for match in re.finditer(r"(?:分別\s*)?(?:預測|预测)\s*", text):
        fragment = text[match.end() : match.end() + 120]
        fragment = re.split(r"[。；;\n]|(?:說明|比較|並提供|並且提供)", fragment, maxsplit=1)[0]
        outcomes.extend(_mentioned_columns(fragment, columns))

    # A single schema column scoped directly by a feature-selection request is
    # the target of that request even when the user does not repeat the word
    # "outcome". This is schema-driven and intentionally does not guess when
    # several unassigned columns are present.
    feature_selection_requested = bool(
        re.search(
            r"(?:特徵選擇|變項選擇|候選特徵|重要特徵|預測特徵|"
            r"feature\s*selection)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if feature_selection_requested and not outcomes and len(mentioned) == 1:
        scoped_column = mentioned[0]
        escaped = re.escape(scoped_column)
        scoped_patterns = (
            rf"(?:針對|關於|為|替)\s*(?:欄位\s*)?{escaped}"
            rf"(?:\s*(?:進行|執行|做|來做|的|之))?\s*"
            rf"(?:特徵選擇|變項選擇|候選特徵|重要特徵|預測特徵)",
            rf"(?:特徵選擇|變項選擇|候選特徵|重要特徵|預測特徵)"
            rf".{0,24}(?:針對|關於|為|替)\s*(?:欄位\s*)?{escaped}",
        )
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in scoped_patterns
        ):
            outcomes.append(scoped_column)

    forced_covariates = _unique(
        _columns_after_cue(
            text,
            columns,
            [r"(?:調整|校正|控制|固定保留|強制保留)"],
        )
        + _columns_before_cue(
            text,
            columns,
            [r"(?:必須|需要|仍要|固定|強制)?保留為?(?:調整|校正|控制)?變項"],
            max_chars=55,
        )
    )
    inferred_group_by = _columns_after_cue(
        text,
        columns,
        [
            r"(?:按|依|根據|按照)\s*",
            r"(?:分組|分層)\s*(?:欄位|變項)?\s*",
            r"(?:分別)?(?:去)?看\s*",
        ],
        max_chars=45,
    )
    group_by = _extract_explicit_group_by(text, columns)
    # A generic association phrase names variables, but does not by itself
    # assign every mentioned variable to the exposure role.  The Lead Agent
    # owns that semantic decision.  Deterministic extraction only locks an
    # exposure when the user states the role next to a concrete column.
    exposures: List[str] = []
    for column in mentioned:
        escaped = re.escape(column)
        exposure_patterns = (
            rf"(?:以|將)?\s*{escaped}\s*(?:作為|當作|設為|為|as)\s*"
            rf"(?:exposure|暴露(?:變項|變數)?|主要變項|自變項)",
            rf"(?:exposure|暴露(?:變項|變數)?|主要變項|自變項)\s*"
            rf"(?:為|用|使用|[:：])?\s*{escaped}(?![A-Za-z0-9_])",
        )
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in exposure_patterns
        ):
            exposures.append(column)
    association_variables = (
        [
            item
            for item in mentioned
            if item not in forced_covariates and item not in group_by
        ]
        if re.search(
            r"(?:關聯|相關|association|effect|效果|影響)",
            text,
            flags=re.IGNORECASE,
        )
        else []
    )
    exposures = [
        item
        for item in exposures
        if item not in outcomes and item not in forced_covariates
    ]

    predictors: List[str] = []
    if re.search(r"(?:使用|利用|根據|以).{0,100}(?:預測|建立.{0,20}模型)", text):
        predictor_fragment = re.split(r"(?:預測|建立)", text, maxsplit=1)[0]
        predictors = [
            item
            for item in _mentioned_columns(predictor_fragment, columns)
            if item not in outcomes
        ]

    methods = []
    occupied_method_spans: List[Tuple[int, int]] = []
    for label, normalized in sorted(
        METHOD_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        for match in re.finditer(re.escape(label), text, flags=re.IGNORECASE):
            span = match.span()
            if any(
                span[0] < occupied_end and span[1] > occupied_start
                for occupied_start, occupied_end in occupied_method_spans
            ):
                continue
            occupied_method_spans.append(span)
            methods.append(normalized)

    method_selection_delegated = bool(
        re.search(
            r"(?:"
            r"(?:合適|適合|適當|合理).{0,24}(?:方法|分析|檢定|模型)"
            r"|(?:請|由|讓).{0,16}(?:系統|主持人|agent|研究團隊)"
            r".{0,20}(?:選擇|判斷|決定|建議).{0,16}(?:方法|分析|檢定|模型)"
            r"|(?:選擇|採用).{0,16}(?:合適|適合|適當|合理)"
            r")",
            text,
            flags=re.IGNORECASE,
        )
    )
    method_selection_authority = (
        "user_explicit"
        if methods
        else ("chair_delegated" if method_selection_delegated else "")
    )

    cluster_count = 0
    cluster_match = re.search(
        r"(?:(\d+)|([一二兩三四五六七八九十]))\s*(?:群|群組|clusters?)",
        text,
        flags=re.IGNORECASE,
    )
    if cluster_match:
        if cluster_match.group(1):
            cluster_count = int(cluster_match.group(1))
        else:
            cluster_count = CHINESE_NUMBERS.get(cluster_match.group(2), 0)

    outputs = []
    output_patterns = {
        "odds_ratio": r"(?:\bOR\b|odds\s*ratio|勝算比)",
        "confidence_interval": r"(?:95\s*%\s*CI|confidence\s*interval|信賴區間)",
        "p_value": r"(?:p[\s_-]*value|p值)",
        "selected_features": r"(?:特徵選擇|穩定.*特徵|候選特徵|重要特徵)",
        "shap": r"\bSHAP\b",
        "figure": r"(?:圖|plot|chart|視覺化)",
        "download": r"(?:下載|CSV|子資料集)",
        "correlation": r"(?:相關係數|相關矩陣|correlation)",
        "explained_variance": (
            r"(?:explained[\s_-]*variance|variance[\s_-]*(?:ratio|explained)|"
            r"解釋變異(?:比例)?|解釋方差(?:比例)?)"
        ),
        "loadings": r"(?:\bloadings?\b|主成分負荷|成分負荷|負荷量)",
    }
    for output, pattern in output_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            outputs.append(output)

    # These are concrete requested results, not merely the planner's choice of
    # descriptive implementation. Keep model sample-size semantics separate.
    summary_pattern = r"描述(?:性)?統計|敘述(?:性)?統計|\b(?:descriptive|summary)\s+statistics\b"
    descriptive_output_spans = {}
    if not methods or re.search(summary_pattern, text, flags=re.IGNORECASE):
        descriptive_patterns = {
            "summary_statistics": summary_pattern,
            "mean": r"平均(?:值|數)?|均數|\b(?:mean(?!\s+(?:squared|absolute))|average)\b",
            "count": r"人數|筆數|樣本數|\b(?:counts?|sample\s+(?:size|count)|row\s+count|number\s+of\s+(?:rows|samples|participants))\b",
            "percentage": r"百分比|百分率|比例|\b(?:percentages?|percents?|proportions?)\b",
        }
        for output, pattern in descriptive_patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                outputs.append(output)
                descriptive_output_spans[output] = match.group(0)

    candidate_scope = ""
    candidate_features: List[str] = []
    if re.search(r"(?:只使用|僅使用|利用|使用).{0,160}(?:特徵|變項|變量|預測)", text):
        candidate_features = [
            item
            for item in mentioned
            if item not in outcomes and item not in forced_covariates
        ]
        if candidate_features:
            candidate_scope = "explicit"
    if re.search(r"(?:自動|廣泛|所有合格).{0,30}(?:候選|特徵|變項|變量)", text):
        candidate_scope = "auto_discover"
        candidate_features = []

    return {
        "mentioned_columns": mentioned,
        "outcomes": _unique(outcomes),
        "exposures": _unique(exposures),
        "predictors": _unique(predictors),
        "forced_covariates": _unique(forced_covariates),
        "candidate_features": _unique(candidate_features),
        "candidate_scope": candidate_scope,
        "group_by": _unique(group_by),
        "role_inferences": {
            "group_by": [
                item for item in _unique(inferred_group_by) if item not in group_by
            ],
            "association_variables": _unique(association_variables),
        },
        "filters": _extract_filters(text, columns),
        "requested_methods": _unique(methods),
        "method_selection_delegated": method_selection_delegated,
        "method_selection_authority": method_selection_authority,
        "required_outputs": _unique(outputs),
        "required_output_spans": descriptive_output_spans,
        "top_k": _parse_top_k(text),
        "visualization_type": visualization_type,
        "visualization_columns": mentioned if visualization_type else [],
        "heatmap_kind": (
            "correlation"
            if correlation_heatmap
            else ("density" if heatmap_requested else "")
        ),
        "visualization_only": visualization_only,
        "explicit_resource_requests": {key: value["state"] == "requested" for key, value in resource_intents.items()},
        "resource_intents": resource_intents,
        "prohibited_resources": [key for key, value in resource_intents.items() if value["state"] == "prohibited"],
        "cluster_count": cluster_count,
        "authority": "deterministic_explicit_fact_extraction",
    }


def _apply_explicit_visualization_contract(
    requests: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Preserve an explicit built-in visualization without inventing extra work."""

    visualization_type = str(facts.get("visualization_type") or "")
    columns = list(facts.get("visualization_columns") or [])
    group_by = list(facts.get("group_by") or [])
    if not visualization_type or not columns:
        return [dict(item) for item in requests]

    visualization_request: Dict[str, Any]
    if visualization_type == "heatmap":
        if len(columns) < 2:
            return [dict(item) for item in requests]
        kind = str(facts.get("heatmap_kind") or "density")
        if kind == "correlation":
            visualization_request = {
                "tool": "heatmap",
                "kind": "correlation",
                "columns": columns,
            }
        else:
            visualization_request = {
                "tool": "heatmap",
                "kind": "density",
                "x": columns[0],
                "y": columns[1],
                "bins": 35,
            }
    elif visualization_type in {"scatter", "line"}:
        if len(columns) < 2:
            return [dict(item) for item in requests]
        visualization_request = {
            "tool": "visualization",
            "kind": visualization_type,
            "x": columns[0],
            "y": columns[1],
            "group": group_by[0] if group_by else "",
        }
    elif visualization_type == "histogram":
        visualization_request = {
            "tool": "visualization",
            "kind": "histogram",
            "x": columns[0],
            "group": group_by[0] if group_by else "",
        }
    elif visualization_type in {"boxplot", "violin"}:
        grouping = group_by[0] if group_by else ""
        measure = next(
            (column for column in columns if column != grouping),
            columns[-1],
        )
        visualization_request = {
            "tool": "visualization",
            "kind": visualization_type,
            "x": grouping,
            "y": measure,
            "group": "",
        }
    elif visualization_type == "bar":
        grouping = group_by[0] if group_by else columns[0]
        measure = next(
            (column for column in columns if column != grouping),
            "",
        )
        visualization_request = {
            "tool": "visualization",
            "kind": "bar",
            "x": grouping,
            "y": measure,
            "aggregation": "count" if not measure else "mean",
            "group": "",
        }
    else:
        return [dict(item) for item in requests]

    # An explicit chart type is an output fact. Alternative chart tools and
    # package inspection added by a planner are not necessary to produce it.
    removable = {
        "dataset_profile",
        "package_status",
        "scatter",
        "histogram",
        "heatmap",
        "visualization",
        "correlation",
    }
    if bool(facts.get("visualization_only")):
        removable.update(
            {
                "dataset_summary",
                "describe",
                "frequency",
                "filter_group",
            }
        )
    replaced = [
        dict(item)
        for item in requests
        if str(item.get("tool") or "") not in removable
    ]
    replaced.insert(0, visualization_request)
    return replaced


def _roles_from_tool_requests(
    requests: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Dict[str, Any]:
    roles: Dict[str, List[Any]] = {
        "outcomes": [],
        "exposures": [],
        "predictors": [],
        "candidate_features": [],
        "forced_covariates": [],
        "group_by": [],
        "filters": [],
        "requested_methods": [],
    }
    top_k = 0
    for request in requests:
        tool = str(request.get("tool") or "")
        if request.get("outcome"):
            roles["outcomes"].append(request.get("outcome"))
        if tool == "adjusted_logistic_association":
            roles["exposures"].extend(request.get("exposures") or [])
            roles["forced_covariates"].extend(request.get("covariates") or [])
        elif tool in {"linear_regression", "logistic_regression"}:
            roles["predictors"].extend(request.get("predictors") or [])
        elif tool in {
            "binary_model_comparison",
            "regression_model_comparison",
            "multiclass_model_comparison",
        }:
            roles["candidate_features"].extend(request.get("features") or [])
            roles["requested_methods"].extend(request.get("methods") or [])
            top_k = max(top_k, int(request.get("top_k") or 0))
        elif tool == "filter_group":
            roles["filters"].extend(request.get("filters") or [])
        elif tool in {"pca", "kmeans_clustering"}:
            roles["predictors"].extend(
                request.get("columns") or request.get("features") or []
            )
        roles["group_by"].extend(_grouping_columns_from_request(request))
        if tool not in {
            "binary_model_comparison",
            "regression_model_comparison",
            "multiclass_model_comparison",
        }:
            roles["requested_methods"].extend(TOOL_METHODS.get(tool, set()))
    normalized: Dict[str, Any] = {
        key: _valid_columns(value, columns)
        if key
        in {
            "outcomes",
            "exposures",
            "predictors",
            "candidate_features",
            "forced_covariates",
            "group_by",
        }
        else _unique(value)
        for key, value in roles.items()
    }
    normalized["requested_methods"] = _unique(
        _normalize_method(item) for item in roles["requested_methods"]
    )
    normalized["top_k"] = top_k
    return normalized


def _grouping_columns_from_request(request: Mapping[str, Any]) -> List[str]:
    """Return the semantic grouping fields used by any supported tool schema."""

    grouping: List[Any] = list(request.get("group_by") or [])
    if request.get("group"):
        grouping.append(request.get("group"))

    tool = str(request.get("tool") or "")
    kind = str(request.get("kind") or "").strip().lower().replace("-", "_")
    if tool == "visualization" and kind in {
        "box",
        "boxplot",
        "violin",
        "bar",
    }:
        x = request.get("x")
        y = request.get("y") or request.get("column")
        if x and x != y:
            grouping.append(x)
    return _unique(str(item) for item in grouping if item not in (None, ""))


def _task_family_from_requests(requests: Sequence[Mapping[str, Any]]) -> str:
    tools = {str(item.get("tool") or "") for item in requests}
    if tools & {"linear_regression", "regression_model_comparison"}:
        return "regression"
    if tools & {
        "logistic_regression",
        "adjusted_logistic_association",
        "binary_model_comparison",
        "multiclass_model_comparison",
    }:
        return "classification_or_association"
    if "cox_survival" in tools:
        return "survival"
    if tools & {"pca", "kmeans_clustering"}:
        return "unsupervised"
    if "filter_group" in tools:
        return "filtered_descriptive"
    if tools & {
        "dataset_summary",
        "describe",
        "frequency",
        "histogram",
        "scatter",
        "heatmap",
        "visualization",
        "correlation",
    }:
        return "descriptive"
    return ""


def _list_from_spec(raw_spec: Mapping[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        value = raw_spec.get(key)
        if isinstance(value, list):
            return list(value)
        if value not in (None, ""):
            return [value]
    return []


def _merge_with_explicit(
    name: str,
    semantic_values: List[Any],
    explicit_values: List[Any],
    notes: List[str],
) -> List[Any]:
    if explicit_values:
        if semantic_values and set(map(str, semantic_values)) != set(
            map(str, explicit_values)
        ):
            notes.append(
                f"{name} 依使用者原文的可驗證事實修正；未採用衝突的語意推論。"
            )
        return _unique(explicit_values)
    return _unique(semantic_values)


def _repair_requests_from_spec(
    requests: Sequence[Mapping[str, Any]],
    spec: TypedResearchTaskSpec,
) -> List[Dict[str, Any]]:
    repaired: List[Dict[str, Any]] = []
    single_outcome = spec.outcomes[0] if len(spec.outcomes) == 1 else ""
    has_kmeans = any(
        str(item.get("tool") or "") == "kmeans_clustering"
        for item in requests
    )
    kmeans_group_outputs = {
        "group_descriptive_stats",
        "group_means",
        "group_distributions",
    }
    for raw_request in requests:
        request = dict(raw_request)
        tool = str(request.get("tool") or "")
        if tool == "filter_group" and has_kmeans:
            group_by = {
                str(item) for item in list(request.get("group_by") or [])
            }
            if group_by.intersection({"cluster", "cluster_assignments"}):
                continue
        if tool == "describe" and has_kmeans:
            describe_columns = {
                str(item) for item in list(request.get("columns") or [])
            }
            if (
                describe_columns.intersection({"cluster", "cluster_assignments"})
                or set(spec.required_outputs).intersection(kmeans_group_outputs)
            ):
                continue
        if single_outcome and tool in {
            "linear_regression",
            "logistic_regression",
            "adjusted_logistic_association",
            "binary_model_comparison",
            "regression_model_comparison",
            "multiclass_model_comparison",
        }:
            request["outcome"] = single_outcome
        if tool == "adjusted_logistic_association":
            if spec.exposures:
                request["exposures"] = list(spec.exposures)
            if spec.forced_covariates:
                request["covariates"] = list(spec.forced_covariates)
        elif tool in {"linear_regression", "logistic_regression"}:
            requested_predictors = _unique(
                list(spec.exposures)
                + list(spec.predictors)
                + list(spec.forced_covariates)
            )
            if requested_predictors:
                request["predictors"] = requested_predictors
        elif tool in {
            "binary_model_comparison",
            "regression_model_comparison",
            "multiclass_model_comparison",
        }:
            if spec.candidate_scope == "explicit" and spec.candidate_features:
                request["features"] = list(spec.candidate_features)
            elif spec.candidate_scope == "auto_discover":
                request["features"] = []
                request["candidate_scope"] = "auto_discover"
                request["forced_covariates"] = list(spec.forced_covariates)
                request["excluded_features"] = list(spec.excluded_features)
            if spec.top_k:
                request["top_k"] = spec.top_k
            if "lasso" in spec.requested_methods and tool == "binary_model_comparison":
                methods = list(request.get("methods") or [])
                if not any(_normalize_method(item) == "lasso" for item in methods):
                    methods.insert(0, "lasso_logistic")
                request["methods"] = _unique(methods)
        elif tool == "filter_group":
            if spec.filters:
                request["filters"] = list(spec.filters)
            if spec.group_by:
                request["group_by"] = list(spec.group_by)
        elif tool == "histogram" and spec.group_by:
            request["group"] = spec.group_by[0]
        elif tool == "scatter" and spec.group_by:
            request["group"] = spec.group_by[0]
        elif tool == "visualization" and spec.group_by:
            kind = (
                str(request.get("kind") or "")
                .strip()
                .lower()
                .replace("-", "_")
            )
            if kind in {"box", "boxplot", "violin", "bar"}:
                request["x"] = spec.group_by[0]
            else:
                request["group"] = spec.group_by[0]
        elif tool == "kmeans_clustering":
            requested_features = (
                list(spec.predictors)
                or list(spec.candidate_features)
            )
            if requested_features:
                request["features"] = _unique(requested_features)
            cluster_count = int(
                dict(spec.explicit_facts or {}).get("cluster_count")
                or request.get("n_clusters")
                or request.get("clusters")
                or 0
            )
            if cluster_count:
                request["n_clusters"] = cluster_count
        repaired.append(request)
    return repaired


def _resource_only_current_turn(
    raw_spec: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> bool:
    resources = dict(facts.get("explicit_resource_requests") or {})
    if not any(bool(value) for value in resources.values()):
        return False
    goal = str(raw_spec.get("goal") or "").strip().lower()
    task_family = str(raw_spec.get("task_family") or "").strip().lower()
    return goal in {
        "research_discussion",
        "literature_review",
        "evidence_review",
        "methodology_review",
    } or task_family in {
        "evidence_research",
        "literature_review",
        "research_discussion",
        "other",
    }


def _prune_redundant_invalid_requests(
    requests: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    required_outputs: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Remove invalid derived-column nodes only when they add no needed output.

    This repairs planner artifacts such as a separate PC1/PC2 scatter after a
    PCA tool that already returns the requested component figure. Nodes with a
    unique required output remain in the graph and are rejected by validation.
    """

    allowed_columns = set(str(item) for item in columns)
    normalized = [dict(item) for item in requests]
    valid_outputs: set[str] = set()
    for request in normalized:
        missing = set(_request_columns(request)) - allowed_columns
        if not missing:
            valid_outputs.update(
                TOOL_OUTPUTS.get(str(request.get("tool") or ""), set())
            )
    required = {
        _normalize_output(item)
        for item in required_outputs
        if str(item).strip()
    }
    kept: List[Dict[str, Any]] = []
    notes: List[str] = []
    for request in normalized:
        capability = str(request.get("tool") or "")
        missing = sorted(set(_request_columns(request)) - allowed_columns)
        provided = set(TOOL_OUTPUTS.get(capability, set()))
        needed = required.intersection(provided)
        if missing and needed.issubset(valid_outputs):
            notes.append(
                "Removed redundant "
                f"{capability} node with unavailable derived columns "
                f"{', '.join(missing)}; verified upstream output already "
                "satisfies the user contract."
            )
            continue
        kept.append(request)
    return kept, notes


def reconcile_research_plan(
    question: str,
    plan: Mapping[str, Any],
    columns: Sequence[str],
) -> Dict[str, Any]:
    result = dict(plan or {})
    raw_spec = (
        dict(result.get("task_spec") or {})
        if isinstance(result.get("task_spec"), Mapping)
        else {}
    )
    requests = [
        dict(item)
        for item in list(result.get("tool_requests") or [])
        if isinstance(item, Mapping)
    ]
    facts = extract_deterministic_facts(question, columns)
    protected_fields = protected_missing_fields(question, columns)
    if protected_fields:
        raw_spec["filters"] = [item for item in normalize_filter_conditions(raw_spec.get("filters") or []) if item["field"] not in protected_fields]
        for request in requests:
            if "filters" in request:
                request["filters"] = [item for item in normalize_filter_conditions(request["filters"]) if item["field"] not in protected_fields]
    if _resource_only_current_turn(raw_spec, facts):
        facts = dict(facts)
        facts["requested_methods"] = []
        facts["method_selection_authority"] = ""
        facts["required_outputs"] = [
            _normalize_output(item)
            for item in list(facts.get("required_outputs") or [])
            if _normalize_output(item) in RESEARCH_SYNTHESIS_OUTPUTS
        ]
        requests = [
            request
            for request in requests
            if str(request.get("tool") or "") not in ANALYTIC_TOOLS
        ]
        raw_spec["goal"] = "research_discussion"
        raw_spec["task_family"] = "evidence_research"
        raw_spec["requested_methods"] = []
        raw_spec["methods"] = []
        raw_spec["method"] = ""
        raw_spec["method_selection_authority"] = "chair_inferred"
        research_outputs = [
            _normalize_output(item)
            for item in _list_from_spec(
                raw_spec,
                "required_outputs",
                "outputs",
            )
            if _normalize_output(item) in RESEARCH_SYNTHESIS_OUTPUTS
        ]
        raw_spec["required_outputs"] = research_outputs or [
            "pmid_list",
            "literature_evidence",
            "evidence_limitation_summary",
        ]
        raw_spec["needs_clarification"] = False
        raw_spec["clarification_question"] = ""
        result["needs_clarification"] = False
        result["clarification_question"] = ""
    requests = _apply_explicit_visualization_contract(requests, facts)
    tool_roles = _roles_from_tool_requests(requests, columns)
    notes: List[str] = []

    visualization_ready = any(
        str(item.get("tool") or "") in {
            "scatter",
            "histogram",
            "heatmap",
            "visualization",
        }
        for item in requests
    )
    if visualization_ready:
        # Built-in heatmaps have safe defaults. Optional axis-range preferences
        # do not justify blocking execution or asking the user to upload data.
        raw_spec["needs_clarification"] = False
        raw_spec["clarification_question"] = ""
        result["needs_clarification"] = False
        result["clarification_question"] = ""

    semantic_outcomes = _valid_columns(
        _list_from_spec(raw_spec, "outcomes", "outcome", "target")
        or tool_roles["outcomes"],
        columns,
    )
    semantic_exposures = _valid_columns(
        _list_from_spec(raw_spec, "exposures", "exposure")
        or tool_roles["exposures"],
        columns,
    )
    semantic_predictors = _valid_columns(
        _list_from_spec(raw_spec, "predictors") or tool_roles["predictors"],
        columns,
    )
    semantic_candidates = _valid_columns(
        _list_from_spec(
            raw_spec,
            "candidate_features",
            "explicit_candidate_features",
        )
        or tool_roles["candidate_features"],
        columns,
    )
    semantic_covariates = _valid_columns(
        _list_from_spec(raw_spec, "forced_covariates", "covariates")
        or tool_roles["forced_covariates"],
        columns,
    )
    semantic_group_by = _valid_columns(
        _list_from_spec(raw_spec, "group_by"),
        columns,
    )
    semantic_filters = normalize_filter_conditions(
        _list_from_spec(raw_spec, "filters") or tool_roles["filters"]
    )
    protected_fields = protected_missing_fields(question, columns)
    semantic_filters = [item for item in semantic_filters if item["field"] not in protected_fields]
    raw_semantic_methods = [
        _normalize_method(item)
        for item in _list_from_spec(
            raw_spec,
            "requested_methods",
            "methods",
            "method",
        )
    ]
    tool_selected_methods = _unique(
        _normalize_method(item)
        for item in list(tool_roles["requested_methods"] or [])
    )
    explicit_methods = _unique(
        _normalize_method(item)
        for item in list(facts.get("requested_methods") or [])
    )
    raw_method_authority = str(
        raw_spec.get("method_selection_authority") or ""
    ).strip()
    if explicit_methods:
        method_selection_authority = "user_explicit"
        final_requested_methods = explicit_methods
    elif facts.get("method_selection_delegated"):
        method_selection_authority = "chair_delegated"
        final_requested_methods = []
    elif raw_method_authority == "user_explicit":
        method_selection_authority = "user_explicit"
        final_requested_methods = _unique(raw_semantic_methods)
    else:
        method_selection_authority = "chair_inferred"
        final_requested_methods = []
    selected_methods = _unique(
        tool_selected_methods
        + (
            raw_semantic_methods
            if method_selection_authority != "user_explicit"
            else []
        )
    )
    if visualization_ready and not explicit_methods:
        # Plotting libraries and implementation details are never locked as
        # user-requested statistical methods.
        final_requested_methods = []
    semantic_outputs = [
        _normalize_output(item)
        for item in _list_from_spec(raw_spec, "required_outputs", "outputs")
    ]
    if "literature" in facts.get("prohibited_resources", []):
        semantic_outputs = [item for item in semantic_outputs if item not in LITERATURE_OUTPUTS]
    explicit_resources = dict(
        facts.get("explicit_resource_requests") or {}
    )
    merged_required_outputs = _merge_with_explicit(
        "required_outputs",
        _unique(semantic_outputs),
        [
            _normalize_output(item)
            for item in list(facts.get("required_outputs") or [])
        ],
        notes,
    )
    evidence_required_outputs = [
        output
        for output in semantic_outputs
        if output in RESEARCH_SYNTHESIS_OUTPUTS
    ]
    if explicit_resources.get("literature") and not evidence_required_outputs:
        evidence_required_outputs = [
            "pmid_list",
            "literature_evidence",
            "evidence_limitation_summary",
        ]
    merged_required_outputs = _unique(
        list(merged_required_outputs) + list(evidence_required_outputs)
    )

    candidate_scope = str(
        raw_spec.get("candidate_scope")
        or raw_spec.get("feature_discovery_scope")
        or facts.get("candidate_scope")
        or ("explicit" if semantic_candidates else "")
    ).strip()
    explicit_candidates = list(facts.get("candidate_features") or [])
    if facts.get("candidate_scope"):
        candidate_scope = str(facts["candidate_scope"])
    if candidate_scope == "auto_discover":
        protected_covariates = set(semantic_covariates) | set(
            facts.get("forced_covariates") or []
        )
        semantic_candidates = [
            column
            for column in semantic_candidates
            if column not in protected_covariates
        ]

    final_outcomes = _merge_with_explicit(
        "outcome",
        semantic_outcomes,
        list(facts.get("outcomes") or []),
        notes,
    )
    final_covariates = _merge_with_explicit(
        "forced_covariates",
        semantic_covariates,
        list(facts.get("forced_covariates") or []),
        notes,
    )
    final_exposures = [
        item
        for item in _merge_with_explicit(
            "exposure",
            semantic_exposures,
            list(facts.get("exposures") or []),
            notes,
        )
        if item not in set(final_outcomes) | set(final_covariates)
    ]
    clarification_text = str(
        raw_spec.get("clarification_question")
        or result.get("clarification_question")
        or ""
    ).strip()
    clarification_needed = bool(
        raw_spec.get("needs_clarification")
        or result.get("needs_clarification")
    )
    method_only_clarification = bool(
        re.search(
            r"(?:方法|檢定|模型|method).{0,80}"
            r"(?:維持|替代|選擇|允許|確認|alternative)"
            r"|(?:維持|替代|選擇|允許|確認|alternative).{0,80}"
            r"(?:方法|檢定|模型|method)",
            clarification_text,
            flags=re.IGNORECASE,
        )
    )
    outcome_only_clarification = bool(
        re.fullmatch(
            r"\s*(?:請)?(?:確認|指定|提供)?(?:要|欲)?(?:進行|執行|做)?"
            r".{0,64}(?:outcome|target|結果變項|依變項|結局)"
            r"(?:\s*欄位)?\s*[。？?]?\s*",
            clarification_text,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            r"\s*(?:請)?(?:確認|指定|提供)?.{0,40}(?:哪個|何者|什麼)"
            r".{0,24}(?:欄位)?.{0,24}(?:outcome|target|結果變項|依變項|結局)"
            r"\s*[。？?]?\s*",
            clarification_text,
            flags=re.IGNORECASE,
        )
    )
    if clarification_needed and len(final_outcomes) == 1 and outcome_only_clarification:
        clarification_needed = False
        clarification_text = ""
        result["needs_clarification"] = False
        result["clarification_question"] = ""
        notes.append(
            "The current utterance explicitly scoped one schema column as the "
            "feature-selection outcome; an outcome-only clarification is satisfied."
        )
    if (
        method_selection_authority == "chair_delegated"
        and requests
        and method_only_clarification
    ):
        clarification_needed = False
        clarification_text = ""
        result["needs_clarification"] = False
        result["clarification_question"] = ""
        notes.append(
            "The user delegated method selection; the chair-selected verified "
            "method is an implementation decision, not a clarification blocker."
        )

    spec = TypedResearchTaskSpec(
        question=str(question or ""),
        goal=str(raw_spec.get("goal") or result.get("research_goal") or "").strip(),
        task_family=str(
            raw_spec.get("task_family") or _task_family_from_requests(requests)
        ).strip(),
        outcomes=final_outcomes,
        exposures=final_exposures,
        predictors=_merge_with_explicit(
            "predictors",
            semantic_predictors,
            list(facts.get("predictors") or []),
            notes,
        ),
        candidate_features=(
            explicit_candidates
            if facts.get("candidate_scope") == "explicit"
            else semantic_candidates
        ),
        candidate_scope=candidate_scope,
        forced_covariates=final_covariates,
        excluded_features=_valid_columns(
            _list_from_spec(raw_spec, "excluded_features", "excluded_variables"),
            columns,
        ),
        group_by=_merge_with_explicit(
            "group_by",
            semantic_group_by,
            list(facts.get("group_by") or []),
            notes,
        ),
        filters=(
            list(facts.get("filters") or [])
            if facts.get("filters")
            else _unique(semantic_filters)
        ),
        requested_methods=final_requested_methods,
        selected_methods=selected_methods,
        method_selection_authority=method_selection_authority,
        required_outputs=merged_required_outputs,
        top_k=int(facts.get("top_k") or raw_spec.get("top_k") or tool_roles["top_k"] or 0),
        needs_clarification=clarification_needed,
        clarification_question=clarification_text,
        explicit_facts=dict(facts),
        reconciliation_notes=notes,
        conflicts=[],
    )
    repaired_requests = _repair_requests_from_spec(requests, spec)
    repaired_requests, pruning_notes = _prune_redundant_invalid_requests(
        repaired_requests,
        columns,
        spec.required_outputs,
    )
    if pruning_notes:
        spec.reconciliation_notes.extend(pruning_notes)
    result["tool_requests"] = repaired_requests
    spec_payload = spec.to_dict()
    requirement = build_user_requirement(
        question,
        spec_payload,
        facts,
        existing=(
            dict(result.get("user_requirement") or {})
            if isinstance(result.get("user_requirement"), Mapping)
            else None
        ),
    )
    spec_payload = project_requirement_to_task_spec(
        requirement,
        spec_payload,
    )
    spec_payload.pop("spec_id", None)
    spec_payload.pop("spec_hash", None)
    spec_payload["spec_id"] = "SPEC_" + canonical_hash(spec_payload)[:16]
    spec_payload["spec_hash"] = canonical_hash(spec_payload)
    result["user_requirement"] = requirement
    result["task_spec"] = spec_payload
    graph = build_analysis_graph(result["task_spec"], result["tool_requests"])
    result["analysis_graph"] = graph
    result["contract_validation"] = validate_analysis_graph(
        result["task_spec"],
        graph,
        columns,
    )
    result["capability_fit"] = assess_capability_fit(
        result["task_spec"],
        graph,
        facts,
    )
    result["evidence_contract"] = {
        "required_outputs": list(evidence_required_outputs),
        "sources": (
            ["pubmed", "europe_pmc"]
            if explicit_resources.get("literature")
            else []
        ),
        "citation_policy": (
            "retrieved_records_only"
            if explicit_resources.get("literature")
            else "not_requested"
        ),
        "status": (
            "pending_retrieval"
            if evidence_required_outputs
            else "not_required"
        ),
    }
    if explicit_resources.get("literature"):
        result["needs_literature"] = True
    if explicit_resources.get("external"):
        result["needs_external_information"] = True
    if explicit_resources.get("custom_code"):
        result["needs_code"] = True
    if result["capability_fit"]["minimum_sufficient_resources"]:
        result["needs_literature"] = False
        result["literature_queries"] = []
        result["needs_external_information"] = False
        result["external_information_queries"] = []
        result["needs_code"] = False
        result["delegations"] = []
        result["resource_pruning"] = {
            "applied": True,
            "reason": (
                "Verified built-in capabilities cover every required method and "
                "output; external resources and extra collaborators are prohibited."
            ),
        }
    return enforce_resource_constraints(question, result)


def build_analysis_graph(
    task_spec: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    nodes: List[TaskNode] = []
    aliases: Dict[str, str] = {}
    cohort_node = ""
    manifests = [
        get_tool_manifest(str(request.get("tool") or ""))
        for request in requests
    ]
    active_branches = {
        str(getattr(manifest, "branch", "analysis") or "analysis")
        for manifest in manifests
        if manifest is not None
    }
    only_visualization = active_branches == {"visualization"}
    only_data = active_branches == {"data"}
    for index, request in enumerate(requests, start=1):
        capability = str(request.get("tool") or "")
        manifest = get_tool_manifest(capability)
        request_payload = dict(request)
        raw_dependencies = list(request_payload.pop("depends_on", []) or [])
        node_hash = canonical_hash(
            {"index": index, "capability": capability, "request": request_payload}
        )
        node_id = f"NODE_{index:02d}_{node_hash[:10]}"
        dependencies: List[str] = []
        for dependency in raw_dependencies:
            dependency_id = aliases.get(str(dependency), str(dependency))
            if dependency_id and dependency_id not in dependencies:
                dependencies.append(dependency_id)
        source = str(request_payload.get("input_dataset") or "").strip()
        dataset_node = "" if source == "original" else aliases.get(source, source) if source else cohort_node
        if capability in {"model_status", "meeting_status", "package_status", "pubmed_connector_search", "europe_pmc_connector_search"}:
            dataset_node = ""
        if dataset_node and dataset_node not in dependencies:
            dependencies.append(dataset_node)
        branch = str(getattr(manifest, "branch", "analysis") or "analysis")
        criticality = "core"
        if branch == "visualization" and not only_visualization:
            criticality = "optional"
        elif branch == "data" and not only_data:
            criticality = "support"
        elif branch in {"evidence", "external_resources"}:
            criticality = "optional"
        timeout_sec = int(getattr(manifest, "timeout_sec", 60) or 60)
        retry = int(
            dict(getattr(manifest, "error_policy", {}) or {}).get(
                "max_retries", 0
            )
            or 0
        )
        node = TaskNode(
            node_id=node_id,
            capability=capability,
            request=request_payload,
            depends_on=dependencies,
            input_contract={
                "dataset": "current_session_dataset",
                "dataset_node_id": dataset_node,
                "columns": _request_columns(request_payload),
            },
            output_contract=sorted(TOOL_OUTPUTS.get(capability, set())),
            required=criticality == "core",
            branch=branch,
            criticality=criticality,
            timeout_sec=timeout_sec,
            retry=retry,
            queue_class=str(
                getattr(manifest, "queue_class", "statistics")
                or "statistics"
            ),
            resource_profile=dict(
                getattr(manifest, "resource_profile_v2", {}) or {}
            ),
            error_policy=dict(getattr(manifest, "error_policy", {}) or {}),
        )
        nodes.append(node)
        aliases[str(index)] = node_id
        aliases[capability] = node_id
        aliases[node_id] = node_id
        if capability == "filter_group":
            cohort_node = node_id
    required_outputs = _execution_required_outputs(task_spec)
    output_bindings = {
        output: [
            node.node_id
            for node in nodes
            if output in set(node.output_contract)
        ]
        for output in required_outputs
    }
    graph = AnalysisGraph(
        spec_hash=str(task_spec.get("spec_hash") or ""),
        nodes=nodes,
        required_outputs=required_outputs,
        output_bindings=output_bindings,
    )
    return graph.to_dict()


def _request_columns(request: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in (
        "outcome",
        "duration",
        "event",
        "x",
        "y",
        "column",
        "group",
    ):
        if request.get(key):
            values.append(request.get(key))
    for key in (
        "columns",
        "predictors",
        "features",
        "exposures",
        "covariates",
        "forced_covariates",
        "excluded_features",
        "group_by",
    ):
        values.extend(request.get(key) or [])
    for condition in request.get("filters") or []:
        if isinstance(condition, Mapping) and condition.get("field"):
            values.append(condition.get("field"))
    return _unique(str(value) for value in values if value not in (None, ""))


def _graph_methods(graph: Mapping[str, Any]) -> List[str]:
    methods: List[str] = []
    for node in list(graph.get("nodes") or []):
        request = dict(node.get("request") or {})
        tool = str(node.get("capability") or request.get("tool") or "")
        methods.extend(TOOL_METHODS.get(tool, set()))
        methods.extend(_normalize_method(item) for item in request.get("methods") or [])
    return _unique(methods)


def _graph_outputs(graph: Mapping[str, Any]) -> List[str]:
    outputs: List[str] = []
    for node in list(graph.get("nodes") or []):
        capability = str(node.get("capability") or "")
        outputs.extend(TOOL_OUTPUTS.get(capability, set()))
        if (
            capability == "heatmap"
            and str(dict(node.get("request") or {}).get("kind") or "")
            == "correlation"
        ):
            outputs.append("correlation")
    return _unique(outputs)


def assess_capability_fit(
    task_spec: Mapping[str, Any],
    graph: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prove whether verified local capabilities fully cover the user contract."""

    nodes = list(graph.get("nodes") or [])
    capabilities = [
        str(node.get("capability") or "")
        for node in nodes
        if str(node.get("capability") or "")
    ]
    research_only = _research_only_contract(task_spec)
    required_methods = {
        _normalize_method(item)
        for item in list(task_spec.get("requested_methods") or [])
        if _normalize_method(item)
    } if not research_only else set()
    available_methods = set(_graph_methods(graph))
    required_outputs = set(_execution_required_outputs(task_spec))
    available_outputs = set(_graph_outputs(graph))
    missing_methods = sorted(required_methods - available_methods)
    missing_outputs = sorted(required_outputs - available_outputs)
    unsupported_capabilities = sorted(
        capability
        for capability in capabilities
        if capability not in ANALYTIC_TOOLS
        and capability
        not in {
            "model_status",
            "meeting_status",
            "package_status",
            "dataset_profile",
        }
    )
    explicit_resources = dict(facts.get("explicit_resource_requests") or {})
    complete = (research_only or bool(nodes)) and not (
        missing_methods or missing_outputs or unsupported_capabilities
    )
    explicit_expansion = any(bool(value) for value in explicit_resources.values())
    return {
        "status": "covered" if complete else "gap",
        "complete": complete,
        "minimum_sufficient_resources": complete and not explicit_expansion,
        "capabilities": capabilities,
        "required_methods": sorted(required_methods),
        "available_methods": sorted(available_methods),
        "missing_methods": missing_methods,
        "required_outputs": sorted(required_outputs),
        "available_outputs": sorted(available_outputs),
        "missing_outputs": missing_outputs,
        "unsupported_capabilities": unsupported_capabilities,
        "explicit_resource_requests": explicit_resources,
        "policy": (
            "When coverage is complete, external search, package discovery and "
            "additional agents cannot be added without an explicit user need."
        ),
        "assessment_hash": canonical_hash(
            {
                "capabilities": capabilities,
                "required_methods": sorted(required_methods),
                "available_methods": sorted(available_methods),
                "required_outputs": sorted(required_outputs),
                "available_outputs": sorted(available_outputs),
                "explicit_resources": explicit_resources,
            }
        ),
    }


def validate_analysis_graph(
    task_spec: Mapping[str, Any],
    graph: Mapping[str, Any],
    columns: Sequence[str],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    allowed_columns = set(str(item) for item in columns)
    nodes = list(graph.get("nodes") or [])
    node_ids = {
        str(node.get("node_id") or "")
        for node in nodes
        if str(node.get("node_id") or "")
    }
    completed_node_ids: set[str] = set()

    if str(graph.get("spec_hash") or "") != str(task_spec.get("spec_hash") or ""):
        errors.append("AnalysisGraph 與 TypedResearchTaskSpec 的 hash 不一致。")

    for node in nodes:
        node_id = str(node.get("node_id") or "")
        capability = str(node.get("capability") or "")
        dependencies = [
            str(item)
            for item in list(node.get("depends_on") or [])
            if str(item)
        ]
        unknown_dependencies = sorted(set(dependencies) - node_ids)
        if unknown_dependencies:
            errors.append(
                f"{node_id} 依賴不存在的分析節點："
                + ", ".join(unknown_dependencies)
            )
        late_dependencies = sorted(set(dependencies) - completed_node_ids)
        if late_dependencies:
            errors.append(
                f"{node_id} 的依賴未先完成或形成循環："
                + ", ".join(late_dependencies)
            )
        if capability not in ANALYTIC_TOOLS and capability not in {
            "model_status",
            "meeting_status",
            "package_status",
            "dataset_profile",
        }:
            errors.append(f"AnalysisGraph 包含未核准能力：{capability}")
        request = dict(node.get("request") or {})
        missing = [
            column
            for column in _request_columns(request)
            if column not in allowed_columns and capability not in {
                "model_status",
                "meeting_status",
                "package_status",
            }
        ]
        if missing:
            errors.append(
                f"{capability} 使用不存在的欄位：{', '.join(missing)}"
            )
        declared_outputs = set(
            str(item) for item in list(node.get("output_contract") or [])
        )
        expected_outputs = set(TOOL_OUTPUTS.get(capability, set()))
        if declared_outputs != expected_outputs:
            errors.append(f"{node_id} 的輸出型別契約與能力登錄不一致。")
        completed_node_ids.add(node_id)

    research_only = _research_only_contract(task_spec)
    required_methods = set(
        _normalize_method(item)
        for item in list(task_spec.get("requested_methods") or [])
    ) if not research_only else set()
    available_methods = set(_graph_methods(graph))
    missing_methods = sorted(required_methods - available_methods)
    if missing_methods and nodes:
        errors.append(
            "AnalysisGraph 未保留使用者指定方法：" + ", ".join(missing_methods)
        )

    required_outputs = set(_execution_required_outputs(task_spec))
    available_outputs = set(_graph_outputs(graph))
    missing_outputs = sorted(required_outputs - available_outputs)
    if missing_outputs and nodes:
        errors.append(
            "AnalysisGraph 無法產生必要輸出：" + ", ".join(missing_outputs)
        )
    graph_required_outputs = set(
        str(item) for item in list(graph.get("required_outputs") or [])
    )
    if graph_required_outputs != required_outputs:
        errors.append("AnalysisGraph 的必要輸出契約與研究需求不一致。")
    output_bindings = {
        str(name): [str(node_id) for node_id in list(providers or [])]
        for name, providers in dict(graph.get("output_bindings") or {}).items()
    }
    for output in sorted(required_outputs):
        providers = output_bindings.get(output, [])
        if not providers:
            errors.append(f"必要輸出沒有綁定提供節點：{output}")
        elif any(provider not in node_ids for provider in providers):
            errors.append(f"必要輸出綁定到不存在的節點：{output}")

    research_only_roles = _research_only_contract(task_spec)
    outcomes = set(str(item) for item in task_spec.get("outcomes") or [])
    graph_outcomes = {
        str(value)
        for node in nodes
        for value in (
            dict(node.get("request") or {}).get("outcome"),
            dict(node.get("request") or {}).get("y"),
        )
        if value
    }
    if (
        not research_only_roles
        and outcomes
        and not outcomes.issubset(graph_outcomes)
    ):
        errors.append(
            "AnalysisGraph 的 outcome 與使用者需求不一致："
            + f"需要 {sorted(outcomes)}，實際 {sorted(graph_outcomes)}"
        )

    required_exposures = set(
        str(item) for item in task_spec.get("exposures") or []
    )
    if required_exposures and not research_only_roles:
        graph_exposures = {
            str(item)
            for node in nodes
            for item in (
                list(dict(node.get("request") or {}).get("exposures") or [])
                + list(dict(node.get("request") or {}).get("predictors") or [])
                + list(dict(node.get("request") or {}).get("features") or [])
                + [dict(node.get("request") or {}).get("x")]
            )
            if item
        }
        if not required_exposures.issubset(graph_exposures):
            errors.append(
                "AnalysisGraph did not bind every requested exposure to an "
                "executable model input: "
                + ", ".join(sorted(required_exposures - graph_exposures))
            )

    required_predictors = set(
        str(item) for item in task_spec.get("predictors") or []
    )
    if required_predictors and not research_only_roles:
        graph_predictors = {
            str(item)
            for node in nodes
            for item in (
                list(dict(node.get("request") or {}).get("predictors") or [])
                + list(dict(node.get("request") or {}).get("features") or [])
                + list(dict(node.get("request") or {}).get("columns") or [])
                + list(dict(node.get("request") or {}).get("exposures") or [])
                + list(dict(node.get("request") or {}).get("covariates") or [])
                + list(
                    dict(node.get("request") or {}).get(
                        "forced_covariates"
                    )
                    or []
                )
                + [dict(node.get("request") or {}).get("x")]
            )
            if item
        }
        if not required_predictors.issubset(graph_predictors):
            errors.append(
                "AnalysisGraph did not bind every requested predictor to an "
                "executable model input: "
                + ", ".join(sorted(required_predictors - graph_predictors))
            )

    required_filters = list(task_spec.get("filters") or [])
    if required_filters and not research_only_roles:
        graph_filters = [
            item
            for node in nodes
            for item in list(dict(node.get("request") or {}).get("filters") or [])
        ]
        def normalized_filter(item: Mapping[str, Any]) -> Dict[str, Any]:
            normalized = normalize_filter_condition(item)
            return {
                "field": normalized["field"],
                "operator": normalized["operator"],
                "value": normalized.get("value"),
                "logic": normalized["logic"],
            }

        if {
            json.dumps(
                normalized_filter(item),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            for item in required_filters
        } != {
            json.dumps(
                normalized_filter(item),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            for item in graph_filters
        }:
            errors.append("AnalysisGraph 的篩選條件與使用者原文不一致。")

    required_group_by = set(str(item) for item in task_spec.get("group_by") or [])
    if required_group_by and not research_only_roles:
        graph_group_by = {
            str(item)
            for node in nodes
            for item in _grouping_columns_from_request(
                dict(node.get("request") or {})
            )
        }
        if not required_group_by.issubset(graph_group_by):
            errors.append("AnalysisGraph 未保留使用者指定的分組欄位。")

    forced = set(str(item) for item in task_spec.get("forced_covariates") or [])
    if forced and not research_only_roles:
        graph_covariates = {
            str(item)
            for node in nodes
            for item in (
                list(dict(node.get("request") or {}).get("covariates") or [])
                + list(
                    dict(node.get("request") or {}).get(
                        "forced_covariates"
                    )
                    or []
                )
                + list(dict(node.get("request") or {}).get("predictors") or [])
            )
        }
        if not forced.issubset(graph_covariates):
            errors.append("AnalysisGraph 未保留使用者指定的固定調整變項。")

    if (
        task_spec.get("candidate_scope") == "explicit"
        and not research_only_roles
    ):
        required_features = set(
            str(item) for item in task_spec.get("candidate_features") or []
        )
        graph_features = {
            str(item)
            for node in nodes
            for item in list(dict(node.get("request") or {}).get("features") or [])
        }
        if graph_features and graph_features != required_features:
            errors.append("AnalysisGraph 改寫了使用者明確指定的候選特徵集合。")

    if not nodes and (
        task_spec.get("requested_methods")
        or task_spec.get("outcomes")
        or task_spec.get("filters")
    ):
        warnings.append("尚未建立可執行 AnalysisGraph；可能需要澄清或能力探索。")

    return {
        "status": "blocked" if errors else ("warning" if warnings else "valid"),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "spec_hash": str(task_spec.get("spec_hash") or ""),
        "graph_hash": str(graph.get("graph_hash") or ""),
    }
