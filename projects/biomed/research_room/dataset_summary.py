from __future__ import annotations

import csv
import html
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEMOGRAPHIC_PRIORITY = [
    "AGE",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "DEPENDENCY",
    "PLACE_CURR",
    "INCOME_SELF",
    "INCOME_FAMILY",
    "JOB_CURR",
    "JOB_EXPERIENCE",
    "BODY_HEIGHT",
    "BODY_WEIGHT",
    "BMI",
    "BODY_WAISTLINE",
    "BODY_BUTTOCKS",
    "WHR",
    "OBESITY",
]

DEMOGRAPHIC_TOKENS = {
    "age",
    "年齡",
    "sex",
    "gender",
    "性別",
    "education",
    "學歷",
    "marriage",
    "婚姻",
    "dependency",
    "獨居",
    "place_curr",
    "居住",
    "income",
    "收入",
    "job",
    "工作",
    "body_height",
    "身高",
    "body_weight",
    "體重",
    "bmi",
    "身體質量",
    "waist",
    "腰圍",
    "buttocks",
    "臀圍",
    "whr",
    "腰臀比",
    "obesity",
    "肥胖",
}


def load_variable_dictionary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    result: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            field = str(row.get("field") or "").strip()
            if not field or field in result:
                continue
            result[field] = {
                "label": str(row.get("filter_sub_name") or "").strip(),
                "category": str(row.get("filter_name") or "").strip(),
                "choice_items": str(row.get("choice_items") or "").strip(),
                "declared_type": str(row.get("type") or "").strip(),
                "remark": str(row.get("remark") or "").strip(),
            }
    return result


def infer_variable_type(series: pd.Series) -> str:
    sample = series.replace("", np.nan).dropna()
    if sample.empty:
        return "Empty"
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=42)
    unique_count = int(sample.nunique(dropna=True))
    numeric = pd.to_numeric(sample, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    values = {
        str(value).strip().lower()
        for value in sample.astype(str).drop_duplicates().head(20)
    }
    binary_tokens = {
        "0",
        "1",
        "yes",
        "no",
        "true",
        "false",
        "y",
        "n",
        "男",
        "女",
        "是",
        "否",
        "有",
        "無",
    }
    if unique_count <= 2 and (
        numeric_ratio >= 0.8 or values.issubset(binary_tokens) or unique_count == 2
    ):
        return "Binary"
    if numeric_ratio >= 0.85:
        numeric_values = numeric.dropna()
        integer_like = bool(
            np.all(np.isclose(numeric_values, np.round(numeric_values), equal_nan=True))
        )
        threshold = max(20, min(80, int(len(numeric_values) * 0.08)))
        if integer_like and 3 <= unique_count <= threshold:
            return "Count / Discrete"
        return "Continuous"
    if unique_count <= max(20, int(len(sample) * 0.2)):
        return "Categorical"
    return "Text / ID"


def _histogram(values: pd.Series) -> List[Dict[str, Any]]:
    if values.empty:
        return []
    unique_count = int(values.nunique())
    bins = min(12, max(4, int(math.sqrt(max(unique_count, 1)))))
    counts, edges = np.histogram(values, bins=bins)
    return [
        {
            "start": float(edges[index]),
            "end": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def summarize_field(
    frame: pd.DataFrame,
    column: str,
    variable_dictionary: Mapping[str, Mapping[str, str]],
    inferred_type: str = "",
    *,
    missing_n: Optional[int] = None,
    unique_count: Optional[int] = None,
) -> Dict[str, Any]:
    series = frame[column].replace("", np.nan)
    field_type = inferred_type or infer_variable_type(series)
    missing_n = (
        int(series.isna().sum())
        if missing_n is None
        else int(missing_n)
    )
    non_missing = series.dropna()
    unique_count = (
        int(non_missing.nunique(dropna=True))
        if unique_count is None
        else int(unique_count)
    )
    dictionary_entry = dict(variable_dictionary.get(column) or {})
    numeric = pd.to_numeric(non_missing, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(non_missing) else 0.0
    statistics: Dict[str, Any] = {}
    top_values: List[Dict[str, Any]] = []
    histogram: List[Dict[str, Any]] = []
    if field_type in {"Continuous", "Count / Discrete"} and numeric_ratio >= 0.85:
        values = numeric.dropna()
        if not values.empty:
            quantiles = values.quantile([0.25, 0.5, 0.75])
            statistics = {
                "mean": float(values.mean()),
                "sd": float(values.std()) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "p25": float(quantiles.loc[0.25]),
                "median": float(quantiles.loc[0.5]),
                "p75": float(quantiles.loc[0.75]),
                "max": float(values.max()),
            }
            histogram = _histogram(values)
    else:
        top_values = [
            {
                "value": str(value),
                "count": int(count),
                "percentage": float(count / len(non_missing) * 100)
                if len(non_missing)
                else 0.0,
            }
            for value, count in non_missing.astype(str).value_counts().head(8).items()
        ]
    return {
        "field": column,
        "label": dictionary_entry.get("label", ""),
        "category": dictionary_entry.get("category", ""),
        "choice_items": dictionary_entry.get("choice_items", ""),
        "declared_type": dictionary_entry.get("declared_type", ""),
        "remark": dictionary_entry.get("remark", ""),
        "inferred_type": field_type,
        "missing_n": missing_n,
        "missing_pct": float(missing_n / len(series) * 100) if len(series) else 0.0,
        "unique_count": unique_count,
        "sample_values": [str(value) for value in non_missing.drop_duplicates().head(3)],
        "statistics": statistics,
        "top_values": top_values,
        "histogram": histogram,
    }


def _is_demographic(
    column: str,
    variable_dictionary: Mapping[str, Mapping[str, str]],
) -> bool:
    item = dict(variable_dictionary.get(column) or {})
    combined = " ".join(
        [
            column,
            str(item.get("label") or ""),
            str(item.get("category") or ""),
        ]
    ).lower()
    return any(token in combined for token in DEMOGRAPHIC_TOKENS)


def _matches_keyword(
    column: str,
    keyword: str,
    variable_dictionary: Mapping[str, Mapping[str, str]],
) -> bool:
    value = str(keyword or "").strip().lower()
    if not value:
        return True
    item = dict(variable_dictionary.get(column) or {})
    haystack = " ".join(
        [
            column,
            str(item.get("label") or ""),
            str(item.get("category") or ""),
            str(item.get("remark") or ""),
        ]
    ).lower()
    return value in haystack


def build_dataset_summary(
    frame: pd.DataFrame,
    meta: Mapping[str, Any],
    variable_dictionary: Mapping[str, Mapping[str, str]],
    *,
    keyword: str = "",
    page: int = 1,
    page_size: int = 30,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if frame is None or frame.empty:
        raise ValueError("Dataset is empty.")
    page_size = 20 if int(page_size or 30) == 20 else 30
    page = max(1, int(page or 1))
    columns = [str(column) for column in frame.columns]
    profile = profile or build_dataset_profile(
        frame,
        variable_dictionary,
    )
    inferred_types = dict(profile.get("inferred_types") or {})
    type_counts = dict(profile.get("type_counts") or {})
    missing_by_column = dict(profile.get("missing_by_column") or {})
    unique_by_column = dict(profile.get("unique_by_column") or {})
    total_missing = int(profile.get("total_missing") or 0)
    total_cells = int(frame.shape[0] * frame.shape[1])
    demographic = list(profile.get("demographic") or [])
    matched = [
        column
        for column in columns
        if _matches_keyword(column, keyword, variable_dictionary)
    ]
    total_pages = max(1, int(math.ceil(len(matched) / page_size)))
    page = min(page, total_pages)
    start = (page - 1) * page_size
    page_columns = matched[start : start + page_size]
    selected = _unique_ordered(demographic + page_columns)
    field_details = profile.setdefault("field_details", {})
    for column in selected:
        if column not in field_details:
            field_details[column] = summarize_field(
                frame,
                column,
                variable_dictionary,
                inferred_types.get(column, ""),
                missing_n=int(missing_by_column.get(column) or 0),
                unique_count=int(unique_by_column.get(column) or 0),
            )
    details = {column: field_details[column] for column in selected}
    return {
        "dataset": {
            "name": str(meta.get("name") or Path(str(meta.get("path") or "")).name),
            "source": str(meta.get("source") or ""),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "description_row_removed": bool(meta.get("description_row_removed")),
        },
        "quality": {
            "missing_cells": total_missing,
            "missing_pct": float(total_missing / total_cells * 100)
            if total_cells
            else 0.0,
            "columns_with_missing": int(profile.get("columns_with_missing") or 0),
            "constant_columns": int(profile.get("constant_columns") or 0),
            "duplicate_rows": int(profile.get("duplicate_rows") or 0),
        },
        "type_counts": type_counts,
        "demographic_fields": [details[column] for column in demographic],
        "fields": [details[column] for column in page_columns],
        "pagination": {
            "keyword": str(keyword or ""),
            "page": page,
            "page_size": page_size,
            "total_fields": len(matched),
            "total_pages": total_pages,
        },
    }


def build_dataset_profile(
    frame: pd.DataFrame,
    variable_dictionary: Mapping[str, Mapping[str, str]],
    *,
    sample_rows: int = 5000,
) -> Dict[str, Any]:
    if frame is None or frame.empty:
        raise ValueError("Dataset is empty.")
    columns = [str(column) for column in frame.columns]
    if len(frame) > sample_rows:
        sample = frame.sample(sample_rows, random_state=42)
    else:
        sample = frame
    inferred_types = {
        column: infer_variable_type(sample[column])
        for column in columns
    }
    missing_series = frame.isna().sum()
    unique_series = frame.nunique(dropna=True)
    demographic = [
        column
        for column in columns
        if _is_demographic(column, variable_dictionary)
    ]
    order = {column: index for index, column in enumerate(DEMOGRAPHIC_PRIORITY)}
    demographic = sorted(
        demographic,
        key=lambda column: (order.get(column, 999), columns.index(column)),
    )[:24]
    return {
        "inferred_types": inferred_types,
        "type_counts": dict(Counter(inferred_types.values())),
        "missing_by_column": {
            str(column): int(value)
            for column, value in missing_series.items()
        },
        "unique_by_column": {
            str(column): int(value)
            for column, value in unique_series.items()
        },
        "total_missing": int(missing_series.sum()),
        "columns_with_missing": int((missing_series > 0).sum()),
        "constant_columns": int((unique_series <= 1).sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "demographic": demographic,
        "field_details": {},
        "type_inference_sample_rows": int(len(sample)),
    }


def _unique_ordered(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def dataset_summary_html(summary: Mapping[str, Any]) -> str:
    dataset = dict(summary.get("dataset") or {})
    quality = dict(summary.get("quality") or {})
    type_counts = dict(summary.get("type_counts") or {})
    rows = []
    for item in list(summary.get("demographic_fields") or []):
        statistics = dict(item.get("statistics") or {})
        if statistics:
            description = (
                f"mean={statistics.get('mean', 0):.4g}; "
                f"SD={statistics.get('sd', 0):.4g}; "
                f"median={statistics.get('median', 0):.4g}; "
                f"range={statistics.get('min', 0):.4g}–{statistics.get('max', 0):.4g}"
            )
        else:
            description = "; ".join(
                f"{entry.get('value')}: {entry.get('count')}"
                for entry in list(item.get("top_values") or [])[:4]
            )
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(item.get('field') or ''))}</strong><br>"
            f"<small>{html.escape(str(item.get('label') or ''))}</small></td>"
            f"<td>{html.escape(str(item.get('inferred_type') or ''))}</td>"
            f"<td>{int(item.get('missing_n') or 0):,} "
            f"({float(item.get('missing_pct') or 0):.2f}%)</td>"
            f"<td>{html.escape(description)}</td>"
            "</tr>"
        )
    return (
        "<section>"
        "<h3>資料集摘要</h3>"
        f"<p><strong>{html.escape(str(dataset.get('name') or '目前資料'))}</strong>："
        f"{int(dataset.get('rows') or 0):,} 列 × "
        f"{int(dataset.get('columns') or 0):,} 欄。</p>"
        f"<p>缺失資料 {float(quality.get('missing_pct') or 0):.2f}%；"
        f"{int(quality.get('columns_with_missing') or 0)} 欄含缺失，"
        f"{int(quality.get('constant_columns') or 0)} 欄為常數，"
        f"{int(quality.get('duplicate_rows') or 0):,} 列重複。</p>"
        "<p>欄位型態："
        + "、".join(
            f"{html.escape(str(key))} {int(value)}"
            for key, value in sorted(type_counts.items())
        )
        + "。</p>"
        "<table><thead><tr><th>欄位</th><th>推定型態</th>"
        "<th>缺失</th><th>摘要</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )
