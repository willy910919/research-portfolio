from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


FILTER_OPERATORS = (
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not_in",
    "is_missing",
    "not_missing",
    "literal_equals",
    "literal_not_equals",
)

_OPERATOR_ALIASES = {
    "=": "==",
    "eq": "==",
    "equals": "==",
    "<>": "!=",
    "ne": "!=",
    "not_equals": "!=",
    "notin": "not_in",
    "isna": "is_missing",
    "is_nan": "is_missing",
    "is_null": "is_missing",
    "missing": "is_missing",
    "为空": "is_missing",
    "為空": "is_missing",
    "空值": "is_missing",
    "缺值": "is_missing",
    "notna": "not_missing",
    "not_nan": "not_missing",
    "is_not_null": "not_missing",
    "non_missing": "not_missing",
    "非空": "not_missing",
    "非空值": "not_missing",
    "非缺值": "not_missing",
}

_MISSING_MARKERS = {
    "nan",
    "na",
    "n/a",
    "null",
    "none",
    "missing",
    "missing_value",
    "空值",
    "缺值",
    "缺失值",
    "遺漏值",
}

_MISSING_CUE = (
    r"(?:空白值|空值|缺失值|缺值|遺漏值|沒有值|"
    r"(?<![A-Za-z])nan(?![A-Za-z])|(?<![A-Za-z])null(?![A-Za-z])|"
    r"(?<![A-Za-z])n/?a(?![A-Za-z])|missing(?:\s+values?)?)"
)
_REMOVE_CUE = r"(?:刪除|移除|排除|去除|剔除|丟棄|drop|remove|delete|exclude)"
_KEEP_CUE = r"(?:保留|留下|篩選|找出|顯示|列出|統計|select|filter|find|show|keep)"
_NOT_MISSING_CUE = (
    r"(?:非空白值|非空值|非缺失值|非缺值|有值|"
    r"not\s+(?:null|missing|na|nan)|non[-_\s]?missing)"
)


def is_missing_marker(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_")
        return normalized in _MISSING_MARKERS
    return False


def normalize_filter_operator(operator: Any) -> str:
    normalized = (
        str(operator or "==")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return _OPERATOR_ALIASES.get(normalized, normalized)


def normalize_filter_condition(condition: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(condition, Mapping):
        raise ValueError("Each filter must be an object with a field and operator.")
    result = dict(condition or {})
    result["field"] = str(result.get("field") or "").strip()
    if not result["field"]:
        raise ValueError("Filter field must not be empty.")
    operator = normalize_filter_operator(result.get("operator"))
    if operator not in FILTER_OPERATORS:
        raise ValueError(f"Unsupported filter operator: {operator}")
    if operator not in {"is_missing", "not_missing"} and "value" not in result:
        raise ValueError(f"Filter {operator} requires an explicit value.")
    value = result.get("value")
    if operator in {"in", "not_in"} and not isinstance(value, list):
        raise ValueError(f"Filter {operator} requires a list of values.")
    if operator == "==" and is_missing_marker(value):
        operator = "is_missing"
    elif operator == "!=" and is_missing_marker(value):
        operator = "not_missing"
    result["operator"] = operator
    result["logic"] = str(result.get("logic") or "and").strip().lower()
    if result["logic"] not in {"and", "or"}:
        raise ValueError("Filter logic must be and or or.")
    if operator in {"is_missing", "not_missing"}:
        result["value"] = None
    return result


def normalize_filter_conditions(
    conditions: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if isinstance(conditions, (str, bytes, Mapping)) or conditions is None:
        raise ValueError("Filters must be a list of condition objects.")
    return [normalize_filter_condition(condition) for condition in conditions]


_CLAUSE_BREAK = r"[。；;，,\n]|(?:並且|而且|以及|然後|接著|但|且)|\b(?:and|then|but)\b"
_NEGATED_REMOVE = r"(?:不要|不必|別|不可|不能|無須|勿|do\s+not|don't|never)\s*" + _REMOVE_CUE


def protected_missing_fields(question: str, columns: Sequence[str]) -> set[str]:
    """A prohibition on dropping missing rows must not become a positive filter."""
    protected = set()
    for clause in re.split(_CLAUSE_BREAK, str(question), flags=re.IGNORECASE):
        if not re.search(_NEGATED_REMOVE, clause, re.IGNORECASE):
            continue
        if not re.search(_MISSING_CUE, clause, re.IGNORECASE):
            continue
        for column in columns:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(column))}(?![A-Za-z0-9_])", clause, re.IGNORECASE):
                protected.add(str(column))
    return protected


def extract_missing_filter_intents(
    question: str,
    columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """Extract explicit missing-row intent without guessing a column or action."""

    located: List[tuple[int, Dict[str, Any]]] = []
    for clause_index, clause in enumerate(re.split(_CLAUSE_BREAK, str(question or ""), flags=re.IGNORECASE)):
        mentions = []
        for column in columns:
            match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(column))}(?![A-Za-z0-9_])", clause, re.IGNORECASE)
            if match:
                mentions.append((match.start(), match.end(), str(column)))
        mentions.sort()
        for index, (start, end, column) in enumerate(mentions):
            # Bind a predicate to its own column, never to a neighboring mention.
            if len(mentions) == 1:
                window = clause
            else:
                suffix = clause[start:mentions[index + 1][0] if index + 1 < len(mentions) else len(clause)]
                if re.search(r"(?:與|和|及|、|\band\b)\s*$", suffix, re.IGNORECASE):
                    continue
                if index and re.fullmatch(r"[\s與和及、]+", clause[mentions[index - 1][1]:start]):
                    continue
                window = clause[:mentions[0][0]] + suffix
            if not re.search(_MISSING_CUE, window, flags=re.IGNORECASE):
                continue
            if re.search(_NEGATED_REMOVE, window, flags=re.IGNORECASE):
                continue
            if re.search(
                r"(?:字串|文字|literal).{0,16}" + _MISSING_CUE
                + r"|" + _MISSING_CUE + r".{0,16}(?:字串|文字|literal)",
                window,
                flags=re.IGNORECASE,
            ):
                continue

            positive_nonmissing = bool(re.search(_NOT_MISSING_CUE, window, re.IGNORECASE))
            removal = bool(re.search(
                    _REMOVE_CUE + r".{0,80}" + _MISSING_CUE,
                    window,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    _MISSING_CUE + r".{0,50}" + _REMOVE_CUE,
                    window,
                    flags=re.IGNORECASE,
                ))
            not_missing = bool(
                (positive_nonmissing and not removal)
                or (removal and not positive_nonmissing)
                or re.search(
                    re.escape(column) + r"\s*(?:!=|<>)\s*" + _MISSING_CUE,
                    window,
                    flags=re.IGNORECASE,
                )
            )
            is_missing = bool(
                re.search(
                    re.escape(column)
                    + r".{0,25}(?:==|=|為|是|屬於)?\s*"
                    + _MISSING_CUE,
                    window,
                    flags=re.IGNORECASE,
                )
                or re.search(
                    _KEEP_CUE + r".{0,80}" + _MISSING_CUE,
                    window,
                    flags=re.IGNORECASE,
                )
            )
            if not_missing or is_missing:
                located.append(
                    (
                        clause_index * 10000 + start,
                        {
                            "field": column,
                            "operator": "not_missing" if not_missing else "is_missing",
                            "value": None,
                        },
                    )
                )

    output: List[Dict[str, Any]] = []
    seen = set()
    for _, item in sorted(located, key=lambda entry: entry[0]):
        marker = (item["field"], item["operator"])
        if marker not in seen:
            seen.add(marker)
            output.append(item)
    return output
