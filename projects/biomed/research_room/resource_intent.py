"""Deterministic, clause-scoped resource requests and explicit prohibitions."""
from __future__ import annotations

import re
from typing import Any, Mapping

RESOURCE_PATTERNS = {
    "literature": r"文獻|方法學|論文|PubMed|Europe\s*PMC|PMID|DOI|相關研究|研究佐證|研究支持|參考來源|\b(?:literature(?:_researcher)?|papers?|references?|methodology|evidence)\b",
    "external": r"官方實作|外部(?:資料|資訊|資源)?|GitHub|PyPI|外掛|套件|\b(?:external(?:\s+(?:sources?|data|resources?))?|official\s+implementation|plugins?|packages?)\b",
    "custom_code": r"受控產碼|自訂程式|生成程式|\b(?:custom\s+code|code\s*generation)\b",
}
LITERATURE_OUTPUTS = {"pmid_list", "citation_list", "literature_evidence", "evidence_limitation_summary", "methodology_evidence"}
_ACTIONS = r"查找|查詢|搜尋|檢索|引用|使用|召集|邀請|交給|生成|產生|查|請|讓|\b(?:search|retrieve|consult|include|use|invite|convene|ask|generate|look\s+up)\b"
_AFFIRMATIVE_NEGATION = r"(?:不要|別|勿|不應|不可)(?:忽略|忽視|漏掉|忘記)|(?:do\s+not|don't|don’t|never)\s+(?:ignore|omit|forget|exclude)\b|(?:不只|不僅|不要只|不要僅)|\bnot\s+(?:only|just)\b"
_NEGATIVE = (
    r"(?:不要|不用|無需|不必|不需要|不希望|不想|不能|不得|禁止|不准|勿|別|避免|不允許)(?:\s*(?:再|去|進行|任何|" + _ACTIONS + r"))*"
    r"|不(?:再)?(?:" + _ACTIONS + r")"
    r"|\b(?:do\s+not|don't|don’t|must\s+not|never|avoid|skip|without|no|not(?!\s+(?:only|just)\b))\b(?:\s+(?:want|need|wish|plan)\s+to)?(?:\s+please)?(?:\s+(?:" + _ACTIONS + r"))*"
)
_DIRECTIVE = re.compile(r"(?P<affirm>" + _AFFIRMATIVE_NEGATION + r")|(?P<negative>" + _NEGATIVE + r")|(?P<positive>" + _ACTIONS + r")", re.I)
_CLAUSE = re.compile(r"[,，;；。\n!?！？]|\b(?:but|however)\b|但是|但|不過", re.I)
_CONVENE = re.compile(r"召集|邀請|交給|請(?:讓|由)?|讓|\b(?:invite|ask|convene|engage|involve)\b", re.I)


def _clauses(question):
    # Conversation context appended by the application is not a new instruction.
    text = str(question or "")
    for marker in ("[可驗證的對話脈絡]", "[Secretary advisory; not user authority]"):
        text = text.split(marker, 1)[0]
    return [part.strip() for part in _CLAUSE.split(text) if part.strip()]


def _state_before(clause, start):
    directives = list(_DIRECTIVE.finditer(clause[:start]))
    if not directives:
        return None
    return "prohibited" if directives[-1].lastgroup == "negative" else "requested"


def parse_resource_intents(question: str) -> dict[str, dict[str, Any]]:
    """Return requested/prohibited/unmentioned with the supporting clauses.

    Directives bind within a clause and carry through a resource list. A later
    explicit directive for the same resource supersedes an earlier one.
    """
    result = {resource: {"state": "unmentioned", "spans": []} for resource in RESOURCE_PATTERNS}
    carry = None
    for clause in _clauses(question):
        mentions = sorted((m.start(), m.end(), resource) for resource, pattern in RESOURCE_PATTERNS.items()
                          for m in re.finditer(pattern, clause, re.I))
        if not mentions:
            carry = None
            continue
        tail = re.sub(r"^\s*(?:也|都|一律)\s*", "", clause[mentions[-1][1]:])
        trailing = _DIRECTIVE.match(tail.strip())
        trailing_state = "prohibited" if trailing and trailing.lastgroup == "negative" else None
        if re.match(r"\s*(?:is|are)\s+(?:not\s+(?:required|needed|allowed)|forbidden|unnecessary)\b", tail, re.I):
            trailing_state = "prohibited"
        first_prefix = clause[:mentions[0][0]].strip()
        list_continuation = bool(re.fullmatch(r"(?:(?:和|或|與|及|以及)|(?:and|or))?", first_prefix, re.I))
        for start, end, resource in mentions:
            state = _state_before(clause, start) or trailing_state or (carry if list_continuation else None) or "requested"
            result[resource]["state"] = state
            result[resource]["spans"].append({"text": clause, "mention": clause[start:end], "state": state})
        carry = result[mentions[-1][2]]["state"]
    return result


def resource_prohibited(question: str, plan: Mapping[str, Any], resource: str) -> bool:
    """Check original user text at dispatch, even if a model rewrote the plan."""
    state = parse_resource_intents(question)[resource]["state"]
    if state != "unmentioned":
        return state == "prohibited"
    requirement = dict(plan.get("user_requirement") or {})
    return resource in set(requirement.get("prohibited_resources") or [])


def tool_resource(tool_id: str, network_access: bool = False) -> str:
    if tool_id in {"pubmed_connector_search", "europe_pmc_connector_search"}:
        return "literature"
    return "external" if network_access else ""


def enforce_resource_constraints(question: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(plan)
    prohibited = {resource for resource in RESOURCE_PATTERNS if resource_prohibited(question, plan, resource)}
    result["resource_constraints"] = {"prohibited": sorted(prohibited), "authority": "user_original_text"}
    settings = {"literature": ("needs_literature", "literature_queries"),
                "external": ("needs_external_information", "external_information_queries"),
                "custom_code": ("needs_code", "code_requests")}
    aliases = {"literature": {"literature", "literature_researcher", "文獻", "文獻研究員"},
               "external": {"external", "external_researcher", "外部資源", "外部資源研究員"}}
    blocked_roles = set().union(*(aliases.get(resource, set()) for resource in prohibited))
    for resource in prohibited:
        for key in settings[resource]:
            result[key] = False if key.startswith("needs_") else []
    if blocked_roles:
        result["delegations"] = [item for item in result.get("delegations") or []
                                 if str(item.get("agent") or item.get("role") or "").lower() not in blocked_roles]
    if "literature" in prohibited:
        result["evidence_contract"] = {"required_outputs": [], "sources": [], "citation_policy": "prohibited", "status": "not_required"}
    return result


def requested_collaborator_roles(question, aliases, allowed_roles):
    """Require an affirmative invitation attached to a role, not a mention."""
    roles = []
    policy = parse_resource_intents(question)
    for clause in _clauses(question):
        matches = sorted((m.start(), -len(alias), role) for alias, role in aliases.items()
                         if alias and role in allowed_roles
                         for m in re.finditer(re.escape(alias), clause, re.I))
        if not matches:
            continue
        first = matches[0][0]
        invitations = list(_CONVENE.finditer(clause[:first]))
        if not invitations:
            continue
        # Plain "請查文獻" does not explicitly invite the literature worker.
        between = clause[invitations[-1].end():first].strip()
        if between and not re.fullmatch(r"(?:以下|下列|我們的|the|our)\s*", between, re.I):
            continue
        for start, _, role in matches:
            if _state_before(clause, start) == "prohibited":
                continue
            if role in policy and policy[role]["state"] == "prohibited":
                continue
            if role not in roles:
                roles.append(role)
    return roles
