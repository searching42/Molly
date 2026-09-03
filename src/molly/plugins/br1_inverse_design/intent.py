"""Deterministic natural-language compiler for the bounded BR1 request."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Mapping

from .errors import Br1Error
from .schema import Br1RunSpec


_INTEGER = r"([0-9]{1,4})"
_WORKSTATION_PREFIX = "workstation"


@dataclass(frozen=True, slots=True)
class Br1Intent:
    """A parsed BR1 request plus safe explanations for the operator UI."""

    spec: Br1RunSpec
    matched_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "matched_fields", tuple(self.matched_fields))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "molly.br1.intent",
            "schema_version": "1",
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.digest,
            "matched_fields": list(self.matched_fields),
            "warnings": list(self.warnings),
        }


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _number_after(text: str, patterns: tuple[str, ...]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


def parse_br1_request(
    goal: str,
    *,
    llm_profile_ref: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Br1Intent:
    """Compile Chinese or English BR1 wording without granting tool authority.

    An optional LLM profile is recorded as a provenance reference only.  This
    compiler remains deterministic so a configured model can explain or refine
    a request later without becoming the source of host paths, commands, or
    execution permissions.
    """

    if not isinstance(goal, str) or not goal.strip() or len(goal) > 8_000 or "\x00" in goal:
        raise Br1Error("BR1 goal must be bounded text")
    text = goal.strip()
    compact = re.sub(r"[\s_\-/]+", "", text.casefold())
    gap_hit = (
        ("homo" in compact and "lumo" in compact)
        or "homo-lumo" in text.casefold()
        or "homo_lumo_gap" in text.casefold()
    )
    quantum_hit = _matches(text, (r"quantum\s*yield", r"量子产率"))
    if gap_hit and quantum_hit:
        raise Br1Error("BR1 target property is ambiguous")
    if gap_hit:
        target = "homo_lumo_gap"
        target_field = "target_property:homo_lumo_gap"
    elif quantum_hit:
        target = "quantum_yield"
        target_field = "target_property:quantum_yield"
    else:
        raise Br1Error("请在任务目标中明确 BR1 target property，例如 HOMO-LUMO gap")

    min_hit = _matches(
        text,
        (r"\bmin(?:imize|imum)?\b", r"\bsmall(?:est|er)?\b", r"\blow(?:est|er)?\b", r"较小", r"越小", r"最小", r"低"),
    )
    max_hit = _matches(
        text,
        (r"\bmax(?:imize|imum)?\b", r"\blargest?\b", r"\bhigh(?:est|er)?\b", r"较大", r"越大", r"最大", r"高"),
    )
    if min_hit and max_hit:
        raise Br1Error("BR1 ranking direction is ambiguous")
    direction = "MIN" if min_hit or target == "homo_lumo_gap" else "MAX"

    candidate_count = _number_after(
        text,
        (
            rf"(?:采样空间|采样数量|候选(?:分子)?数量|sampling\s*space|candidate(?:s)?(?:\s+count)?|sample(?:d)?\s+molecules?)[^0-9]{{0,24}}{_INTEGER}",
            rf"{_INTEGER}[^0-9]{{0,8}}(?:个)?(?:候选分子|候选|分子)",
        ),
    ) or 1000
    top_n = _number_after(
        text,
        (
            rf"\btop\s*[-_ ]?\s*{_INTEGER}",
            rf"前\s*{_INTEGER}\s*(?:个|名|分子)?",
        ),
    ) or 5

    workstation_hits = []
    for number in (1, 2):
        if _matches(text, (r"workstation\s*" + str(number), r"工作站\s*" + str(number))):
            workstation_hits.append(_WORKSTATION_PREFIX + str(number))
    if len(workstation_hits) > 1:
        raise Br1Error("不能同时指定两个工作站")
    host = workstation_hits[0] if workstation_hits else "auto"

    no_scaffold = _matches(
        text,
        (r"no\s+scaffold", r"unrestricted", r"without\s+(?:a\s+)?scaffold", r"不限制骨架", r"不限定骨架", r"不限骨架"),
    )
    scaffold_mentioned = _matches(text, (r"scaffold", r"骨架"))
    if scaffold_mentioned and not no_scaffold:
        raise Br1Error("当前 BR1 入口只支持明确的无骨架限制请求")

    gpu_count = _number_after(text, (rf"(?:gpu|显卡|GPU数量)[^0-9]{{0,12}}{_INTEGER}",))
    cpu_count = _number_after(text, (rf"(?:cpu|线程|CPU线程)[^0-9]{{0,12}}{_INTEGER}",))
    if _matches(text, (r"\bonly\s+cpu\b", r"cpu\s*only", r"仅\s*cpu", r"只用\s*cpu")):
        gpu_count = 0
    if gpu_count is None:
        gpu_count = 1
    if cpu_count is None:
        cpu_count = 8
    seed = _number_after(text, (rf"(?:seed|随机种子)[^0-9]{{0,12}}{_INTEGER}",)) or 42

    values: dict[str, Any] = {
        "target_property": target,
        "direction": direction,
        "candidate_count": candidate_count,
        "top_n": top_n,
        "scaffold_constraint": "NONE",
        "seed": seed,
        "host_preference": host,
        "cpu_threads": cpu_count,
        "gpu_count": gpu_count,
    }
    if llm_profile_ref is not None:
        values["llm_profile_ref"] = llm_profile_ref
    if overrides is not None:
        allowed = {
            "direction",
            "candidate_count",
            "top_n",
            "seed",
            "host_preference",
            "cpu_threads",
            "gpu_count",
            "walltime_sec",
            "llm_profile_ref",
            "source_format",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise Br1Error(f"BR1 request has unsupported override fields: {sorted(unknown)!r}")
        values.update(dict(overrides))
    spec = Br1RunSpec(**values)
    warnings: list[str] = []
    if not _matches(text, (r"采样空间", r"candidate", r"sampling", r"候选")):
        warnings.append("未在原文中找到采样数量，使用默认 1000")
    if not _matches(text, (r"top", r"前\s*[0-9]+")):
        warnings.append("未在原文中找到 Top-N，使用默认 5")
    return Br1Intent(
        spec=spec,
        matched_fields=(target_field, f"direction:{direction}", f"candidate_count:{candidate_count}", f"top_n:{top_n}", f"host:{host}"),
        warnings=tuple(warnings),
    )


def with_source_format(intent: Br1Intent, source_format: str) -> Br1Intent:
    """Bind a server-detected input format without reparsing user wording."""

    return replace(intent, spec=replace(intent.spec, source_format=source_format))


__all__ = ["Br1Intent", "parse_br1_request", "with_source_format"]
