"""Core data types shared by the SHNM store, governance, baselines, controller and simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Status(str, Enum):
    """Tool-execution status derived from tool wrappers / return codes."""

    SUCCESS = "SUCCESS"
    UNKNOWN = "UNKNOWN"
    FAILED = "FAILED"

    @classmethod
    def from_code(cls, code: Optional[int]) -> "Status":
        """ASSUMPTION: HTTP-style return codes. 200/201 -> SUCCESS, missing/204/3xx -> UNKNOWN, else FAILED."""
        if code is None:
            return cls.UNKNOWN
        if code in (200, 201):
            return cls.SUCCESS
        if code == 204 or 300 <= code < 400:
            return cls.UNKNOWN
        return cls.FAILED


class SourceType(str, Enum):
    TOOL = "tool"
    USER = "user"
    REASONER = "reasoner"
    SUMMARY = "summary"  # consolidated recall (M2)
    INDEX = "index"  # thematic index entry (M3)


@dataclass(frozen=True)
class ProvPointer:
    """(tool, log_id, step_id, field, start, end, hash): points at a character span in a raw episodic log."""

    tool: str
    log_id: str
    step_id: int
    field: str
    start: int
    end: int
    hash: str

    def as_cite(self) -> str:
        return (
            f"[[CITE tool={self.tool} log={self.log_id} step={self.step_id} "
            f"field={self.field} start={self.start} end={self.end}]]"
        )


@dataclass(frozen=True)
class Claim:
    """An extracted (key, value) assertion with the evidence it came from."""

    key: str
    value: str
    step_id: int
    status: Status
    item_id: str = ""


@dataclass
class Observation:
    """One logged tool/environment observation at step ``step``."""

    step: int
    timestamp: float
    tool: str
    log_id: str
    text: str
    status: Status = Status.UNKNOWN
    source: SourceType = SourceType.TOOL
    task_version: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)
    error_payload: Optional[str] = None

    @property
    def obs_id(self) -> str:
        return f"{self.tool}:{self.log_id}"


@dataclass
class MemoryItem:
    """m = (x, v, E, P). Epistemic state E is *computed* at read time from the raw metadata kept here."""

    id: str
    layer: int
    text: str
    vec: Optional[np.ndarray] = field(repr=False)
    status: Status
    source: SourceType
    step_id: int
    timestamp: float
    task_version: Optional[str]
    tool: str = ""
    log_id: str = ""
    provenance: List[ProvPointer] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)  # ids in the next-higher layer
    children: List[str] = field(default_factory=list)  # ids in the next-lower layer
    claims: List[Claim] = field(default_factory=list)
    conflicts: List[Claim] = field(default_factory=list)  # older/alternative values recorded by Consolidate
    error_payload: Optional[str] = None
    hits: int = 0  # access count, used by the Prune utility
    audit_ids: List[str] = field(default_factory=list)  # observation ids this item derives from (evaluation only)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadContext:
    """Where the agent currently is: used for staleness, recency and version checks."""

    step: int
    time: float = 0.0
    task_version: Optional[str] = None
    diagnostic: bool = False


@dataclass
class ScoredItem:
    item: MemoryItem
    score: float
    sim: float
    valid: bool = True
    reasons: List[str] = field(default_factory=list)  # subset of {"failed","stale","contradicted"}
    readmitted: bool = False  # invalid item let back in by the soft fallback


@dataclass
class RetrievalResult:
    items: List[ScoredItem] = field(default_factory=list)
    grounding: List[str] = field(default_factory=list)  # supporting M1 span texts from provenance expansion
    text: str = ""  # rendered <memory> payload (without tags)
    tokens: int = 0
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    path: str = ""  # e.g. "M3>M2>M1", "M3>M2", "M1"
    latency_ms: float = 0.0
