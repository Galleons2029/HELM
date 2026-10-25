"""Benchmark plumbing: ``Task`` / ``EpisodeResult``, the adapter protocol, summaries and the CSV results writer.

Provenance rule (enforced by ``ResultsWriter``): every row written to disk carries a ``source`` column.

    ``mock``      a stand-in component was in the loop (``MockLLM``, the offline hashing embedder, a scripted
                  environment). Such rows exercise the pipeline; they are NEVER comparable to the paper.
    ``measured``  a real model client (``OpenAICompatClient``) or a real embedding model produced the run.

``llm_source`` is deliberately conservative: only a client it positively recognises as real is labelled ``measured``;
a ``MockLLM`` -- or anything unknown -- can never be.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..llm.client import MockLLM, OpenAICompatClient
from ..metrics import bootstrap_ci, mean_se

if TYPE_CHECKING:  # pragma: no cover
    from .harness import Agent

SOURCES = ("mock", "measured")


class BenchmarkError(RuntimeError):
    """Raised for user-fixable problems (missing data, missing optional dependency); the CLI prints it without a stack."""


class MissingDataError(BenchmarkError):
    pass


class MissingDependencyError(BenchmarkError):
    pass


# --------------------------------------------------------------------------------------------- data classes
@dataclass
class Task:
    """One benchmark item. ``prompt`` is what the agent is asked; ``gold`` is benchmark-specific (never shown)."""

    task_id: str
    prompt: str
    gold: Any = None
    split: str = ""
    version: Optional[str] = None  # task_version stamped on observations (environment snapshot / tool version)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        return self.prompt


@dataclass
class EpisodeResult:
    """Outcome of one episode. ``epi_err`` is a percentage of consumed recalls (NaN when nothing was recalled); the raw
    counts live in ``extra`` (``epi_consumed`` / ``epi_invalid``) so rates can be pooled across episodes correctly."""

    score: float
    success: bool
    ctu: int = 0
    epi_err: float = float("nan")
    steps: int = 0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    split: str = ""


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]: ...

    def run_episode(self, task: Task, agent: "Agent") -> EpisodeResult: ...

    def score(self, task: Task, prediction: Any) -> float: ...


class BaseAdapter:
    """Shared defaults. Sequential benchmarks (StreamBench, LifelongAgentBench) override ``run_all``."""

    name = "base"

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        raise NotImplementedError

    def run_episode(self, task: Task, agent: "Agent") -> EpisodeResult:
        raise NotImplementedError

    def score(self, task: Task, prediction: Any) -> float:
        raise NotImplementedError

    def run_all(self, tasks: Sequence[Task], agent: "Agent") -> List[EpisodeResult]:
        return [self.run_episode(t, agent) for t in tasks]

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        """Rows of summary metrics (one per reporting group); the CLI adds benchmark/method/seed/source columns."""
        return [summarize(results, seed=seed)]


# --------------------------------------------------------------------------------------------- provenance labels
def llm_source(llm: Any) -> str:
    """``measured`` only for a positively-identified real client; ``mock`` for MockLLM and for anything unrecognised."""
    if isinstance(llm, MockLLM) or getattr(llm, "is_mock", False):
        return "mock"
    if isinstance(llm, OpenAICompatClient) or getattr(llm, "is_real_llm", False) is True:
        return "measured"
    return "mock"


def embedder_source(embedder: Any) -> str:
    """Retrieval-only benchmarks have no LLM: the offline hashing embedder is a stand-in (``mock``)."""
    name = type(embedder).__name__
    if name == "SentenceTransformerEmbedder":
        return "measured"
    return "mock"


# --------------------------------------------------------------------------------------------- summaries
def _nanmean(xs: Iterable[float]) -> float:
    v = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def pooled_epi_err(results: Sequence[EpisodeResult]) -> float:
    """EpiErr pooled over episodes: invalid recalled items / consumed recalled items, in percent."""
    consumed = sum(int(r.extra.get("epi_consumed", 0)) for r in results)
    invalid = sum(int(r.extra.get("epi_invalid", 0)) for r in results)
    return 100.0 * invalid / consumed if consumed else float("nan")


def summarize(results: Sequence[EpisodeResult], seed: int = 0, n_boot: int = 1000) -> Dict[str, Any]:
    """Aggregate episode results: mean score with a bootstrap CI over episodes, CTU per task / per step, pooled EpiErr."""
    scores = [float(r.score) for r in results]
    mean, lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
    steps = sum(r.steps for r in results)
    ctu_total = sum(r.ctu for r in results)
    return {
        "n": len(results),
        "score": mean,
        "score_ci_lo": lo,
        "score_ci_hi": hi,
        "success_rate": _nanmean(1.0 if r.success else 0.0 for r in results),
        "ctu_per_task": _nanmean(float(r.ctu) for r in results),
        "ctu_per_step": ctu_total / steps if steps else float("nan"),
        "epi_err_pct": pooled_epi_err(results),
        "steps_per_task": _nanmean(float(r.steps) for r in results),
    }


def aggregate_over_seeds(per_seed: Sequence[Mapping[str, Any]], keys: Sequence[str] = ("score", "success_rate", "ctu_per_task", "epi_err_pct")) -> Dict[str, Any]:
    """Mean and standard error across seeds (paper: "mean and standard error across seeds")."""
    out: Dict[str, Any] = {"n_seeds": len(per_seed)}
    for k in keys:
        vals = [float(r[k]) for r in per_seed if k in r]
        m, se = mean_se(vals)
        out[k] = m
        out[k + "_se"] = se
    return out


# --------------------------------------------------------------------------------------------- results writer
_LEAD = ("benchmark", "method", "seed", "split", "source")


class ResultsWriter:
    """Append rows to a CSV. Refuses any row without a valid ``source`` column; grows the header if new keys appear."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    # ---- helpers
    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return "nan" if math.isnan(v) else repr(v)
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, sort_keys=True, default=str)
        if v is None:
            return ""
        return str(v)

    def _check(self, row: Mapping[str, Any]) -> Dict[str, str]:
        src = row.get("source")
        if src not in SOURCES:
            raise ValueError(f"every result row needs source in {SOURCES}; got {src!r}")
        return {k: self._fmt(v) for k, v in row.items()}

    def _read(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            return [], []
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            rows = list(reader)
        if "source" not in header:
            raise ValueError(f"{self.path} exists but has no 'source' column; refusing to mix it with labelled rows")
        return header, rows

    # ---- api
    def append(self, row: Mapping[str, Any]) -> None:
        self.append_many([row])

    def append_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        new = [self._check(r) for r in rows]
        if not new:
            return
        header, old = self._read()
        grown = list(header)
        for r in new:
            for k in r:
                if k not in grown:
                    grown.append(k)
        if not header:
            lead = [k for k in _LEAD if k in grown]
            grown = lead + [k for k in grown if k not in lead]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if grown == header:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=grown, restval="").writerows(new)
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:  # schema grew (or new file): rewrite
            w = csv.DictWriter(f, fieldnames=grown, restval="")
            w.writeheader()
            w.writerows(old)
            w.writerows(new)

    def read_rows(self) -> List[Dict[str, str]]:
        return self._read()[1]


def episode_row(res: EpisodeResult, **ident: Any) -> Dict[str, Any]:
    """Flat per-episode row (scalars of ``extra`` are inlined under ``x_*``; the rest stays in ``extra_json``)."""
    row: Dict[str, Any] = dict(ident)
    row.update(
        task_id=res.task_id,
        split=res.split or ident.get("split", ""),
        score=res.score,
        success=int(bool(res.success)),
        ctu=res.ctu,
        epi_err_pct=res.epi_err,
        steps=res.steps,
    )
    nested = {}
    for k, v in res.extra.items():
        if k == "limits":  # identical for every episode of a run; it is in the printed config and cfg_hash
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            row["x_" + k] = v
        else:
            nested[k] = v
    if nested:
        row["extra_json"] = nested
    return row
