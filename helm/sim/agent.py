"""Scripted reasoner + episode runner for the synthetic benchmark.

The reasoner is deliberately simple and *identical for every memory method*: it reads the recalled items in the order the
backend returns them, takes the first item that asserts the queried key, and answers with its value. The only
behavioural difference between memory systems is therefore what they put in front of the reasoner:

* A backend whose memory block is *annotated* with epistemic metadata (governed HELM variants) lets the reasoner see
  INVALID flags; it skips flagged items and, when nothing valid remains, triggers a fresh tool call (paper: "trigger
  fresh evidence acquisition").
* Unannotated backends hand over raw text, and the reasoner trusts whatever comes first -- the failure mode in the paper.

Metrics are computed mechanically: task success against the world's ground truth, CTU from token counts, and EpiErr from
the backend-independent ``AuditLog``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..backends import MemoryBackend, RecalledItem
from ..config import HelmConfig
from ..memory.claims import normalize_value
from ..memory.tokens import DEFAULT_COUNTER, TokenCounter
from ..metrics import AuditLog, CTUMeter, EpiErrMeter, provenance_faithfulness
from ..types import Observation, ReadContext, SourceType, Status
from .world import Episode

FILLER = ("please kindly confirm whether this is accurate and up to date also note any recent changes mentioned anywhere "
          "in the logs about this item today").split()


@dataclass
class EpisodeMetrics:
    n_queries: int = 0
    n_correct: int = 0
    ctu: int = 0
    n_steps: int = 0
    epi_consumed: int = 0
    epi_invalid: int = 0  # ground-truth staleness (primary)
    epi_invalid_meta: int = 0  # paper's metadata rule (version / age); equals what governance filters on
    epi_stale: int = 0
    epi_failed: int = 0
    epi_contradicted: int = 0
    n_fresh: int = 0  # fresh tool acquisitions triggered at query time
    n_recall_miss: int = 0
    n_recall_tokens: int = 0
    n_interventions: int = 0
    interventions_by_reason: Dict[str, int] = field(default_factory=dict)
    n_fallback: int = 0
    recall_latency_ms: float = 0.0
    n_cited_recalls: int = 0  # recalls produced by the A1/A2 generators
    cite_cov_sum: float = 0.0
    cite_valid_sum: float = 0.0
    cite_prec_sum: float = 0.0
    cite_faith_sum: float = 0.0
    paths: Dict[str, int] = field(default_factory=dict)
    ops: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> float:
        return self.n_correct / self.n_queries if self.n_queries else float("nan")

    @property
    def epierr(self) -> float:
        return 100.0 * self.epi_invalid / self.epi_consumed if self.epi_consumed else float("nan")

    def _cite(self, total: float) -> float:
        return total / self.n_cited_recalls if self.n_cited_recalls else float("nan")

    @property
    def cite_cov(self) -> float:
        return 100.0 * self._cite(self.cite_cov_sum)

    @property
    def cite_valid(self) -> float:
        return 100.0 * self._cite(self.cite_valid_sum)

    @property
    def cite_prec(self) -> float:
        return 100.0 * self._cite(self.cite_prec_sum)

    @property
    def epierr_meta(self) -> float:
        return 100.0 * self.epi_invalid_meta / self.epi_consumed if self.epi_consumed else float("nan")


_CODE = re.compile(r"""["']?code["']?\s*[:=]\s*(\d{3})""")


def _raw_error(text: str) -> bool:
    m = _CODE.search(text)
    return bool(m) and int(m.group(1)) >= 400


def _first_claim(item: RecalledItem, key: str) -> Optional[str]:
    for k, v in item.claims:
        if k == key:
            return v
    return None


class SimAgent:
    def __init__(
        self,
        backend: MemoryBackend,
        cfg: HelmConfig,
        tokens: Optional[TokenCounter] = None,
        max_fresh_attempts: int = 2,
        vigilant: bool = False,
    ):
        self.backend = backend
        self.cfg = cfg
        self.tokens = tokens or DEFAULT_COUNTER
        self.max_fresh = max_fresh_attempts
        # vigilant=True: even without epistemic annotations the reasoner reads the raw return code in a log line and
        # distrusts error traces (a stronger baseline reasoner; HELM's advantage should shrink but not vanish).
        self.vigilant = vigilant
        self.audit = AuditLog(cfg.governance)

    def run(self, ep: Episode) -> EpisodeMetrics:
        be, tok = self.backend, self.tokens
        be.reset_episode()
        self.audit.reset()
        m = EpisodeMetrics()
        ctu, epi, self._epi_meta = CTUMeter(), EpiErrMeter(), EpiErrMeter()
        ops_before = dict(be.op_counts)
        for ev in ep.events:
            ctx = ReadContext(step=ev.step, time=ev.step * 60.0, task_version=ev.version)
            if ev.kind in ("lookup", "distract"):
                obs = self._observation(ev, f"S{ev.step}")
                be.observe(obs, ctx)
                self.audit.add(obs)
                ctu.step(be.prompt_history_tokens(), 0, tok.count(obs.text))
                continue
            self._answer_query(ep, ev, ctx, be, ctu, epi, m)
        m.ctu, m.n_steps = ctu.total, len(ctu.per_step)
        m.epi_consumed, m.epi_invalid = epi.consumed, epi.invalid
        m.epi_invalid_meta = self._epi_meta.invalid
        m.epi_stale, m.epi_failed, m.epi_contradicted = (epi.by_reason.get(r, 0) for r in ("stale", "failed", "contradicted"))
        m.ops = {k: v - ops_before.get(k, 0) for k, v in be.op_counts.items() if v - ops_before.get(k, 0)}
        return m

    # ------------------------------------------------------------------ helpers
    def _observation(self, ev, log_id: str) -> Observation:
        if ev.kind == "lookup":
            return Observation(ev.step, ev.step * 60.0, "lookup", log_id, ev.result.text, ev.result.status,
                               SourceType.TOOL, ev.version)
        return Observation(ev.step, ev.step * 60.0, "search", log_id, ev.text, Status.SUCCESS, SourceType.TOOL, ev.version)

    def _answer_query(self, ep: Episode, ev, ctx: ReadContext, be: MemoryBackend, ctu: CTUMeter, epi: EpiErrMeter, m: EpisodeMetrics) -> None:
        tok = self.tokens
        key = ev.key
        kstr = f"{key[0]}.{key[1]}".lower()
        query = f"current {key[1]} of {key[0]}"
        if ep.cfg.query_extra_tokens:
            query += " " + " ".join((FILLER * 40)[: ep.cfg.query_extra_tokens])
        res = be.recall(query, ctx)
        m.n_queries += 1
        m.n_recall_tokens += res.tokens
        m.recall_latency_ms += res.latency_ms
        m.n_interventions += len(res.interventions)
        for iv in res.interventions:
            m.interventions_by_reason[iv["reason"]] = m.interventions_by_reason.get(iv["reason"], 0) + 1
        m.n_fallback += int(res.fallback_used)
        if res.path:
            m.paths[res.path] = m.paths.get(res.path, 0) + 1
        ctu.step(be.prompt_history_tokens(), res.tokens, 0)
        store = getattr(be, "store", None)
        if store is not None and res.info.get("variant") in ("A1", "A2") and res.text:
            f = provenance_faithfulness(f"<memory>{res.text}</memory>", store.spans)  # lexical-overlap judge (proxy)
            if f["coverage"] == f["coverage"]:  # not NaN
                m.n_cited_recalls += 1
                m.cite_cov_sum += f["coverage"]
                m.cite_valid_sum += 0.0 if f["valid"] != f["valid"] else f["valid"]
                m.cite_prec_sum += 0.0 if f["precision"] != f["precision"] else f["precision"]
                m.cite_faith_sum += 0.0 if f["faithfulness"] != f["faithfulness"] else f["faithfulness"]

        consumed: Optional[RecalledItem] = None
        answer: Optional[str] = None
        for it in res.items:
            val = _first_claim(it, kstr)
            if val is None:
                continue
            if be.annotated and it.flagged_invalid:
                continue  # the memory block marks it INVALID: the reasoner does not rely on it
            if self.vigilant and not be.annotated and _raw_error(it.text):
                continue
            consumed, answer = it, val
            break
        truth = normalize_value(ep.value_at(key, ev.step))
        if consumed is not None:
            epi.add(self.audit.classify(consumed.audit_ids, kstr, answer, ctx, truth=truth))
            self._epi_meta.add(self.audit.classify(consumed.audit_ids, kstr, answer, ctx))
        else:
            m.n_recall_miss += 1
            answer = self._fresh_acquire(ep, ev, ctx, be, ctu, m)
        m.n_correct += int(answer == truth)

    def _fresh_acquire(self, ep: Episode, ev, ctx: ReadContext, be: MemoryBackend, ctu: CTUMeter, m: EpisodeMetrics) -> Optional[str]:
        answer = None
        for attempt in range(self.max_fresh):
            res = ep.lookup(ev.key, ev.step, attempt + 1)
            obs = Observation(ev.step, ev.step * 60.0, "lookup", f"F{ev.step}-{attempt}", res.text, res.status,
                              SourceType.TOOL, ctx.task_version)
            be.observe(obs, ctx)
            self.audit.add(obs)
            m.n_fresh += 1
            ctu.step(be.prompt_history_tokens(), 0, self.tokens.count(obs.text))
            answer = normalize_value(res.payload["value"])
            if res.status != Status.FAILED:
                break
        return answer
