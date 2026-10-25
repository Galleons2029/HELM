"""Semantic-HNSW Nested Memory (SHNM): three layers, one ANN index per layer, provenance links across layers.

    M1  episodic traces        raw tool logs / interaction snippets (verifiable evidence)
    M2  consolidated recalls   per-key distilled facts, linked to the supporting M1 traces
    M3  thematic indices       sparse entry points (one per topic/entity), linked to their M2 recalls

Operations (Appendix, "Explicit memory operations"): Read, Write, Consolidate, Prune. Governance lives in
``governance.py``; the algorithms live in ``retrieve.py`` / ``consolidate.py`` / ``prune.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import HelmConfig
from ..embedding.base import Embedder
from ..index import make_index
from ..types import MemoryItem, Observation, ReadContext, RetrievalResult
from ..types import Claim
from .claims import ClaimRegistry, extract_claims
from .governance import Governor
from .provenance import SpanTable
from .tokens import DEFAULT_COUNTER, TokenCounter


class LayerStore:
    def __init__(self, layer: int, dim: int, cfg: HelmConfig, seed: int = 0):
        m = cfg.memory
        self.layer = layer
        self.items: Dict[str, MemoryItem] = {}
        self.index = make_index(
            m.index_kind, dim, M=m.hnsw_M, ef_construction=m.ef_construction, ef_search=m.ef_search, seed=seed + layer
        )

    def add(self, item: MemoryItem) -> None:
        self.items[item.id] = item
        self.index.add(item.id, item.vec)

    def remove(self, item_id: str) -> Optional[MemoryItem]:
        item = self.items.pop(item_id, None)
        if item is not None:
            self.index.remove(item_id)
        return item

    def search(self, q: np.ndarray, k: int, allowed: Optional[Set[str]] = None) -> List[Tuple[MemoryItem, float]]:
        out = []
        for ext, sim in self.index.search(q, k, allowed=allowed):
            it = self.items.get(ext)
            if it is not None:
                out.append((it, sim))
        return out

    def __len__(self) -> int:
        return len(self.items)


class ExtractiveSummarizer:
    """Deterministic offline summariser. An LLM summariser can be dropped in with the same two methods."""

    def summarize_claim(self, key: str, value: str, step: int, version: Optional[str], ambiguous: bool = False) -> str:
        s = f"{key} = {value} as of step {step}"
        if version:
            s += f" version {version}"
        if ambiguous:
            s += " (ambiguous)"
        return s

    def summarize_theme(self, topic: str, attrs: List[str]) -> str:
        return f"{topic}: " + ", ".join(sorted(attrs))


class SHNM:
    def __init__(
        self,
        cfg: HelmConfig,
        embedder: Embedder,
        tokens: Optional[TokenCounter] = None,
        summarizer: Optional[Any] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.mcfg = cfg.memory
        self.embedder = embedder
        self.tokens = tokens or DEFAULT_COUNTER
        self.summarizer = summarizer or ExtractiveSummarizer()
        self.seed = seed
        self.annotate = cfg.governance.enabled  # governed memory blocks carry epistemic metadata
        self._init_state()

    def _init_state(self) -> None:
        self.registry = ClaimRegistry()
        self.governor = Governor(self.cfg.governance, self.registry)
        self.spans = SpanTable()
        self.layers: Dict[int, LayerStore] = {
            l: LayerStore(l, self.embedder.dim, self.cfg, self.seed) for l in (1, 2, 3)
        }
        self.by_obs: Dict[str, str] = {}
        self.m2_by_key: Dict[str, str] = {}
        self.m3_by_topic: Dict[str, str] = {}
        self.events: List[Dict[str, Any]] = []
        self.pending: List[str] = []  # M1 ids not yet consolidated
        self._writes_at: Dict[int, int] = {}
        self._consolidations = 0
        self._counter = 0

    # ------------------------------------------------------------------ lifecycle
    def new_episode(self, keep_memory: bool = False) -> None:
        """Start an episode. ``keep_memory=False`` clears the store (independent episodes); True keeps it
        (continual / lifelong settings) and only resets the per-episode operation counters."""
        if keep_memory:
            self._consolidations = 0
            self._writes_at = {}
        else:
            self._init_state()

    def _new_id(self, layer: int) -> str:
        self._counter += 1
        return f"m{layer}-{self._counter}"

    def item(self, item_id: str) -> Optional[MemoryItem]:
        for store in self.layers.values():
            it = store.items.get(item_id)
            if it is not None:
                return it
        return None

    def sizes(self) -> Dict[int, int]:
        return {l: len(s) for l, s in self.layers.items()}

    def log(self, op: str, **kw: Any) -> None:
        self.events.append({"op": op, **kw})

    # ------------------------------------------------------------------ Write
    def write(self, obs: Observation) -> Optional[MemoryItem]:
        """Store an episodic trace (M1) with status, timestamp, version and a provenance pointer to the raw log."""
        n = self._writes_at.get(obs.step, 0)
        if n >= self.mcfg.max_writes_per_step:
            self.log("WRITE_SKIPPED", step=obs.step, reason="max_writes_per_step")
            return None
        self._writes_at[obs.step] = n + 1
        fields = obs.fields or {"text": obs.text}
        if "text" not in fields:
            fields = {**fields, "text": obs.text}
        self.spans.add_log(obs.tool, obs.log_id, obs.step, fields)
        iid = self._new_id(1)
        vec = self.embedder.encode([obs.text])[0]
        claims = [Claim(k, v, obs.step, obs.status, iid) for k, v in extract_claims(obs.text, obs.tool)]
        item = MemoryItem(
            id=iid,
            layer=1,
            text=obs.text,
            vec=vec,
            status=obs.status,
            source=obs.source,
            step_id=obs.step,
            timestamp=obs.timestamp,
            task_version=obs.task_version,
            tool=obs.tool,
            log_id=obs.log_id,
            provenance=[self.spans.pointer(obs.tool, obs.log_id, obs.step, "text")],
            claims=claims,
            error_payload=obs.error_payload,
            audit_ids=[obs.obs_id],
        )
        self.layers[1].add(item)
        for c in claims:
            self.registry.add(c)
        self.by_obs[obs.obs_id] = iid
        self.pending.append(iid)
        self.log("WRITE", step=obs.step, item=iid, status=obs.status.value)
        if len(self.layers[1]) > self.mcfg.budget_m1:
            self.prune(ReadContext(step=obs.step, time=obs.timestamp, task_version=obs.task_version))
        return item

    # ------------------------------------------------------------------ Read / Consolidate / Prune
    def read(self, query: str, ctx: ReadContext, instruction: str = "retrieve valid evidence") -> RetrievalResult:
        from .retrieve import read as _read

        return _read(self, query, ctx, instruction)

    def consolidate(self, ctx: ReadContext) -> List[MemoryItem]:
        from .consolidate import consolidate as _consolidate

        return _consolidate(self, ctx)

    def prune(self, ctx: ReadContext) -> Dict[int, int]:
        from .prune import prune as _prune

        return _prune(self, ctx)

    # ------------------------------------------------------------------ rendering
    def render(self, item: MemoryItem, reasons: Optional[List[str]] = None, readmitted: bool = False) -> str:
        if not self.annotate:
            return item.text
        tag = f"[M{item.layer}|{item.status.value}|step={item.step_id}"
        if item.task_version:
            tag += f"|ver={item.task_version}"
        flags = list(reasons or [])
        if readmitted:
            flags.append("readmitted")
        if flags:
            tag += "|INVALID:" + ",".join(flags)
        return f"{tag}] {item.text}"
