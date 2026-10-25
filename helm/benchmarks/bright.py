"""BRIGHT-style retrieval evaluation: nDCG@10 / Recall@10 in pure numpy, a BM25 baseline, a dense retriever over any
``Embedder`` and an SHNM-backed retriever.

Data: local JSON, either one file ``{"queries": [...], "documents": [...]}`` or a directory holding ``examples``/
``queries`` and ``documents`` as .json/.jsonl. Query rows: ``id``, ``query`` (or ``text``), ``gold_ids`` (the BRIGHT
``gold_ids_long`` variant with ``use_long=True``), optional ``excluded_ids`` (BRIGHT removes these from the ranking; the
placeholder ``"N/A"`` means none). Document rows: ``id`` and ``content`` (or ``text``). The field names follow
the BRIGHT dataset card (``xlangai/BRIGHT``): check them against a real download.

Metrics use binary relevance (every gold id has gain 1): nDCG@k = DCG / IDCG with a 1/log2(rank+1) discount and IDCG over
min(|gold|, k) ideal hits; Recall@k = |top-k intersect gold| / |gold|. Queries without gold ids are skipped.

Retrievers implemented: ``BM25Retriever`` (Okapi BM25, Lucene-style idf, lowercase alphanumeric tokens, small stopword
list, NO stemming, so it is weaker than the Pyserini BM25 usually reported), ``DenseRetriever(embedder)`` and
``SHNMRetriever`` (documents are written to an SHNM store as episodic traces and ranked with ``SHNM.read``). Not
implemented: ReasonIR-style retrievers. Caveat for SHNM: plain documents carry no ``key = value`` claims, so Consolidate
creates no M2/M3 entries and, with governance signals that are meaningless on a static corpus switched off (freshness,
consistency), ``read`` reduces to an HNSW search over M1. Whatever hierarchy the paper's BRIGHT run used (e.g. an LLM-built
one) is not reproduced here.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from ..config import HelmConfig
from ..embedding.base import Embedder
from ..memory.shnm import SHNM
from ..types import Observation, ReadContext, Status
from .base import BaseAdapter, EpisodeResult, MissingDataError, Task, summarize

# --------------------------------------------------------------------------------------------- metrics
def ndcg_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    top = list(ranked[:k])
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rel = np.array([1.0 if d in gold else 0.0 for d in top])
    dcg = float((rel * disc[: len(top)]).sum())
    idcg = float(disc[: min(len(gold), k)].sum())
    return dcg / idcg


def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    return len(set(ranked[:k]) & gold) / len(gold)


# --------------------------------------------------------------------------------------------- data
@dataclass
class RetrievalQuery:
    qid: str
    text: str
    gold_ids: Set[str]
    excluded_ids: Set[str] = field(default_factory=set)


@dataclass
class RetrievalData:
    doc_ids: List[str]
    doc_texts: List[str]
    queries: List[RetrievalQuery]


def _rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise MissingDataError(f"{path}: expected a JSON list")
    return data


def _find(d: Path, stems: Sequence[str]) -> Optional[Path]:
    for s in stems:
        for ext in (".json", ".jsonl"):
            if (d / (s + ext)).is_file():
                return d / (s + ext)
    return None


def parse_retrieval(queries: Sequence[Dict[str, Any]], documents: Sequence[Dict[str, Any]], use_long: bool = False) -> RetrievalData:
    ids, texts = [], []
    for i, d in enumerate(documents):
        did = d.get("id", d.get("doc_id"))
        txt = d.get("content", d.get("text"))
        if did is None or txt is None:
            raise MissingDataError(f"document {i} needs 'id' and 'content'/'text' (keys: {sorted(d)})")
        ids.append(str(did))
        texts.append(str(txt))
    known = set(ids)
    qs: List[RetrievalQuery] = []
    gold_key = "gold_ids_long" if use_long else "gold_ids"
    for i, q in enumerate(queries):
        text = q.get("query", q.get("text"))
        if text is None:
            raise MissingDataError(f"query {i} needs 'query'/'text' (keys: {sorted(q)})")
        gold = {str(g) for g in q.get(gold_key, q.get("gold_ids", []))}
        missing = gold - known
        if missing:
            raise MissingDataError(f"query {q.get('id', i)}: gold ids not in the corpus: {sorted(missing)[:3]}")
        excl = {str(e) for e in q.get("excluded_ids", []) if str(e) != "N/A"}
        qs.append(RetrievalQuery(str(q.get("id", i)), str(text), gold, excl))
    if not qs or not ids:
        raise MissingDataError("retrieval data needs at least one query and one document")
    return RetrievalData(ids, texts, qs)


def load_bright(path: str | Path, limit: Optional[int] = None, use_long: bool = False) -> RetrievalData:
    p = Path(path)
    if not p.exists():
        raise MissingDataError(f"BRIGHT data path does not exist: {p}")
    if p.is_file():
        blob = json.loads(p.read_text(encoding="utf-8"))
        if not (isinstance(blob, dict) and "queries" in blob and "documents" in blob):
            raise MissingDataError(f"{p}: expected an object with 'queries' and 'documents'")
        queries, docs = blob["queries"], blob["documents"]
    else:
        qf, df = _find(p, ("examples", "queries")), _find(p, ("documents", "corpus"))
        if qf is None or df is None:
            raise MissingDataError(f"{p}: need examples/queries and documents (.json or .jsonl) in the directory")
        queries, docs = _rows(qf), _rows(df)
    data = parse_retrieval(queries[:limit] if limit is not None else queries, docs, use_long)
    return data


# --------------------------------------------------------------------------------------------- retrievers
class Retriever(Protocol):
    name: str

    def index(self, doc_ids: Sequence[str], doc_texts: Sequence[str]) -> None: ...

    def rank(self, query: str, k: int, exclude: Set[str] = frozenset()) -> List[str]: ...


_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or such that the their then there these "
    "they this to was were will with you your i we he she not no do does did can could should would".split()
)


def tokenize(text: str, stopwords: bool = True) -> List[str]:
    toks = _TOK.findall(text.lower())
    return [t for t in toks if t not in _STOP] if stopwords else toks


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, stopwords: bool = True):
        self.k1, self.b, self.stopwords = k1, b, stopwords
        self.ids: List[str] = []

    def index(self, doc_ids: Sequence[str], doc_texts: Sequence[str]) -> None:
        self.ids = list(doc_ids)
        n = len(self.ids)
        toks = [tokenize(t, self.stopwords) for t in doc_texts]
        self.dl = np.array([len(t) for t in toks], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if n and self.dl.sum() else 1.0
        postings: Dict[str, Tuple[List[int], List[int]]] = {}
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                lst = postings.setdefault(w, ([], []))
                lst[0].append(i)
                lst[1].append(c)
        self.post = {w: (np.array(ix), np.array(tf, dtype=np.float64)) for w, (ix, tf) in postings.items()}
        self.idf = {w: math.log(1.0 + (n - len(ix) + 0.5) / (len(ix) + 0.5)) for w, (ix, _) in self.post.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids))
        for w in tokenize(query, self.stopwords):  # repeated query terms count repeatedly (as Lucene's clause sum)
            if w not in self.post:
                continue
            ix, tf = self.post[w]
            norm = self.k1 * (1.0 - self.b + self.b * self.dl[ix] / self.avgdl)
            s[ix] += self.idf[w] * tf * (self.k1 + 1.0) / (tf + norm)
        return s

    def rank(self, query: str, k: int, exclude: Set[str] = frozenset()) -> List[str]:
        return _top(self.ids, self.scores(query), k, exclude)


def _top(ids: Sequence[str], scores: np.ndarray, k: int, exclude: Set[str]) -> List[str]:
    order = np.lexsort((np.arange(len(ids)), -scores))  # stable: ties broken by corpus order
    out: List[str] = []
    for j in order:
        if ids[j] in exclude:
            continue
        out.append(ids[j])
        if len(out) >= k:
            break
    return out


class _BatchedEmbedder:
    """Encode the corpus in batches once, then serve per-document ``encode`` calls from the cache (SHNM writes one at a time)."""

    def __init__(self, inner: Embedder, texts: Sequence[str], batch: int = 32):
        self.inner, self.dim = inner, inner.dim
        self._cache: Dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(texts))
        for i in range(0, len(uniq), batch):
            chunk = uniq[i : i + batch]
            for t, v in zip(chunk, inner.encode(chunk)):
                self._cache[t] = v

    def encode(self, texts: Sequence[str], instruction: Optional[str] = None) -> np.ndarray:
        if instruction is None and all(t in self._cache for t in texts):
            return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return self.inner.encode(texts, instruction=instruction)


QUERY_INSTRUCTION = "Given a question, retrieve documents that help answer it"


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, instruction: str = QUERY_INSTRUCTION, batch: int = 32):
        self.embedder, self.instruction, self.batch = embedder, instruction, batch
        self.ids: List[str] = []

    def index(self, doc_ids: Sequence[str], doc_texts: Sequence[str]) -> None:
        self.ids = list(doc_ids)
        mats = [self.embedder.encode(list(doc_texts[i : i + self.batch])) for i in range(0, len(doc_texts), self.batch)]
        self.mat = np.concatenate(mats, axis=0) if mats else np.zeros((0, self.embedder.dim), np.float32)

    def rank(self, query: str, k: int, exclude: Set[str] = frozenset()) -> List[str]:
        q = self.embedder.encode([query], instruction=self.instruction)[0]
        return _top(self.ids, self.mat @ q, k, exclude)


def variant_overrides(variant: str) -> Dict[str, Any]:
    """Config overrides of a HELM variant from ``helm.methods`` (so ablations can be run as retrievers too)."""
    from ..methods import _HELM_VARIANTS

    if variant not in _HELM_VARIANTS:
        raise KeyError(f"unknown HELM variant '{variant}'; choose from {sorted(_HELM_VARIANTS)}")
    return dict(_HELM_VARIANTS[variant][1])


class SHNMRetriever:
    """Documents -> SHNM M1 traces (status SUCCESS, one write per step); ranking = ``SHNM.read`` restricted to M1 items.

    ``depth`` is how many items ``read`` returns (it must cover k plus the largest exclusion set).
    """

    name = "shnm"

    def __init__(self, embedder: Embedder, cfg: Optional[HelmConfig] = None, variant: str = "helm", depth: int = 100, seed: int = 0):
        self.embedder, self.base_cfg, self.variant, self.depth, self.seed = embedder, cfg or HelmConfig(), variant, depth, seed
        self.store: Optional[SHNM] = None
        self._step = 0

    def index(self, doc_ids: Sequence[str], doc_texts: Sequence[str]) -> None:
        n = len(doc_ids)
        cfg = self.base_cfg.override(**variant_overrides(self.variant)).override(
            **{
                "memory.budget_m1": n + 1,  # corpus-sized store: nothing may be pruned
                "memory.k1": max(self.depth, 1),
                "memory.pool_mult": 1,
                "memory.max_recall_tokens": 10**9,  # rank by score, not by the agent-time token cap
                "governance.use_freshness": False,  # a static corpus has no notion of age or recency
                "governance.use_consistency": False,  # free text must not be mined for key=value contradictions
            }
        )
        emb = _BatchedEmbedder(self.embedder, doc_texts)
        self.store = SHNM(cfg, emb, seed=self.seed)
        for i, (did, text) in enumerate(zip(doc_ids, doc_texts), 1):
            obs = Observation(step=i, timestamp=1.7e9 + i, tool="doc", log_id=str(did), text=text, status=Status.SUCCESS)
            self.store.write(obs)
        self._step = n + 1

    def rank(self, query: str, k: int, exclude: Set[str] = frozenset()) -> List[str]:
        if self.store is None:
            raise RuntimeError("call index() first")
        res = self.store.read(query, ReadContext(step=self._step, time=1.7e9 + self._step))
        ids = [s.item.log_id for s in res.items if s.item.layer == 1 and s.item.log_id not in exclude]
        return ids[:k]


# --------------------------------------------------------------------------------------------- evaluation / adapter
def evaluate_retrieval(data: RetrievalData, retriever: Retriever, k: int = 10, index: bool = True) -> Dict[str, Any]:
    """Index (optionally) and score every query; returns means plus the per-query lists (for bootstrap CIs)."""
    if index:
        retriever.index(data.doc_ids, data.doc_texts)
    nd, rc = [], []
    for q in data.queries:
        ranked = retriever.rank(q.text, k, q.excluded_ids)
        nd.append(ndcg_at_k(ranked, q.gold_ids, k))
        rc.append(recall_at_k(ranked, q.gold_ids, k))
    ok = [i for i, x in enumerate(nd) if not math.isnan(x)]
    return {
        f"ndcg@{k}": float(np.mean([nd[i] for i in ok])) if ok else float("nan"),
        f"recall@{k}": float(np.mean([rc[i] for i in ok])) if ok else float("nan"),
        "n_queries": len(ok),
        "per_query_ndcg": nd,
        "per_query_recall": rc,
    }


class BrightAdapter(BaseAdapter):
    """Each query is a ``Task``; the 'agent' handed to ``run_episode`` / ``run_all`` is a ``Retriever``."""

    name = "bright"

    def __init__(self, k: int = 10, use_long: bool = False):
        self.k, self.use_long = k, use_long
        self.data: Optional[RetrievalData] = None

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("BRIGHT needs --data PATH to a local json (queries + documents), see bright.py")
        self.data = load_bright(path, limit, self.use_long)
        return [
            Task(q.qid, q.text, gold=q.gold_ids, split=split or "", meta={"excluded": q.excluded_ids})
            for q in self.data.queries
        ]

    def score(self, task: Task, prediction: Sequence[str]) -> float:
        return ndcg_at_k(prediction, task.gold, self.k)

    def run_episode(self, task: Task, retriever: Any) -> EpisodeResult:
        ranked = retriever.rank(task.prompt, self.k, task.meta.get("excluded", set()))
        nd, rc = ndcg_at_k(ranked, task.gold, self.k), recall_at_k(ranked, task.gold, self.k)
        return EpisodeResult(
            score=nd, success=rc > 0, task_id=task.task_id, split=task.split,
            extra={f"recall@{self.k}": rc, f"ndcg@{self.k}": nd, "ranked": ranked},
        )

    def run_all(self, tasks: Sequence[Task], retriever: Any) -> List[EpisodeResult]:
        if self.data is None:
            raise MissingDataError("call load() before run_all()")
        retriever.index(self.data.doc_ids, self.data.doc_texts)
        return [self.run_episode(t, retriever) for t in tasks]

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        row = summarize([r for r in results if not math.isnan(r.score)], seed=seed)
        key = f"recall@{self.k}"
        vals = [r.extra[key] for r in results if not math.isnan(r.score)]
        row[f"ndcg@{self.k}"] = row["score"]
        row[key] = float(np.mean(vals)) if vals else float("nan")
        return [row]


def build_retriever(method: str, embedder: Embedder, cfg: Optional[HelmConfig] = None, seed: int = 0, depth: int = 100) -> Retriever:
    """``bm25`` | ``dense`` | ``shnm`` | any HELM variant name (e.g. ``helm_no_gov``) from ``helm.methods``."""
    if method == "bm25":
        return BM25Retriever()
    if method in ("dense", "flat_rag"):
        return DenseRetriever(embedder)
    if method == "shnm":
        method = "helm"
    try:
        variant_overrides(method)
    except KeyError:
        raise KeyError(f"'{method}' is not a retrieval method; use bm25, dense, shnm or a HELM variant") from None
    return SHNMRetriever(embedder, cfg, variant=method, depth=depth, seed=seed)
