"""Approximate nearest-neighbour indexes: a pure-numpy HNSW (Malkov & Yashunin), an exact flat index, and an optional
``hnswlib`` wrapper. All three share one interface and use cosine similarity on L2-normalised vectors.

The paper uses HNSW with ``M=16`` and ``ef_search=64``; ``ef_construction`` is not stated (we use 200).
Deletion is by tombstone (graph connectivity is preserved), with compaction once tombstones dominate.
"""

from __future__ import annotations

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class FlatIndex:
    """Exact cosine search. Used as ground truth in tests and as ``index_kind: flat``."""

    def __init__(self, dim: int, **_: object):
        self.dim = dim
        self._ids: List[str] = []
        self._pos: Dict[str, int] = {}
        self._vecs: List[np.ndarray] = []
        self._mat: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, ext_id: str) -> bool:
        return ext_id in self._pos

    def add(self, ext_id: str, vec: np.ndarray) -> None:
        if ext_id in self._pos:
            self.remove(ext_id)
        self._pos[ext_id] = len(self._ids)
        self._ids.append(ext_id)
        self._vecs.append(np.asarray(vec, dtype=np.float32))
        self._mat = None

    def remove(self, ext_id: str) -> None:
        i = self._pos.pop(ext_id, None)
        if i is None:
            return
        last = len(self._ids) - 1
        if i != last:
            self._ids[i] = self._ids[last]
            self._vecs[i] = self._vecs[last]
            self._pos[self._ids[i]] = i
        self._ids.pop()
        self._vecs.pop()
        self._mat = None

    def search(
        self, q: np.ndarray, k: int, ef: Optional[int] = None, allowed: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        if not self._ids:
            return []
        if self._mat is None:
            self._mat = np.stack(self._vecs)
        sims = self._mat @ np.asarray(q, dtype=np.float32)
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for i in order:
            ext = self._ids[i]
            if allowed is not None and ext not in allowed:
                continue
            out.append((ext, float(sims[i])))
            if len(out) >= k:
                break
        return out


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 0,
        **_: object,
    ):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._mL = 1.0 / math.log(max(M, 2))
        self._rng = random.Random(seed)
        self._seed = seed
        self._reset()

    def _reset(self) -> None:
        self._vecs = np.zeros((64, self.dim), dtype=np.float32)
        self._n = 0
        self._ext: List[Optional[str]] = []
        self._ext2int: Dict[str, int] = {}
        self._levels: List[int] = []
        self._links: List[Dict[int, List[int]]] = []
        self._entry: Optional[int] = None
        self._max_level = -1
        self._deleted: Set[int] = set()

    # ------------------------------------------------------------------ bookkeeping
    def __len__(self) -> int:
        return len(self._ext2int)

    def __contains__(self, ext_id: str) -> bool:
        return ext_id in self._ext2int

    def _alloc(self, ext_id: str, vec: np.ndarray) -> int:
        if self._n == len(self._vecs):
            self._vecs = np.concatenate([self._vecs, np.zeros_like(self._vecs)], axis=0)
        i = self._n
        self._vecs[i] = vec
        self._n += 1
        self._ext.append(ext_id)
        self._ext2int[ext_id] = i
        return i

    # ------------------------------------------------------------------ core graph ops
    def _search_layer(self, q: np.ndarray, eps: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        visited = set(eps)
        cand: List[Tuple[float, int]] = []  # max-heap by sim (store -sim)
        res: List[Tuple[float, int]] = []  # min-heap by sim
        for e in eps:
            s = float(self._vecs[e] @ q)
            heapq.heappush(cand, (-s, e))
            heapq.heappush(res, (s, e))
        links = self._links[level]
        while cand:
            neg_s, c = heapq.heappop(cand)
            if len(res) >= ef and -neg_s < res[0][0]:
                break
            nbrs = [n for n in links.get(c, ()) if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            sims = self._vecs[nbrs] @ q
            for n, s in zip(nbrs, sims):
                s = float(s)
                if len(res) < ef or s > res[0][0]:
                    heapq.heappush(cand, (-s, n))
                    heapq.heappush(res, (s, n))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted(res, reverse=True)

    def _greedy(self, q: np.ndarray, ep: int, level: int) -> int:
        best, best_s = ep, float(self._vecs[ep] @ q)
        improved = True
        while improved:
            improved = False
            nbrs = self._links[level].get(best, ())
            if not nbrs:
                break
            sims = self._vecs[list(nbrs)] @ q
            j = int(np.argmax(sims))
            if float(sims[j]) > best_s:
                best, best_s = nbrs[j], float(sims[j])
                improved = True
        return best

    def _shrink(self, node: int, level: int, m_max: int) -> None:
        nl = self._links[level][node]
        if len(nl) <= m_max:
            return
        sims = self._vecs[nl] @ self._vecs[node]
        keep = np.argsort(-sims)[:m_max]
        self._links[level][node] = [nl[j] for j in keep]

    # ------------------------------------------------------------------ public API
    def add(self, ext_id: str, vec: np.ndarray) -> None:
        if ext_id in self._ext2int:
            self.remove(ext_id)
        vec = np.asarray(vec, dtype=np.float32)
        i = self._alloc(ext_id, vec)
        level = int(-math.log(1.0 - self._rng.random()) * self._mL)
        self._levels.append(level)
        while len(self._links) <= level:
            self._links.append({})
        for l in range(level + 1):
            self._links[l][i] = []
        if self._entry is None:
            self._entry, self._max_level = i, level
            return
        ep = self._entry
        for l in range(self._max_level, level, -1):
            ep = self._greedy(vec, ep, l)
        eps = [ep]
        for l in range(min(level, self._max_level), -1, -1):
            cands = self._search_layer(vec, eps, self.ef_construction, l)
            m_max = self.M0 if l == 0 else self.M
            neigh = [n for _, n in cands[: self.M] if n != i]
            self._links[l][i] = list(neigh)
            for n in neigh:
                self._links[l][n].append(i)
                self._shrink(n, l, m_max)
            eps = [n for _, n in cands]
        if level > self._max_level:
            self._entry, self._max_level = i, level

    def remove(self, ext_id: str) -> None:
        i = self._ext2int.pop(ext_id, None)
        if i is None:
            return
        self._deleted.add(i)
        self._ext[i] = None
        if len(self._deleted) > 32 and len(self._deleted) > len(self._ext2int):
            self._compact()

    def _compact(self) -> None:
        live = [(self._ext[i], self._vecs[i].copy()) for i in range(self._n) if self._ext[i] is not None]
        self._rng = random.Random(self._seed)
        self._reset()
        for ext, v in live:
            self.add(ext, v)

    def search(
        self, q: np.ndarray, k: int, ef: Optional[int] = None, allowed: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        if not self._ext2int or self._entry is None:
            return []
        q = np.asarray(q, dtype=np.float32)
        ef = max(ef or self.ef_search, k)
        if allowed is not None and len(allowed) <= 2 * ef:
            return self._exact(q, k, allowed)
        n_live = len(self._ext2int)
        cur_ef = ef if allowed is None else max(ef, 4 * k)
        while True:
            ep = self._entry
            for l in range(self._max_level, 0, -1):
                ep = self._greedy(q, ep, l)
            cands = self._search_layer(q, [ep], cur_ef + len(self._deleted), 0)
            out: List[Tuple[str, float]] = []
            for s, n in cands:
                ext = self._ext[n]
                if ext is None or (allowed is not None and ext not in allowed):
                    continue
                out.append((ext, s))
                if len(out) >= k:
                    break
            if len(out) >= k or cur_ef >= n_live:
                break
            cur_ef = min(cur_ef * 2, n_live)
        if allowed is not None and len(out) < min(k, len(allowed & set(self._ext2int))):
            return self._exact(q, k, allowed)
        return out

    def _exact(self, q: np.ndarray, k: int, allowed: Optional[Set[str]]) -> List[Tuple[str, float]]:
        ids = [e for e in (allowed if allowed is not None else self._ext2int) if e in self._ext2int]
        if not ids:
            return []
        rows = [self._ext2int[e] for e in ids]
        sims = self._vecs[rows] @ q
        order = np.argsort(-sims)[:k]
        return [(ids[j], float(sims[j])) for j in order]


class HnswlibIndex:  # pragma: no cover - optional dependency, not exercised in the offline test-suite
    """Thin wrapper over ``hnswlib`` with the same interface (``index_kind: hnswlib``)."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef_search: int = 64, max_elements: int = 100_000, **_):
        try:
            import hnswlib
        except ImportError as e:
            raise ImportError("pip install hnswlib to use index_kind=hnswlib") from e
        self._idx = hnswlib.Index(space="cosine", dim=dim)
        self._idx.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M, allow_replace_deleted=True)
        self._idx.set_ef(ef_search)
        self._ext2int: Dict[str, int] = {}
        self._int2ext: Dict[int, str] = {}
        self._next = 0

    def __len__(self) -> int:
        return len(self._ext2int)

    def __contains__(self, ext_id: str) -> bool:
        return ext_id in self._ext2int

    def add(self, ext_id: str, vec: np.ndarray) -> None:
        if ext_id in self._ext2int:
            self.remove(ext_id)
        i = self._next
        self._next += 1
        self._ext2int[ext_id] = i
        self._int2ext[i] = ext_id
        self._idx.add_items(np.asarray(vec, dtype=np.float32)[None, :], [i], replace_deleted=True)

    def remove(self, ext_id: str) -> None:
        i = self._ext2int.pop(ext_id, None)
        if i is not None:
            self._int2ext.pop(i, None)
            self._idx.mark_deleted(i)

    def search(self, q, k, ef=None, allowed=None):
        if not self._ext2int:
            return []
        if ef:
            self._idx.set_ef(max(ef, k))
        kk = min(len(self._ext2int), k if allowed is None else max(k, 4 * k))
        labels, dists = self._idx.knn_query(np.asarray(q, dtype=np.float32)[None, :], k=kk)
        out = []
        for lab, d in zip(labels[0], dists[0]):
            ext = self._int2ext.get(int(lab))
            if ext is None or (allowed is not None and ext not in allowed):
                continue
            out.append((ext, 1.0 - float(d)))
            if len(out) >= k:
                break
        return out


def make_index(kind: str, dim: int, *, M: int = 16, ef_construction: int = 200, ef_search: int = 64, seed: int = 0):
    if kind == "hnsw":
        return HNSWIndex(dim, M=M, ef_construction=ef_construction, ef_search=ef_search, seed=seed)
    if kind == "flat":
        return FlatIndex(dim)
    if kind == "hnswlib":
        return HnswlibIndex(dim, M=M, ef_construction=ef_construction, ef_search=ef_search)
    raise ValueError(f"unknown index kind: {kind}")
