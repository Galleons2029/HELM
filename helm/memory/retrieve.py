"""Governed retrieval (Algorithm "Retrieve with epistemic governance") and ProvenanceExpansion.

Two traversal modes (``memory.traversal``):

* ``independent``     literal Alg. 1: search every layer with its own pool, keep TopK per layer by governed score S,
                      merge and sort by S.
* ``coarse_to_fine``  the text of Sec. 3.2 / Fig. 1: route through M3, rank M2 recalls (plus the recalls linked from the
                      selected M3 entries), then *ground* the top valid recalls in their supporting M1 spans via
                      ProvenanceExpansion; fall back to a direct M1 search only when no valid M2 recall exists.
                      This is the default because it is what makes the hierarchy token-efficient.

Both modes apply the hard validity filter with the soft fallback of ``Governor.rank`` and respect the per-step recall
token cap. Every item the filter removed from the similarity top-k is logged as an *intervention*.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

from ..types import MemoryItem, ReadContext, RetrievalResult, ScoredItem


def _search(store, layer: int, q: np.ndarray, k: int, extra_ids: Optional[Set[str]] = None):
    m = store.mcfg
    L = store.layers[layer]
    cands = L.search(q, max(k * m.pool_mult, k))
    if extra_ids:
        have = {it.id for it, _ in cands}
        for iid in extra_ids - have:
            it = L.items.get(iid)
            if it is not None and it.vec is not None:
                cands.append((it, float(it.vec @ q)))
    return cands


def _rank_layer(store, layer, q, k, ctx, interventions, extra_ids=None):
    cands = _search(store, layer, q, k, extra_ids)
    selected, dropped, fallback = store.governor.rank(cands, ctx, k)
    naive_top = {it.id for it, _ in sorted(cands, key=lambda t: t[1], reverse=True)[:k]}
    for d in dropped:
        if d.item.id in naive_top and d.reasons:
            interventions.append(
                {"step": ctx.step, "layer": layer, "item": d.item.id, "reason": d.reasons[0], "reasons": d.reasons}
            )
    return selected, fallback


def provenance_expansion(store, seeds: List[MemoryItem], budget: int, max_depth: int) -> List[str]:
    """BFS from the seed recalls along provenance links; returns the supporting episodic span texts.

    Pointers are resolved against the raw-log ``SpanTable`` and hash-checked, so a span that no longer matches its
    pointer is skipped rather than quoted. Stops at ``budget`` tokens or ``max_depth`` hops.
    """
    out: List[str] = []
    used = 0
    seen: Set[str] = set()
    queue = deque((s, 0) for s in seeds)
    while queue and used < budget:
        node, depth = queue.popleft()
        if node.id in seen:
            continue
        seen.add(node.id)
        if node.layer == 1:
            ptrs = node.provenance[:1]
        else:
            ptrs = node.provenance
        for p in ptrs:
            span = store.spans.resolve(p)
            if span is None:
                continue
            cost = store.tokens.count(span)
            if used + cost > budget:
                return out
            if span not in out:
                out.append(span)
                used += cost
        if depth + 1 < max_depth:
            for cid in node.children:
                ch = store.item(cid)
                if ch is not None:
                    queue.append((ch, depth + 1))
    return out


def read(store, query: str, ctx: ReadContext, instruction: str) -> RetrievalResult:
    t0 = time.perf_counter()
    m, g = store.mcfg, store.cfg.governance
    if g.diagnostic_exemption and not ctx.diagnostic:
        low = query.lower()
        if any(w in low for w in g.diagnostic_keywords):
            ctx = ReadContext(ctx.step, ctx.time, ctx.task_version, diagnostic=True)
    q = store.embedder.encode([query], instruction=instruction)[0]
    interventions: List[Dict] = []
    fallback = False
    chosen: List[ScoredItem] = []
    grounding: List[str] = []
    path: List[str] = []

    if m.traversal == "independent":
        for layer, k, enabled in ((3, m.k3, m.use_m3), (2, m.k2, m.use_m2), (1, m.k1, m.ground_to_m1)):
            if not enabled or len(store.layers[layer]) == 0:
                continue
            sel, fb = _rank_layer(store, layer, q, k, ctx, interventions)
            fallback |= fb
            if sel:
                path.append(f"M{layer}")
            chosen.extend(sel)
        chosen.sort(key=lambda s: s.score, reverse=True)
    else:
        extra: Set[str] = set()
        if m.use_m3 and len(store.layers[3]):
            sel3, _ = _rank_layer(store, 3, q, m.k3, ctx, interventions)
            if sel3:
                path.append("M3")
            for s in sel3:
                extra.update(s.item.children)
        sel2: List[ScoredItem] = []
        if m.use_m2 and len(store.layers[2]):
            sel2, fb2 = _rank_layer(store, 2, q, m.k2, ctx, interventions, extra or None)
            fallback |= fb2
            if sel2:
                path.append("M2")
        good2 = [s for s in sel2 if s.valid and not s.readmitted]
        chosen.extend(sel2)
        if not good2 and m.ground_to_m1 and len(store.layers[1]):
            sel1, fb1 = _rank_layer(store, 1, q, m.k1, ctx, interventions)  # coverage gap: direct M1 search
            fallback |= fb1
            if sel1:
                path.append("M1")
            chosen.extend(sel1)
        chosen.sort(key=lambda s: s.score, reverse=True)
        if good2 and m.ground_to_m1 and m.provenance:
            seeds = [s.item for s in good2[: m.ground_top_n]]
            used = sum(store.tokens.count(store.render(s.item, s.reasons, s.readmitted)) for s in chosen)
            grounding = provenance_expansion(store, seeds, max(0, m.max_recall_tokens - used), m.prov_depth)
            if grounding:
                path.append("M1")

    # enforce the per-step recall token cap in score order; index (M3) entries only route in coarse-to-fine mode
    lines: List[str] = []
    kept: List[ScoredItem] = []
    used = 0
    for s in chosen:
        if m.traversal != "independent" and s.item.layer == 3:
            continue
        line = store.render(s.item, s.reasons if (not s.valid or s.readmitted) else None, s.readmitted)
        cost = store.tokens.count(line)
        if used + cost > m.max_recall_tokens:
            break
        lines.append(line)
        kept.append(s)
        used += cost
        s.item.hits += 1
    for span in grounding:
        cost = store.tokens.count(span) + 2
        if used + cost > m.max_recall_tokens:
            break
        lines.append("  evidence: " + span)
        used += cost
    result = RetrievalResult(
        items=kept,
        grounding=grounding,
        text="\n".join(lines),
        tokens=used,
        interventions=interventions,
        fallback_used=fallback,
        path=">".join(path) if path else "",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )
    store.log("READ", step=ctx.step, tokens=used, path=result.path, interventions=len(interventions), fallback=fallback)
    return result
