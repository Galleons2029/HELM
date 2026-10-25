"""Algorithm "Prune under memory budgets" (Appendix, governance pseudocode).

Per layer: (1) mark stale / contradicted items and surplus failed traces for removal, keeping the *minimal* failed
evidence (``keep_failed_per_key`` per claim key) needed for later verification; (2) collapse redundancy clusters (same
key, same value, same status -> keep the newest); (3) while over budget, evict the lowest-utility item.

Utility U combines recency, epistemic status, access count and whether the item still supports a live parent. With
governance off ("-Gov") the epistemic terms are dropped and Prune degrades to a recency/usage policy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from ..types import MemoryItem, ReadContext, Status


def _utility(store, it: MemoryItem, ctx: ReadContext) -> float:
    g = store.cfg.governance
    u = math.exp(-max(0, ctx.step - it.step_id) / max(g.recency_scale, 1e-6))
    u += 0.3 * min(it.hits, 5) / 5.0
    if g.enabled and g.use_status:
        u += {Status.SUCCESS: 0.5, Status.UNKNOWN: 0.0, Status.FAILED: -0.5}[it.status]
    live_parents = [p for p in it.parents if store.item(p) is not None]
    if live_parents:
        u += 0.5  # still grounds a live abstraction
    return u


def _key_of(it: MemoryItem) -> str:
    if it.claims:
        return it.claims[0].key
    return it.meta.get("key") or it.tool


def prune(store, ctx: ReadContext) -> Dict[int, int]:
    m = store.mcfg
    g = store.cfg.governance
    budgets = {1: m.budget_m1, 2: m.budget_m2, 3: m.budget_m3}
    removed: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    for layer in (1, 2, 3):
        L = store.layers[layer]
        doomed: List[str] = []
        if g.enabled and layer < 3:
            by_key_failed: Dict[str, List[MemoryItem]] = defaultdict(list)
            for it in L.items.values():
                reasons = store.governor.invalid_reasons(it, ctx)
                if "stale" in reasons or "contradicted" in reasons:
                    doomed.append(it.id)
                elif it.status == Status.FAILED:
                    by_key_failed[_key_of(it)].append(it)
            for lst in by_key_failed.values():  # keep only the newest failed traces per key
                lst.sort(key=lambda x: x.step_id, reverse=True)
                doomed.extend(x.id for x in lst[m.keep_failed_per_key :])
        # redundancy clusters
        seen: Dict[tuple, MemoryItem] = {}
        for it in sorted(L.items.values(), key=lambda x: x.step_id, reverse=True):
            if it.id in doomed or not it.claims or layer == 3:
                continue
            sig = (_key_of(it), it.claims[0].value, it.status)
            if sig in seen:
                doomed.append(it.id)
            else:
                seen[sig] = it
        for iid in set(doomed):
            L.remove(iid)
            removed[layer] += 1
        excess = len(L) - budgets[layer]
        if excess > 0:
            ranked = sorted(L.items.values(), key=lambda x: _utility(store, x, ctx))
            for it in ranked[:excess]:
                L.remove(it.id)
                removed[layer] += 1
    # drop dangling links
    for layer in (1, 2, 3):
        for it in store.layers[layer].items.values():
            it.parents = [p for p in it.parents if store.item(p) is not None]
            it.children = [c for c in it.children if store.item(c) is not None]
    for key in [k for k, i in store.m2_by_key.items() if store.layers[2].items.get(i) is None]:
        del store.m2_by_key[key]
    for t, i in list(store.m3_by_topic.items()):  # an index entry without live recalls routes nowhere
        m3 = store.layers[3].items.get(i)
        if m3 is not None and not m3.children:
            store.layers[3].remove(i)
            removed[3] += 1
            m3 = None
        if m3 is None:
            del store.m3_by_topic[t]
    store.log("PRUNE", step=ctx.step, removed=dict(removed))
    return removed
