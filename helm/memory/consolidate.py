"""Algorithm "Consolidate with conflict resolution" (Appendix, governance pseudocode).

For each claim key among the not-yet-consolidated M1 traces: rank candidates by the epistemic preference order
(SUCCESS > UNKNOWN > FAILED, newer > older, trace-supported > ungrounded), keep the winner, record the other values as
conflicts, and flag the result ambiguous when the preference gap is below ``conflict_tau``. The winner becomes (or
replaces) the M2 recall for that key; an M3 thematic index entry links all M2 recalls of one topic.

Provenance is what lets governance reach abstractions: with ``memory.provenance`` on, an M2 recall inherits the winning
trace's status, step, version and claim, so staleness/failure/contradiction are decided on the *evidence*, not on the
time it was summarised. With it off ("-Prov") the recall looks freshly written and unverified (status UNKNOWN, no claim).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..types import Claim, MemoryItem, ReadContext, SourceType, Status

_STATUS_RANK = {Status.SUCCESS: 2, Status.UNKNOWN: 1, Status.FAILED: 0}


def topic_of(key: str) -> str:
    return key.split(".", 1)[0]


def _pref(c: Claim, grounded: bool, max_step: int, epistemic: bool) -> float:
    rank = _STATUS_RANK[c.status] if epistemic else 0
    return rank + c.step_id / (max_step + 1.0) + (0.5 if grounded else 0.0)


def consolidate(store, ctx: ReadContext) -> List[MemoryItem]:
    m = store.mcfg
    g = store.cfg.governance
    if not m.use_m2:
        return []
    if store._consolidations >= m.max_consolidations_per_episode:
        store.log("CONSOLIDATE_SKIPPED", step=ctx.step, reason="max_consolidations_per_episode")
        return []
    epistemic = g.enabled and g.use_status  # the preference order is part of governance
    # 1. gather candidate claims from pending M1 traces, plus the current M2 winner per key (so a weak or failed new
    #    observation cannot silently replace verified knowledge).
    groups: Dict[str, List[Tuple[Claim, MemoryItem, bool]]] = defaultdict(list)
    consumed: List[str] = []
    for iid in store.pending:
        it = store.layers[1].items.get(iid)
        if it is None:
            continue
        consumed.append(iid)
        for c in it.claims:
            groups[c.key].append((c, it, True))
    store.pending = []
    created: List[MemoryItem] = []
    for key, cands in groups.items():
        old_id = store.m2_by_key.get(key)
        old = store.layers[2].items.get(old_id) if old_id else None
        if old is not None and old.claims:
            cands.append((old.claims[0], old, True))
        max_step = max(c.step_id for c, _, _ in cands)
        ranked = sorted(cands, key=lambda t: _pref(t[0], t[2], max_step, epistemic), reverse=True)
        win_c, win_it, _ = ranked[0]
        if old is not None and win_it is old and len(cands) == 1:
            continue
        runner = next((t for t in ranked[1:] if t[0].value != win_c.value), None)
        gap = (
            _pref(win_c, True, max_step, epistemic) - _pref(runner[0], runner[2], max_step, epistemic)
            if runner
            else float("inf")
        )
        ambiguous = gap < m.conflict_tau
        conflicts = [t[0] for t in ranked[1:] if t[0].value != win_c.value]
        created.append(_make_m2(store, key, win_c, win_it, conflicts, ambiguous, ctx))
    for iid in consumed:  # link consolidated traces to their recall
        it = store.layers[1].items.get(iid)
        if it is not None:
            for c in it.claims:
                m2 = store.m2_by_key.get(c.key)
                if m2 and m2 not in it.parents:
                    it.parents.append(m2)
    if created and m.use_m3:
        _update_themes(store, {topic_of(it.meta["key"]) for it in created}, ctx)
    store._consolidations += 1
    store.log("CONSOLIDATE", step=ctx.step, created=len(created), keys=[it.meta["key"] for it in created])
    if len(store.layers[2]) > m.budget_m2 or len(store.layers[3]) > m.budget_m3:
        store.prune(ctx)
    return created


def _make_m2(store, key, win_c: Claim, win_it: MemoryItem, conflicts, ambiguous: bool, ctx: ReadContext) -> MemoryItem:
    m = store.mcfg
    old_id = store.m2_by_key.get(key)
    if old_id:
        old = store.layers[2].remove(old_id)
        if old is not None:  # detach from M3 / M1 links
            for pid in old.parents:
                p = store.layers[3].items.get(pid)
                if p and old_id in p.children:
                    p.children.remove(old_id)
    iid = store._new_id(2)
    grounded = m.provenance
    status = win_c.status if grounded else Status.UNKNOWN
    if ambiguous and status == Status.SUCCESS:
        status = Status.UNKNOWN
    step = win_c.step_id if grounded else ctx.step
    version = win_it.task_version if grounded else ctx.task_version
    text = store.summarizer.summarize_claim(key, win_c.value, win_c.step_id, win_it.task_version, ambiguous)
    vec = store.embedder.encode([text])[0]
    claims = [Claim(key, win_c.value, step, status, iid)] if grounded else []
    item = MemoryItem(
        id=iid,
        layer=2,
        text=text,
        vec=vec,
        status=status,
        source=SourceType.SUMMARY,
        step_id=step,
        timestamp=win_it.timestamp if grounded else ctx.time,
        task_version=version,
        tool="consolidate",
        provenance=list(win_it.provenance) if grounded else [],
        children=[win_it.id] if grounded else [],
        claims=claims,
        conflicts=list(conflicts) if grounded else [],
        audit_ids=list(win_it.audit_ids),
        meta={"key": key, "ambiguous": ambiguous},
    )
    store.layers[2].add(item)
    store.m2_by_key[key] = iid
    for c in claims:
        store.registry.add(c)
    if grounded and iid not in win_it.parents:
        win_it.parents.append(iid)
    return item


def _update_themes(store, topics, ctx: ReadContext) -> None:
    for topic in topics:
        keys = sorted(k for k in store.m2_by_key if topic_of(k) == topic)
        children = [store.m2_by_key[k] for k in keys]
        attrs = [k.split(".", 1)[1] if "." in k else k for k in keys]
        text = store.summarizer.summarize_theme(topic, attrs)
        old_id = store.m3_by_topic.get(topic)
        if old_id:
            store.layers[3].remove(old_id)
        iid = store._new_id(3)
        item = MemoryItem(
            id=iid,
            layer=3,
            text=text,
            vec=store.embedder.encode([text])[0],
            status=Status.SUCCESS,
            source=SourceType.INDEX,
            step_id=ctx.step,
            timestamp=ctx.time,
            task_version=None,
            tool="theme",
            children=children,
            meta={"structural": True, "topic": topic},
        )
        store.layers[3].add(item)
        store.m3_by_topic[topic] = iid
        for cid in children:
            ch = store.layers[2].items.get(cid)
            if ch is not None:
                ch.parents = [iid]
