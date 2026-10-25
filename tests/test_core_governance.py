import unittest

import _util
from _util import ctx, mkobs, mkstore
from helm.types import ReadContext, Status


def populate(store):
    store.write(mkobs(1, "widget-7", "stock", 40))           # old SUCCESS (will be superseded)
    store.write(mkobs(5, "widget-7", "stock", 35))           # newer SUCCESS
    store.write(mkobs(6, "widget-7", "stock", 99, code=503))  # FAILED, plausible value
    store.write(mkobs(7, "gadget-3", "stock", 12))


class TestGovernance(unittest.TestCase):
    def read(self, store, q="current stock of widget-7", c=None, **kw):
        return store.read(q, c or ctx(8), **kw)

    def values(self, res):
        return [(s.item.claims[0].value if s.item.claims else None) for s in res.items]

    def test_failed_and_superseded_traces_are_excluded(self):
        st = mkstore(**{"memory.use_m2": False, "memory.use_m3": False})
        populate(st)
        res = self.read(st)
        self.assertNotIn("99", self.values(res))  # FAILED
        self.assertNotIn("40", self.values(res))  # contradicted by newer SUCCESS value 35
        self.assertIn("35", self.values(res))
        reasons = {iv["reason"] for iv in res.interventions}
        self.assertTrue({"failed", "contradicted"} <= reasons)

    def test_without_governance_failed_trace_is_returned(self):
        st = mkstore(**{"governance.enabled": False, "memory.use_m2": False, "memory.use_m3": False})
        populate(st)
        self.assertIn("99", self.values(self.read(st)))

    def test_version_mismatch_is_stale_and_fallback_flags_items(self):
        st = mkstore(**{"memory.use_m2": False, "memory.use_m3": False})
        populate(st)
        res = self.read(st, c=ctx(9, ver="v2"))
        self.assertTrue(res.fallback_used)
        self.assertTrue(all(s.readmitted and "stale" in s.reasons for s in res.items))
        self.assertLessEqual(len(res.items), st.cfg.governance.fallback_k)
        self.assertIn("INVALID", res.text)

    def test_step_window_staleness(self):
        st = mkstore(**{"governance.check_version": False, "memory.use_m2": False, "memory.use_m3": False})
        st.write(mkobs(1, "widget-7", "stock", 40))
        near = self.read(st, c=ReadContext(step=100, time=100.0, task_version="v1"))
        far = self.read(st, c=ReadContext(step=300, time=300.0, task_version="v1"))
        self.assertFalse(near.fallback_used)
        self.assertTrue(far.fallback_used)

    def test_per_signal_switches(self):
        for flag, reason in (("use_status", "failed"), ("use_freshness", "stale"), ("use_consistency", "contradicted")):
            st = mkstore(**{f"governance.{flag}": False, "memory.use_m2": False, "memory.use_m3": False})
            populate(st)
            c = ctx(9, "v2") if reason == "stale" else ctx(8)
            res = self.read(st, c=c)
            self.assertNotIn(reason, {iv["reason"] for iv in res.interventions}, flag)

    def test_score_orders_success_above_failed_even_if_failed_is_more_similar(self):
        st = mkstore(**{"governance.hard_filter": False, "memory.use_m2": False, "memory.use_m3": False})
        a = st.write(mkobs(2, "widget-7", "stock", 40))
        b = st.write(mkobs(3, "widget-7", "stock", 41, code=503))
        g = st.governor
        self.assertGreater(g.score(a, 0.5, ctx(4)), g.score(b, 0.9, ctx(4)))  # lambda_status=10 dominates

    def test_diagnostic_exemption_suspends_status_penalty(self):
        st = mkstore(**{"governance.diagnostic_exemption": True, "memory.use_m2": False, "memory.use_m3": False})
        populate(st)
        res = st.read("why did the lookup for widget-7 stock fail with an error", ctx(8))
        self.assertIn("99", self.values(res))  # failed trace readable when the subtask is diagnostic

    def test_recall_token_cap(self):
        st = mkstore(**{"memory.max_recall_tokens": 40, "memory.use_m2": False, "memory.use_m3": False})
        populate(st)
        res = self.read(st)
        self.assertLessEqual(res.tokens, 40)


if __name__ == "__main__":
    unittest.main()
