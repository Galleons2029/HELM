import unittest

import _util
from _util import ctx, mkobs, mkstore
from helm.memory.retrieve import provenance_expansion
from helm.types import Status


class TestWriteConsolidatePrune(unittest.TestCase):
    def test_write_limit_per_step(self):
        st = mkstore()
        self.assertIsNotNone(st.write(mkobs(1, "a-1", "stock", 1)))
        self.assertIsNone(st.write(mkobs(1, "b-2", "stock", 2)))  # second write in the same step is refused
        self.assertEqual(len(st.layers[1]), 1)

    def test_consolidate_keeps_newest_verified_and_records_conflicts(self):
        st = mkstore()
        st.write(mkobs(1, "widget-7", "stock", 40))
        st.write(mkobs(5, "widget-7", "stock", 35))
        st.write(mkobs(6, "widget-7", "stock", 99, code=503))  # newest but FAILED
        made = st.consolidate(ctx(7))
        self.assertEqual(len(made), 1)
        m2 = made[0]
        self.assertEqual(m2.claims[0].value, "35")
        self.assertEqual(m2.status, Status.SUCCESS)
        self.assertEqual({c.value for c in m2.conflicts}, {"40", "99"})
        self.assertEqual(m2.step_id, 5)  # inherits the *evidence* step through provenance
        self.assertEqual(len(st.layers[3]), 1)  # one thematic index for the entity

    def test_ambiguous_when_preference_gap_is_small(self):
        st = mkstore()
        st.write(mkobs(10, "widget-7", "stock", 40))
        st.write(mkobs(11, "widget-7", "stock", 41))  # two SUCCESS values one step apart
        m2 = st.consolidate(ctx(12))[0]
        self.assertTrue(m2.meta["ambiguous"])
        self.assertEqual(m2.status, Status.UNKNOWN)

    def test_naive_consolidation_without_governance_trusts_newest_even_if_failed(self):
        st = mkstore(**{"governance.enabled": False})
        st.write(mkobs(5, "widget-7", "stock", 35))
        st.write(mkobs(6, "widget-7", "stock", 99, code=503))
        self.assertEqual(st.consolidate(ctx(7))[0].claims[0].value, "99")

    def test_no_provenance_makes_summaries_look_fresh_and_unverified(self):
        st = mkstore(**{"memory.provenance": False})
        st.write(mkobs(1, "widget-7", "stock", 40))
        m2 = st.consolidate(ctx(50))[0]
        self.assertEqual(m2.step_id, 50)  # looks freshly written ...
        self.assertEqual(m2.status, Status.UNKNOWN)  # ... and cannot be verified
        self.assertEqual(m2.claims, [])
        self.assertEqual(m2.provenance, [])

    def test_consolidation_op_budget(self):
        st = mkstore(**{"memory.max_consolidations_per_episode": 2})
        for i in range(1, 5):
            st.write(mkobs(i, f"e-{i}", "stock", i))
            st.consolidate(ctx(i))
        self.assertEqual(st._consolidations, 2)

    def test_prune_enforces_budget_and_keeps_minimal_failed_evidence(self):
        st = mkstore(**{"memory.budget_m1": 6})
        for i in range(1, 15):
            st.write(mkobs(i, f"e-{i % 5}", "stock", i, code=503 if i % 3 == 0 else 200))
        self.assertLessEqual(len(st.layers[1]), 6)
        st2 = mkstore()
        for i in range(1, 8):
            st2.write(mkobs(i, "widget-7", "stock", 90 + i, code=503))
        st2.prune(ctx(9))
        failed = [it for it in st2.layers[1].items.values() if it.status == Status.FAILED]
        self.assertEqual(len(failed), st2.mcfg.keep_failed_per_key)  # never deleted outright, minimal retained

    def test_prune_removes_stale_version_and_orphan_index_entries(self):
        st = mkstore()
        st.write(mkobs(1, "widget-7", "stock", 40))
        st.consolidate(ctx(2))
        st.prune(ctx(3, "v2"))
        self.assertEqual(st.sizes(), {1: 0, 2: 0, 3: 0})

    def test_provenance_expansion_respects_budget_and_hash(self):
        st = mkstore()
        for i in range(1, 4):
            st.write(mkobs(i, f"e-{i}", "stock", i))
        made = st.consolidate(ctx(5))
        spans = provenance_expansion(st, made, budget=10_000, max_depth=2)
        self.assertEqual(len(spans), 3)
        tiny = provenance_expansion(st, made, budget=5, max_depth=2)
        self.assertLess(len(tiny), 3)
        # tamper with a raw log -> its pointer no longer resolves, so it is skipped rather than quoted
        key = next(iter(st.spans._fields))
        st.spans._fields[key] = "tampered"
        self.assertEqual(len(provenance_expansion(st, made, budget=10_000, max_depth=2)), 2)


class TestRetrievalModes(unittest.TestCase):
    def build(self, **ov):
        st = mkstore(**ov)
        st.write(mkobs(1, "widget-7", "stock", 40))
        st.write(mkobs(2, "widget-7", "price", 5))
        st.write(mkobs(3, "gadget-3", "stock", 12))
        st.consolidate(ctx(4))
        return st

    def test_coarse_to_fine_routes_through_layers_and_grounds_in_m1(self):
        res = self.build().read("current stock of widget-7", ctx(5))
        self.assertEqual(res.path, "M3>M2>M1")
        self.assertTrue(res.grounding)
        self.assertEqual(res.items[0].item.layer, 2)

    def test_independent_mode_returns_all_layers(self):
        res = self.build(**{"memory.traversal": "independent"}).read("current stock of widget-7", ctx(5))
        self.assertEqual({s.item.layer for s in res.items}, {1, 2, 3})

    def test_layer_ablations(self):
        self.assertNotIn("M1", self.build(**{"memory.ground_to_m1": False}).read("stock widget-7", ctx(5)).path)
        res = self.build(**{"memory.use_m3": False}).read("stock widget-7", ctx(5))
        self.assertNotIn("M3", res.path)

    def test_hnsw_and_flat_index_agree_on_top_result(self):
        a = self.build().read("price of widget-7", ctx(5)).items[0].item.text
        b = self.build(**{"memory.index_kind": "flat"}).read("price of widget-7", ctx(5)).items[0].item.text
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
