import math
import unittest

import _util
from _util import ctx, mkobs
from helm.config import HelmConfig
from helm.controller import (
    ControllerState, HeuristicController, LLMController, MemOp, MemoryGenerator, OpKind, RuleController,
    extract_recall_requests, format_ops, parse_ops, render_state, replace_recalls,
)
from helm.llm import MockLLM
from helm.methods import available, build_method
from helm.metrics import AuditLog, CTUMeter, EpiErrMeter, bootstrap_ci, pearson, provenance_faithfulness
from helm.sim import Episode, SimAgent, SimConfig, run_grid, summarize
from helm.types import Status


class TestControllers(unittest.TestCase):
    def test_op_format_round_trip(self):
        ops = [MemOp(OpKind.READ, "stock of widget-7"), MemOp(OpKind.WRITE), MemOp(OpKind.CONSOLIDATE)]
        back = parse_ops(format_ops(ops))
        self.assertEqual([o.kind for o in back], [o.kind for o in ops])
        self.assertEqual(back[0].query, "stock of widget-7")
        self.assertEqual(parse_ops("garbage")[0].kind, OpKind.NOOP)
        self.assertEqual(parse_ops("<op>EXPLODE</op>")[0].kind, OpKind.NOOP)

    def test_heuristic_policy(self):
        h, r = HeuristicController(), RuleController()
        noise = ControllerState(step=3, obs_text="cron ok", obs_status=Status.SUCCESS, obs_has_claims=False, budgets={1: 200, 2: 50, 3: 20})
        self.assertEqual([o.kind for o in h.decide(noise)], [OpKind.NOOP])  # skips low-information noise
        self.assertIn(OpKind.WRITE, [o.kind for o in r.decide(noise)])  # rule controller writes everything
        failed = ControllerState(step=3, obs_text="err", obs_status=Status.FAILED, budgets={1: 200})
        self.assertIn(OpKind.WRITE, [o.kind for o in h.decide(failed)])  # failed traces are kept as evidence
        ask = ControllerState(step=9, recall_request="stock of widget-7", budgets={1: 200})
        self.assertEqual(h.decide(ask)[0].kind, OpKind.READ)
        full = ControllerState(step=9, sizes={1: 195}, budgets={1: 200, 2: 50, 3: 20}, steps_since_consolidate=2)
        self.assertIn(OpKind.CONSOLIDATE, [o.kind for o in h.decide(full)])
        self.assertIn("step: 9", render_state(full))

    def test_llm_controller_parses_and_falls_back(self):
        s = ControllerState(step=2, obs_text="x=1", obs_has_claims=True, recall_request="q", budgets={1: 200})
        ops = LLMController(MockLLM(["<op>READ</op><q>custom q</q>\n<op>WRITE</op>"])).decide(s)
        self.assertEqual(ops[0].query, "custom q")
        bad = LLMController(MockLLM(["I think we should remember things"]))
        self.assertTrue(any(o.kind == OpKind.READ for o in bad.decide(s)))
        self.assertEqual(bad.n_fallbacks, 1)

    def test_recall_tags(self):
        t = "think <recall>stock of widget-7</recall> then <recall>price</recall>"
        self.assertEqual(extract_recall_requests(t), ["stock of widget-7", "price"])
        self.assertEqual(
            replace_recalls(t, ["<memory>A</memory>", "<memory>B</memory>"]), "think <memory>A</memory> then <memory>B</memory>"
        )


class TestBackendsAndGenerators(unittest.TestCase):
    def helm(self, variant="A0", client=None):
        be = build_method("helm", HelmConfig(), seed=0)
        be.generator = MemoryGenerator(variant, client=client)
        be.reset_episode()
        for i, (e, a, v, c) in enumerate([("widget-7", "stock", 40, 200), ("widget-7", "price", 5, 200), ("gadget-3", "stock", 9, 503)], 1):
            be.observe(mkobs(i, e, a, v, code=c), ctx(i))
        return be

    def test_every_registered_method_builds_and_runs(self):
        for name in available():
            be = build_method(name, HelmConfig(), seed=0)
            be.reset_episode()
            be.observe(mkobs(1, "widget-7", "stock", 40), ctx(1))
            res = be.recall("current stock of widget-7", ctx(2))
            self.assertIsNotNone(res.text, name)

    def test_a1_cited_recall_is_valid_and_a2_is_extractive(self):
        for variant in ("A1", "A2"):
            be = self.helm(variant)
            be.store.consolidate(ctx(4))
            res = be.recall("current stock of widget-7", ctx(5))
            m = provenance_faithfulness(f"<memory>{res.text}</memory>", be.store.spans)
            self.assertEqual(m["coverage"], 1.0, variant)
            self.assertEqual(m["valid"], 1.0, variant)

    def test_a1_with_llm_regenerates_on_invalid_output(self):
        good = "<memory>stock is 40 [[CITE tool=lookup log=L1 step=1 field=text start=0 end=10]].</memory>"
        be = self.helm("A1", MockLLM(["<memory>no citation here.</memory>", good]))
        be.store.consolidate(ctx(4))
        res = be.recall("current stock of widget-7", ctx(5))
        self.assertEqual(be.generator.last_info["attempts"], 2)
        self.assertIn("CITE", res.text)


class TestMetrics(unittest.TestCase):
    def test_ctu_is_sum_of_per_step_context(self):
        m = CTUMeter(system_tokens=10)
        m.step(5, 3, 2)
        m.step(5, 0, 2)
        self.assertEqual(m.total, (10 + 5 + 3 + 2) + (10 + 5 + 0 + 2))

    def test_audit_classification_uses_log_metadata_only(self):
        a = AuditLog()
        a.add(mkobs(1, "widget-7", "stock", 40))
        a.add(mkobs(5, "widget-7", "stock", 35))
        a.add(mkobs(6, "widget-7", "stock", 99, code=503))
        self.assertEqual(a.classify(["lookup:L1"], "widget-7.stock", "40", ctx(8)), ["contradicted"])
        self.assertEqual(a.classify(["lookup:L6"], "widget-7.stock", "99", ctx(8)), ["failed"])
        self.assertEqual(a.classify(["lookup:L5"], "widget-7.stock", "35", ctx(8)), [])
        self.assertIn("stale", a.classify(["lookup:L5"], "widget-7.stock", "35", ctx(8, "v2")))  # version mismatch
        self.assertEqual(a.classify(["lookup:L5"], "widget-7.stock", "35", ctx(8), truth="30"), ["stale"])  # oracle mode

    def test_epierr_rate(self):
        m = EpiErrMeter()
        m.add([])
        m.add(["failed"])
        self.assertAlmostEqual(m.rate, 50.0)
        self.assertTrue(math.isnan(EpiErrMeter().rate))

    def test_stats(self):
        self.assertEqual(bootstrap_ci([1.0] * 20), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)


class TestSimulator(unittest.TestCase):
    def test_episodes_are_deterministic_and_identical_across_methods(self):
        a, b = Episode(SimConfig(), 7), Episode(SimConfig(), 7)
        self.assertEqual([(e.kind, e.key, e.text) for e in a.events], [(e.kind, e.key, e.text) for e in b.events])
        self.assertEqual(a.lookup(("widget-1", "stock"), 50, 1).text, b.lookup(("widget-1", "stock"), 50, 1).text)
        self.assertNotEqual([e.kind for e in Episode(SimConfig(), 8).events], [e.kind for e in a.events])

    def test_world_contains_all_three_trap_types(self):
        stale = failed = contra = 0
        for s in range(40):
            ep = Episode(SimConfig(), s)
            seen = {}
            for e in ep.events:
                if e.kind == "lookup":
                    seen.setdefault(e.key, []).append(e.result)
                elif e.kind == "query" and e.key in seen:
                    obs = seen[e.key]
                    ok = [r.payload["value"] for r in obs if r.status == Status.SUCCESS]
                    truth = ep.value_at(e.key, e.step)
                    stale += bool(ok) and ok[-1] != truth
                    failed += obs[-1].status == Status.FAILED
                    contra += len(set(ok)) > 1
        self.assertGreater(stale, 0)
        self.assertGreater(failed, 0)
        self.assertGreater(contra, 0)

    def test_failed_lookup_carries_plausible_wrong_value(self):
        ep = Episode(SimConfig(p_fail=1.0, p_unknown=0.0), 3)
        key = ep.hot[0]
        r = ep.lookup(key, 60, 0)
        self.assertEqual(r.status, Status.FAILED)
        self.assertNotEqual(r.payload["value"], ep.value_at(key, 60))

    def test_governed_memory_beats_unguarded_memory_on_trap_structured_episodes(self):
        rows = run_grid(["flat_rag", "helm_no_gov", "helm"], HelmConfig(), SimConfig(), seeds=(0,), n_episodes=15, log=None)
        s = {r["method"]: r for r in summarize(rows)}
        self.assertTrue(all(r.source == "simulated" for r in rows))
        self.assertGreater(s["HELM (full)"]["success"], s["Flat RAG"]["success"] + 20)
        self.assertGreater(s["HELM -Gov"]["epierr"], 2 * s["HELM (full)"]["epierr"])  # removing governance multiplies EpiErr

    def test_trap_free_episodes_remove_the_advantage(self):
        calm = SimConfig(p_fail=0.0, p_unknown=0.0, version_bumps=0, p_silent_drift=0.0, query_trap_bias=0.0)
        rows = run_grid(["flat_rag", "helm"], HelmConfig(), calm, seeds=(0,), n_episodes=15, log=None)
        s = {r["method"]: r for r in summarize(rows)}
        self.assertLess(abs(s["HELM (full)"]["success"] - s["Flat RAG"]["success"]), 15)

    def test_agent_metrics_consistent(self):
        ep = Episode(SimConfig(), 11)
        m = SimAgent(build_method("flat_rag", HelmConfig()), HelmConfig()).run(ep)
        self.assertEqual(m.n_queries, len(ep.query_steps))
        self.assertGreater(m.ctu, 0)
        self.assertLessEqual(m.n_correct, m.n_queries)


if __name__ == "__main__":
    unittest.main()


class TestConfigs(unittest.TestCase):
    def test_default_yaml_matches_dataclass_defaults(self):
        from pathlib import Path

        loaded = HelmConfig.from_yaml(Path(_util.ROOT) / "configs" / "default.yaml")
        self.assertEqual(loaded.governance.lambda_status, 10.0)
        self.assertEqual((loaded.memory.budget_m1, loaded.memory.budget_m2, loaded.memory.budget_m3), (200, 50, 20))
        self.assertEqual((loaded.memory.k1, loaded.memory.k2, loaded.memory.k3, loaded.memory.ef_search, loaded.memory.hnsw_M), (5, 5, 3, 64, 16))
        self.assertEqual((loaded.controller.alpha, loaded.controller.beta), (1.0, 0.05))
        self.assertEqual(loaded.seeds, (0, 42, 123))

    def test_ablation_yaml_overlays_match_registry(self):
        from pathlib import Path

        from helm.methods import _HELM_VARIANTS, variant_overrides

        base = Path(_util.ROOT) / "configs"
        for name in _HELM_VARIANTS:
            via_yaml = HelmConfig.from_yaml(base / "default.yaml", base / "ablations" / f"{name}.yaml")
            via_registry = HelmConfig.from_yaml(base / "default.yaml").override(**variant_overrides(name))
            self.assertEqual(via_yaml.to_dict(), via_registry.to_dict(), name)

    def test_unknown_override_is_an_error(self):
        with self.assertRaises(AttributeError):
            HelmConfig().override(**{"governance.not_a_field": 1})
        with self.assertRaises(KeyError):
            HelmConfig.from_dict({"nope": 1})


class TestControllerData(unittest.TestCase):
    def test_sim_derived_pairs_round_trip_through_the_op_format(self):
        from helm.sim.controller_data import collect_controller_dataset

        rows = collect_controller_dataset(n_episodes=2, seed=0)
        self.assertGreater(len(rows), 50)
        kinds = set()
        for r in rows:
            self.assertEqual(r["source"], "simulated")
            ops = parse_ops(r["completion"])
            self.assertEqual(format_ops(ops), r["completion"])
            kinds |= {o.kind for o in ops}
        self.assertTrue({OpKind.READ, OpKind.WRITE, OpKind.CONSOLIDATE} <= kinds)


class TestContinualHistory(unittest.TestCase):
    def test_long_context_history_persists_when_memory_persists(self):
        be = build_method("long_context", HelmConfig())
        be.reset_episode()
        for i in range(1, 6):
            be.observe(mkobs(i, "widget-7", "stock", i), ctx(i))
        before = be.prompt_history_tokens()
        be.reset_episode(keep_memory=True)
        self.assertEqual(be.prompt_history_tokens(), before)  # items and their prompt cost stay together
        be.reset_episode()
        self.assertEqual(be.prompt_history_tokens(), 0)


class TestSimPairsMatchTrainingSchema(unittest.TestCase):
    def test_state_round_trips_and_reproduces_teacher_completion(self):
        from helm.sim.controller_data import collect_controller_dataset
        from helm.training.data_synthesis import state_from_dict

        teacher = HeuristicController()
        for r in collect_controller_dataset(n_episodes=2, seed=1):
            state = state_from_dict(r["state"])
            self.assertEqual(render_state(state), r["prompt"])
            self.assertEqual(format_ops(teacher.decide(state)), r["completion"])
