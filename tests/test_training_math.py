"""Pure-math / no-model unit tests for helm.training (stdlib unittest only).

Data synthesis and rewards need no torch; the loss/pooling/schedule tests need torch tensors (no model) and skip cleanly
if torch is missing.
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone

from helm.controller.base import OpKind, format_ops, parse_ops
from helm.controller.heuristic import HeuristicController
from helm.controller.llm import SYSTEM_PROMPT
from helm.llm.client import MockLLM
from helm.memory.claims import extract_claims
from helm.memory.tokens import DEFAULT_COUNTER
from helm.training import SOURCES, thirds, write_metrics
from helm.training import data_synthesis as ds
from helm.training import rewards as rw
from helm.types import Status

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOL_OUT = re.compile(r"tool_output: (\{.*\})")


def payload(text: str) -> dict:
    return json.loads(_TOOL_OUT.search(text).group(1))


def ts(p: dict) -> datetime:
    return datetime.strptime(p["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class TestDataSynthesis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = ds.SynthConfig(n=120, seed=0, long_frac=0.0)
        cls.recs = ds.generate_offline_dataset(cls.cfg)

    def test_triplet_hierarchy_and_lengths(self):
        st = ds.dataset_stats(self.recs)
        m = st["mean_tokens"]
        self.assertTrue(m["m3"] < m["m2"] < m["m1"], m)
        self.assertTrue(6 <= m["m3"] <= 11, m)  # thematic index keys are about 8 tokens
        for r in self.recs:
            self.assertTrue(all(r["triplet"][k] for k in ("m1", "m2", "m3")))

    def test_positive_comes_from_every_layer(self):
        self.assertEqual({r["layer"] for r in self.recs}, {1, 2, 3})
        for r in self.recs:
            self.assertEqual(r["positive"], r["triplet"][f"m{r['layer']}"])
            self.assertEqual(r["instruction"], ds.INSTRUCTIONS[r["layer"]])

    def test_three_counterfactuals_per_query_one_per_kind(self):
        for r in self.recs:
            self.assertEqual(len(r["negatives"]), 3)
            self.assertEqual({n["violation"] for n in r["negatives"]}, set(ds.VIOLATIONS))
            for n in r["negatives"]:
                self.assertNotEqual(n["text"], r["positive"])

    def test_counterfactuals_violate_validity_and_keep_topic(self):
        checked = 0
        for r in self.recs:
            if r["layer"] != 1:
                continue
            pos = payload(r["positive"])
            self.assertEqual(Status.from_code(pos["code"]), Status.SUCCESS)
            key_pos = extract_claims(r["positive"])[0]
            for n in r["negatives"]:
                neg = payload(n["text"])
                self.assertEqual((neg["entity"], neg["attr"]), (pos["entity"], pos["attr"]))  # same topic/lexical content
                if n["violation"] == "failed":
                    self.assertEqual(Status.from_code(neg["code"]), Status.FAILED)
                    self.assertIsNone(neg["value"])
                    self.assertIn("error", neg)
                elif n["violation"] == "stale":
                    self.assertGreaterEqual((ts(pos) - ts(neg)).days, 7)  # outside the 7-day window
                    self.assertNotEqual(neg["version"], pos["version"])
                else:
                    key_neg = extract_claims(n["text"])[0]
                    self.assertEqual(key_neg[0], key_pos[0])  # same extracted key, disagreeing value
                    self.assertNotEqual(key_neg[1], key_pos[1])
                    self.assertEqual(Status.from_code(neg["code"]), Status.SUCCESS)
                    self.assertLess(ts(neg), ts(pos))
                    self.assertLess((ts(pos) - ts(neg)).days, 7)
            checked += 1
        self.assertGreater(checked, 20)

    def test_summary_layer_counterfactuals_expose_the_violation(self):
        for r in self.recs:
            if r["layer"] != 2:
                continue
            self.assertIn("status SUCCESS", r["positive"])
            for n in r["negatives"]:
                if n["violation"] == "failed":
                    self.assertIn("status FAILED", n["text"])
                    self.assertNotIn("status SUCCESS", n["text"])

    def test_m3_positive_negatives_are_rendered_traces(self):
        for r in self.recs:
            if r["layer"] == 3:
                self.assertTrue(all(n["layer"] in (1, 2) for n in r["negatives"]))

    def test_long_horizon_expansion(self):
        recs = ds.generate_offline_dataset(ds.SynthConfig(n=24, seed=3, long_frac=1.0))
        lengths = []
        for r in recs:
            self.assertEqual(r["kind"], "long")
            ent = r["meta"]["entity"]
            texts = [r["positive"]] + [n["text"] for n in r["negatives"]]
            for t in texts:
                self.assertTrue(512 <= DEFAULT_COUNTER.count(t) <= 8192, DEFAULT_COUNTER.count(t))
                self.assertEqual(t.count(f'entity="{ent}"'), 1)  # the needle appears exactly once
            # negatives differ from the positive only at the needle block: identical distractors everywhere else
            blocks = [t.split("\n\n") for t in texts]
            idx = r["meta"]["needle_index"]
            for b in blocks[1:]:
                self.assertEqual(len(b), len(blocks[0]))
                self.assertEqual(b[:idx], blocks[0][:idx])
                self.assertEqual(b[idx + 1 :], blocks[0][idx + 1 :])
                self.assertNotEqual(b[idx], blocks[0][idx])
            lengths.append(r["meta"]["context_tokens"])
        self.assertGreater(max(lengths) - min(lengths), 1000)  # lengths vary across the configured range

    def test_deterministic_and_seed_dependent(self):
        a = ds.generate_offline_dataset(ds.SynthConfig(n=10, seed=5))
        b = ds.generate_offline_dataset(ds.SynthConfig(n=10, seed=5))
        c = ds.generate_offline_dataset(ds.SynthConfig(n=10, seed=6))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x", "t.jsonl")
            self.assertEqual(ds.write_jsonl(path, self.recs[:10]), 10)
            self.assertEqual(ds.read_jsonl(path), self.recs[:10])

    def test_distractors_do_not_share_the_needle_key(self):
        import random

        rng = random.Random(0)
        for _ in range(200):
            spec = rng.choice(ds.BUILTIN_SEEDS)
            f = ds.draw_facts(spec, rng)
            ep = ds.distractor_episode(ds.BUILTIN_SEEDS, rng, f.key, 5, ds.NOW)
            self.assertNotIn(f'entity="{f.entity}"', ep)

    # ---- teacher-LLM path (MockLLM only)
    GOOD = json.dumps(
        {
            "tool": "git.log", "entity": "repo_x", "attr": "head", "value": "a1b2c3", "user": "What is HEAD of repo_x?",
            "m1": '[step <<step>>] tool_call: git.log(repo="repo_x")\n[step <<step>>] tool_output: '
                  '{"head": "<<value>>", "code": <<code>>, "ts": "<<ts>>", "version": "<<version>>"}',
            "m2": "To read HEAD of repo_x call git.log; head = <<value>> at step <<step>> (<<version>>).",
            "m3": "git repository head lookup",
        }
    )

    def test_teacher_generator_with_mock_llm(self):
        client = MockLLM(["sorry, no json here", "```json\n" + self.GOOD + "\n```"])
        got = ds.synthesize_triplet_with_teacher(client, "some seed", max_retries=2)
        self.assertIsNotNone(got)
        self.assertEqual(len(client.calls), 2)  # first reply unusable, second accepted
        self.assertIn("some seed", client.calls[0][-1]["content"])
        facts, tpl = got
        self.assertIn("a1b2c3", tpl.m1(facts))
        self.assertNotIn("<<", tpl.m1(facts) + tpl.m2(facts))

        recs, failed = ds.generate_teacher_dataset(MockLLM([self.GOOD]), ["s1", "s2", "s3"], ds.SynthConfig(seed=1, long_frac=0.0))
        self.assertEqual((len(recs), failed), (3, 0))
        for r in recs:
            self.assertEqual({n["violation"] for n in r["negatives"]}, set(ds.VIOLATIONS))
            self.assertEqual(r["meta"]["generator"], "teacher_llm")
            failed_neg = next(n for n in r["negatives"] if n["violation"] == "failed")
            if failed_neg["layer"] == 1:
                self.assertIn("null", failed_neg["text"])
                self.assertRegex(failed_neg["text"], r'"code": [45]\d\d')
            self.assertNotIn("<<", r["positive"])

    def test_teacher_generator_counts_failures_and_validates_placeholders(self):
        recs, failed = ds.generate_teacher_dataset(MockLLM(["garbage"]), ["a", "b"], ds.SynthConfig(seed=1))
        self.assertEqual((recs, failed), ([], 2))
        bad = json.loads(self.GOOD)
        bad["m1"] = "no placeholders at all"
        self.assertIsNone(ds.parse_teacher_reply(json.dumps(bad)))
        self.assertIsNotNone(ds.parse_teacher_reply(self.GOOD))

    def test_teacher_dataset_long_horizon(self):
        recs, _ = ds.generate_teacher_dataset(MockLLM([self.GOOD]), ["s1", "s2"], ds.SynthConfig(seed=1, long_frac=1.0, min_ctx_tokens=512, max_ctx_tokens=2000))
        for r in recs:
            self.assertTrue(512 <= DEFAULT_COUNTER.count(r["positive"]) <= 2000)

    def test_fineweb_loader_is_lazy_and_filters(self):
        self.assertTrue(ds.looks_procedural("First, install it. Then run the command. Next, check step 3 and finally commit."))
        self.assertFalse(ds.looks_procedural("The capital of France is Paris and it is large."))
        try:
            import datasets  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                ds.load_fineweb_edu_seeds(1)

    def test_cli_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.jsonl")
            r = subprocess.run([sys.executable, "scripts/synthesize_data.py", "--smoke", "--out", out], cwd=ROOT,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, "stats.json")) as f:
                stats = json.load(f)
            self.assertEqual(stats["source"], "smoke")
            self.assertEqual(len(ds.read_jsonl(out)), 64)


class TestControllerDataset(unittest.TestCase):
    def test_completion_is_heuristic_decision(self):
        recs = ds.make_synthetic_controller_dataset(150, seed=1)
        h = HeuristicController()
        kinds = set()
        for r in recs:
            s = ds.state_from_dict(r["state"])
            self.assertEqual(r["completion"], format_ops(h.decide(s)))
            self.assertEqual(r["prompt"], SYSTEM_PROMPT + "\n\n" + ds.render_state(s) + "\n")  # == HFController prompt
            self.assertEqual([o.kind for o in parse_ops(r["completion"])], [o.kind for o in h.decide(s)])
            kinds.update(o.kind for o in parse_ops(r["completion"]))
        self.assertGreaterEqual(kinds, {OpKind.READ, OpKind.WRITE, OpKind.CONSOLIDATE, OpKind.PRUNE, OpKind.NOOP})

    def test_wrap_controller_prompt_matches_hfcontroller_framing(self):
        s = ds.random_controller_state(__import__("random").Random(0))
        bare = ds.render_state(s)
        self.assertEqual(ds.wrap_controller_prompt(bare), ds.controller_prompt(s))
        self.assertEqual(ds.wrap_controller_prompt(ds.controller_prompt(s)), ds.controller_prompt(s))  # idempotent

    def test_state_roundtrip_json_safe_and_deterministic(self):
        a = ds.make_synthetic_controller_dataset(20, seed=2)
        self.assertEqual(a, ds.make_synthetic_controller_dataset(20, seed=2))
        for r in a:
            r2 = json.loads(json.dumps(r))
            self.assertEqual(ds.state_from_dict(r2["state"]), ds.state_from_dict(r["state"]))


class TestRewards(unittest.TestCase):
    def test_hybrid_defaults_are_paper_values(self):
        self.assertAlmostEqual(rw.hybrid_reward(0.5, 1.0), 1.0 * 0.5 + 0.05 * 1.0)
        self.assertAlmostEqual(rw.hybrid_reward(1.0, -1.0, alpha=2.0, beta=0.1), 2.0 - 0.1)

    def test_memory_reward_bounded_and_monotone(self):
        best = rw.memory_reward(0, 100, 0, 0, 4)
        self.assertEqual(best, 1.0)
        worst = rw.memory_reward(10_000, 100, 10, 10, 4)
        self.assertEqual(worst, -1.0)
        self.assertGreater(rw.memory_reward(10, 100, 0, 0, 4), rw.memory_reward(50, 100, 0, 0, 4))
        self.assertGreater(rw.memory_reward(10, 100, 0, 0, 4), rw.memory_reward(10, 100, 1, 0, 4))
        self.assertGreater(rw.memory_reward(10, 100, 0, 0, 4), rw.memory_reward(10, 100, 0, 1, 4))
        for args in [(0, 100, 0, 0, 0), (50, 100, 3, 3, 3), (1e9, 1, 99, 99, 1)]:
            self.assertTrue(-1.0 <= rw.memory_reward(*args) <= 1.0)
        with self.assertRaises(ValueError):
            rw.memory_reward(1, 0, 0, 0, 1)

    def test_toy_task_reward(self):
        recs = ds.make_synthetic_controller_dataset(80, seed=4)
        n_read = 0
        for r in recs:
            s = ds.state_from_dict(r["state"])
            self.assertEqual(rw.toy_task_reward(s, r["completion"]), 1.0)  # the teacher's own decision
            self.assertEqual(rw.toy_task_reward(s, "no ops here"), 0.0)
            if s.recall_request:
                n_read += 1
                self.assertLess(rw.toy_task_reward(s, "<op>NOOP</op>"), 1.0)
                wrong = r["completion"].replace(s.recall_request, "something else")
                self.assertAlmostEqual(rw.toy_task_reward(s, wrong), 0.5)  # READ present but wrong query
        self.assertGreater(n_read, 10)

    def test_toy_memory_stats_flags_redundant_writes_and_risky_reads(self):
        recs = ds.make_synthetic_controller_dataset(40, seed=5, recall_prob=1.0)
        s = ds.state_from_dict(recs[0]["state"])
        text = f"<op>READ</op><q>{s.recall_request}</q>\n<op>WRITE</op>\n<op>WRITE</op>\n<op>READ</op><q>unrelated</q>"
        st = rw.toy_memory_stats(s, text)
        self.assertEqual(st, {"n_ops": 4, "n_redundant_writes": 1, "n_risky_reads": 1})
        clean = rw.toy_hybrid_reward(s, f"<op>READ</op><q>{s.recall_request}</q>", 10)
        dirty = rw.toy_hybrid_reward(s, text, 10)
        self.assertGreater(clean["r_mem"], dirty["r_mem"])

    def test_make_toy_reward_fn_uses_record_state(self):
        rec = ds.make_synthetic_controller_dataset(1, seed=7)[0]
        out = rw.make_toy_reward_fn()(rec, rec["completion"], 12)
        self.assertEqual(out["r_task"], 1.0)
        self.assertAlmostEqual(out["reward"], out["r_task"] + 0.05 * out["r_mem"])


class TestMetricsHelpers(unittest.TestCase):
    def test_write_metrics_requires_valid_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_metrics(os.path.join(d, "m.json"), "smoke", a=1)
            with open(p) as f:
                self.assertEqual(json.load(f), {"source": "smoke", "a": 1})
            with self.assertRaises(ValueError):
                write_metrics(os.path.join(d, "m2.json"), "paper")
        self.assertEqual(set(SOURCES), {"smoke", "measured"})

    def test_thirds(self):
        self.assertEqual(thirds([1, 1, 1, 2, 2, 2, 3, 3, 3]), (1, 3))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestEmbeddingMath(unittest.TestCase):
    def setUp(self):
        from helm.training import embedding_ft as ef

        self.ef = ef
        torch.manual_seed(0)

    def test_weighted_pooling_matches_manual(self):
        h = torch.randn(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        got = self.ef.pool_hidden(h, mask, "weighted")
        w0 = torch.tensor([1.0, 2.0, 3.0])
        want0 = (h[0, :3] * w0[:, None]).sum(0) / w0.sum()
        w1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        want1 = (h[1] * w1[:, None]).sum(0) / w1.sum()
        torch.testing.assert_close(got, torch.stack([want0, want1]))
        torch.testing.assert_close(self.ef.pool_hidden(h, mask, "mean")[0], h[0, :3].mean(0))

    def test_pooling_ignores_padding_content_and_side(self):
        h = torch.randn(1, 3, 5)
        right = torch.cat([h, torch.randn(1, 2, 5) * 100], 1)
        left = torch.cat([torch.randn(1, 2, 5) * 100, h], 1)
        for mode in ("weighted", "mean"):
            a = self.ef.pool_hidden(right, torch.tensor([[1, 1, 1, 0, 0]]), mode)
            b = self.ef.pool_hidden(left, torch.tensor([[0, 0, 1, 1, 1]]), mode)
            torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            self.ef.pool_hidden(h, torch.ones(1, 3), "max")

    def _manual_infonce(self, q, pos, negs, tau, groups=None, cross=True):
        q, pos, negs = (torch.nn.functional.normalize(x, dim=-1) for x in (q, pos, negs))
        B = q.size(0)
        losses = []
        for i in range(B):
            sims = [float(q[i] @ pos[i]) / tau]
            logits = [sims[0]]
            for j in range(B):
                if j != i and not (groups and groups[i] == groups[j]):
                    logits.append(float(q[i] @ pos[j]) / tau)
            for j in range(B):
                if cross or j == i:
                    for k in range(negs.size(1)):
                        logits.append(float(q[i] @ negs[j, k]) / tau)
            losses.append(-(logits[0] - math.log(sum(math.exp(x) for x in logits))))
        return sum(losses) / B

    def test_info_nce_matches_the_papers_formula(self):
        q, pos, negs = torch.randn(4, 6), torch.randn(4, 6), torch.randn(4, 3, 6)
        for tau in (0.02, 0.5):
            got, _ = self.ef.info_nce_loss(q, pos, negs, tau)
            self.assertAlmostEqual(float(got), self._manual_infonce(q, pos, negs, tau), places=4)
        got, _ = self.ef.info_nce_loss(q, pos, negs, 0.3, cross_query_negatives=False)
        self.assertAlmostEqual(float(got), self._manual_infonce(q, pos, negs, 0.3, cross=False), places=4)

    def test_info_nce_masks_same_seed_positives(self):
        q, pos, negs = torch.randn(3, 6), torch.randn(3, 6), torch.randn(3, 2, 6)
        groups = ["a", "a", "b"]
        got, _ = self.ef.info_nce_loss(q, pos, negs, 0.2, groups)
        self.assertAlmostEqual(float(got), self._manual_infonce(q, pos, negs, 0.2, groups), places=4)
        plain, _ = self.ef.info_nce_loss(q, pos, negs, 0.2)
        self.assertNotAlmostEqual(float(got), float(plain), places=3)

    def test_info_nce_without_counterfactuals_and_gradients(self):
        q = torch.randn(3, 5, requires_grad=True)
        pos = torch.randn(3, 5, requires_grad=True)
        negs = torch.randn(3, 2, 5, requires_grad=True)
        loss, stats = self.ef.info_nce_loss(q, pos, negs, 0.05)
        loss.backward()
        self.assertTrue(all(t.grad is not None and t.grad.abs().sum() > 0 for t in (q, pos, negs)))
        plain, _ = self.ef.info_nce_loss(q.detach(), pos.detach(), None, 0.05)
        self.assertTrue(torch.isfinite(plain))
        self.assertIn("acc", stats)
        # a perfectly aligned positive with orthogonal negatives -> near-zero loss
        eye = torch.eye(4)
        near0, st = self.ef.info_nce_loss(eye, eye, None, 0.02)
        self.assertLess(float(near0), 1e-6)
        self.assertEqual(st["acc"], 1.0)

    def test_counterfactual_negatives_raise_the_loss(self):
        q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        pos = q.clone()
        hard = q[:, None, :].expand(-1, 2, -1) * 0.99 + 0.01 * torch.randn(4, 2, 8)  # nearly identical to the positive
        with_hard, _ = self.ef.info_nce_loss(q, pos, hard, 0.05)
        without, _ = self.ef.info_nce_loss(q, pos, None, 0.05)
        self.assertGreater(float(with_hard), float(without))

    def test_cosine_with_warmup(self):
        f = self.ef.cosine_with_warmup
        self.assertAlmostEqual(f(0, 4, 20), 0.25)
        self.assertAlmostEqual(f(3, 4, 20), 1.0)
        self.assertAlmostEqual(f(4, 4, 20), 1.0)
        self.assertAlmostEqual(f(12, 4, 20), 0.5)
        self.assertAlmostEqual(f(20, 4, 20), 0.0, places=6)
        self.assertAlmostEqual(f(100, 4, 20, min_ratio=0.1), 0.1, places=6)
        self.assertAlmostEqual(f(0, 0, 10), 1.0)

    def test_bidirectional_additive_mask(self):
        m = self.ef.bidirectional_additive_mask(torch.tensor([[1, 1, 0]]), torch.float32)
        self.assertEqual(tuple(m.shape), (1, 1, 3, 3))
        self.assertTrue(torch.all(m[0, 0, :, :2] == 0))  # no causal structure among real tokens
        self.assertTrue(torch.all(m[0, 0, :, 2] == torch.finfo(torch.float32).min))

    def test_defaults_match_paper_hyperparameters(self):
        cfg = self.ef.EmbeddingFTConfig()
        self.assertEqual((cfg.tau, cfg.lr, cfg.warmup_steps, cfg.batch_size, cfg.epochs), (0.02, 1e-5, 500, 256, 3))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestOPDMath(unittest.TestCase):
    def setUp(self):
        from helm.training import opd

        self.opd = opd
        torch.manual_seed(0)

    def test_advantage_is_negative_reverse_kl_and_detached(self):
        s = torch.tensor([-1.0, -2.0], requires_grad=True)
        t = torch.tensor([-1.5, -0.5], requires_grad=True)
        a = self.opd.reverse_kl_advantages(s, t)
        torch.testing.assert_close(a, torch.tensor([-0.5, 1.5]))  # -(log pi - log piT)
        self.assertFalse(a.requires_grad)

    def test_mean_reverse_kl_respects_mask(self):
        s = torch.tensor([[-1.0, -2.0, 0.0]])
        t = torch.tensor([[-1.5, -0.5, 0.0]])
        m = torch.tensor([[1, 1, 0]])
        self.assertAlmostEqual(float(self.opd.mean_reverse_kl(s, t, m)), (0.5 - 1.5) / 2)

    def test_loss_value_and_gradient_at_ratio_one(self):
        lp = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
        adv = torch.tensor([[0.5, -1.0, 2.0]])
        mask = torch.tensor([[1, 1, 0]])
        loss = self.opd.opd_loss(lp, lp.detach(), adv, mask)
        self.assertAlmostEqual(float(loss.detach()), -(0.5 - 1.0) / 2)
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[-0.25, 0.5, 0.0]]))  # -A/n on valid tokens, 0 on padding
        lp.grad = None
        self.opd.opd_loss(lp, lp.detach(), adv, mask, normalizer=4).backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[-0.125, 0.25, 0.0]]))

    def test_importance_ratio_enters_the_gradient(self):
        old = torch.tensor([[-1.0]])
        new = (old + 0.5).requires_grad_()
        self.opd.opd_loss(new, old, torch.tensor([[2.0]]), torch.ones(1, 1)).backward()
        self.assertAlmostEqual(float(new.grad), -2.0 * math.exp(0.5), places=5)

    def test_expected_gradient_equals_reverse_kl_gradient(self):
        # exact check on a categorical: E_{a~p}[ -A(a) grad log p(a) ] == grad KL(p || q), A = -(log p - log q)
        theta = torch.randn(7, requires_grad=True)
        q_logits = torch.randn(7)
        logp = torch.log_softmax(theta, -1)
        logq = torch.log_softmax(q_logits, -1)
        kl = (logp.exp() * (logp - logq)).sum()
        (g_exact,) = torch.autograd.grad(kl, theta, retain_graph=True)
        p = logp.detach().exp()
        total = 0.0
        for a in range(7):
            lp_a = logp[a].view(1, 1)
            adv = self.opd.reverse_kl_advantages(lp_a, logq[a].view(1, 1))
            total = total + p[a] * self.opd.opd_loss(lp_a, lp_a.detach(), adv, torch.ones(1, 1))
        (g_opd,) = torch.autograd.grad(total, theta)
        torch.testing.assert_close(g_opd, g_exact, atol=1e-6, rtol=1e-5)

    def test_descending_the_opd_loss_reduces_exact_kl(self):
        theta = torch.randn(5, requires_grad=True)
        q = torch.log_softmax(torch.randn(5), -1)
        opt = torch.optim.SGD([theta], lr=0.5)
        kls = []
        for _ in range(40):
            logp = torch.log_softmax(theta, -1)
            kls.append(float((logp.exp() * (logp - q)).sum().detach()))
            idx = torch.multinomial(logp.exp().detach(), 64, replacement=True)
            lp = logp[idx].view(1, -1)
            adv = self.opd.reverse_kl_advantages(lp, q[idx].view(1, -1))
            loss = self.opd.opd_loss(lp, lp.detach(), adv, torch.ones_like(lp))
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(kls[-1], 0.2 * kls[0])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestGSPOMath(unittest.TestCase):
    def setUp(self):
        from helm.training import gspo

        self.g = gspo
        torch.manual_seed(0)

    def test_group_advantages(self):
        r = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
        a = self.g.group_advantages(r)
        std = math.sqrt(1.25)  # population std of 1..4
        torch.testing.assert_close(a[0], (r[0] - 2.5) / (std + 1e-6))
        self.assertAlmostEqual(float(a[0].mean()), 0.0, places=6)
        torch.testing.assert_close(a[1], torch.zeros(4))  # no signal in a constant group

    def test_sequence_ratio_is_length_normalised(self):
        delta = 0.1
        old = torch.tensor([[-1.0, -1.0, -1.0, 0.0, 0.0], [-2.0, -2.0, -2.0, -2.0, -2.0]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        s = self.g.sequence_ratio(old + delta * mask, old, mask)
        torch.testing.assert_close(s, torch.full((2,), math.exp(delta)))  # same per-token shift -> same ratio

    def test_gradient_at_ratio_one(self):
        lp = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -1.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        adv = torch.tensor([1.5, -0.5])
        loss = self.g.gspo_loss(lp, lp.detach(), adv, mask, 0.2, 0.3)
        self.assertAlmostEqual(float(loss.detach()), -(1.5 - 0.5) / 2)
        loss.backward()
        want = torch.tensor([[-1.5 / 3 / 2] * 3, [0.5 / 2 / 2, 0.5 / 2 / 2, 0.0]])  # -A_i / (|y_i| * B)
        torch.testing.assert_close(lp.grad, want)

    def _grad_for(self, s_log: float, adv: float, lo=0.2, hi=0.3):
        old = torch.zeros(1, 2)
        new = (old + s_log).requires_grad_()
        loss = self.g.gspo_loss(new, old, torch.tensor([adv]), torch.ones(1, 2), lo, hi)
        loss.backward()
        return float(loss.detach()), float(new.grad.abs().sum())

    def test_clipping_cases(self):
        up, down = math.log(1.5), math.log(0.5)  # ratios 1.5 (> 1.3) and 0.5 (< 0.8)
        loss, g = self._grad_for(up, +1.0)  # A>0, s too high -> clipped at 1+eps_high, no gradient
        self.assertAlmostEqual(loss, -1.3, places=5)
        self.assertEqual(g, 0.0)
        loss, g = self._grad_for(up, -1.0)  # A<0, s too high -> unclipped (min picks s*A), gradient flows
        self.assertAlmostEqual(loss, 1.5, places=5)
        self.assertGreater(g, 0.0)
        loss, g = self._grad_for(down, +1.0)  # A>0, s too low -> unclipped
        self.assertAlmostEqual(loss, -0.5, places=5)
        self.assertGreater(g, 0.0)
        loss, g = self._grad_for(down, -1.0)  # A<0, s too low -> clipped at 1-eps_low, no gradient
        self.assertAlmostEqual(loss, 0.8, places=5)
        self.assertEqual(g, 0.0)
        loss, g = self._grad_for(math.log(1.1), +1.0)  # inside the clip range -> plain s*A
        self.assertAlmostEqual(loss, -1.1, places=5)
        self.assertGreater(g, 0.0)

    def test_zero_advantage_group_has_zero_gradient(self):
        lp = torch.randn(4, 3, requires_grad=True)
        adv = self.g.group_advantages(torch.ones(1, 4)).reshape(-1)
        self.g.gspo_loss(lp, lp.detach(), adv, torch.ones(4, 3), 3e-4, 4e-4).backward()
        self.assertEqual(float(lp.grad.abs().sum()), 0.0)

    def test_defaults_are_documented_assumptions(self):
        cfg = self.g.GSPOConfig()
        self.assertEqual((cfg.clip_low, cfg.clip_high, cfg.group_size), (3e-4, 4e-4, 8))
        self.assertIn("could not verify", self.g.__doc__)

    def test_diagnostics_report_clip_fraction(self):
        old = torch.zeros(2, 2)
        new = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        d = self.g.gspo_diagnostics(new, old, torch.tensor([1.0, -1.0]), torch.ones(2, 2), 0.2, 0.3)
        self.assertAlmostEqual(d["clip_frac"], 0.5)  # only the A>0 sequence is clipped


if __name__ == "__main__":
    unittest.main()
