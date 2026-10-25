"""Tiny-model end-to-end tests for helm.training (CPU, random-init Qwen3-config models, stdlib unittest only).

These are smoke tests: they show that the training code runs, that the losses move in the right direction and that the
bidirectional-attention switch really works on the installed transformers version. They say nothing about quality.
"""

import copy
import json
import os
import tempfile
import unittest
from unittest import mock

try:
    import torch
    import transformers

    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

if HAVE_DEPS:
    from helm.training import data_synthesis as ds
    from helm.training import embedding_ft as ef
    from helm.training import gspo, opd, thirds
    from helm.training.rewards import make_toy_reward_fn
    from helm.training.tiny import (
        ByteTokenizer,
        build_smoke_controller_setup,
        load_tokenizer,
        tiny_causal_lm,
        tiny_encoder,
    )

_CACHE = {}


def controller_setup():
    if "ctrl" not in _CACHE:
        _CACHE["ctrl"] = build_smoke_controller_setup(seed=0)
    return _CACHE["ctrl"]


def gspo_setup():
    if "gspo" not in _CACHE:
        _CACHE["gspo"] = build_smoke_controller_setup(seed=0, student_steps=100, with_teacher=False)
    return _CACHE["gspo"]


def embedding_data():
    if "emb" not in _CACHE:
        recs = ds.generate_offline_dataset(ds.SynthConfig(n=120, seed=0, long_frac=0.2, min_ctx_tokens=160, max_ctx_tokens=300))
        held, train = recs[:24], recs[24:]
        texts = [t for r in train for t in [r["query"], r["positive"]] + [n["text"] for n in r["negatives"]]]
        _CACHE["emb"] = (train, held, ByteTokenizer.train_bpe(texts, 300))
    return _CACHE["emb"]


def make_encoder(seed=0, **kw):
    _, _, tok = embedding_data()
    return ef.TextEncoder(tiny_encoder(seed, vocab_size=tok.vocab_size), tok, max_length=320, **kw)


@unittest.skipUnless(HAVE_DEPS, "torch/transformers not installed")
class TestTokenizer(unittest.TestCase):
    def test_roundtrip_and_bpe_shortens(self):
        texts = ["<op>READ</op><q>stock level of widget_12</q>", "héllo ✓ 123\n  x"]
        raw = ByteTokenizer()
        bpe = ByteTokenizer.train_bpe(texts * 5 + ["<op>WRITE</op>"] * 20, 40)
        for t in texts:
            self.assertEqual(raw.decode(raw.encode(t)), t)
            self.assertEqual(bpe.decode(bpe.encode(t)), t)
        self.assertLess(len(bpe.encode(texts[0])), len(raw.encode(texts[0])))
        with tempfile.TemporaryDirectory() as d:
            bpe.save_pretrained(d)
            self.assertEqual(load_tokenizer(d).encode(texts[0]), bpe.encode(texts[0]))


@unittest.skipUnless(HAVE_DEPS, "torch/transformers not installed")
class TestBidirectionalAttention(unittest.TestCase):
    """Verifies empirically (not by version assumption) that the causal mask is really removed."""

    def _pos0_change(self, backbone, mask=None, **kw):
        V = backbone.config.vocab_size
        a = torch.randint(0, V, (1, 10), generator=torch.Generator().manual_seed(1))
        b = a.clone()
        b[0, 5:] = (b[0, 5:] + 7) % V
        mask = torch.ones_like(a) if mask is None else mask
        with torch.no_grad():
            ha = backbone(input_ids=a, attention_mask=mask, **kw).last_hidden_state[0, 0]
            hb = backbone(input_ids=b, attention_mask=mask, **kw).last_hidden_state[0, 0]
        return float((ha - hb).abs().max())

    def test_default_decoder_is_causal_and_flag_makes_it_bidirectional(self):
        for attn in ("sdpa", "eager"):
            bb = tiny_encoder(0, attn_implementation=attn).eval()
            self.assertEqual(self._pos0_change(bb), 0.0, attn)  # causal: position 0 never sees later tokens
            ef.set_bidirectional(bb, True)
            if self._pos0_change(bb) == 0.0:  # transformers < 5: the flag alone is not enough; TextEncoder falls back to an explicit mask
                self.skipTest(f"config.is_causal=False does not remove the causal mask on transformers {transformers.__version__}")
            self.assertGreater(self._pos0_change(bb), 1e-3, attn)
            padded = torch.tensor([[1] * 8 + [0] * 2])
            if self._pos0_change(bb, padded) == 0.0:  # transformers < 5: a padding mask re-introduces causality
                self.skipTest(f"config.is_causal=False is not bidirectional with padding on transformers {transformers.__version__}")
            ef.set_bidirectional(bb, False)
            self.assertEqual(self._pos0_change(bb), 0.0, attn)

    def test_text_encoder_negotiates_and_probes(self):
        for attn in ("sdpa", "eager"):
            enc = ef.TextEncoder(tiny_encoder(0, attn_implementation=attn), ByteTokenizer())
            self.assertTrue(enc.probe_bidirectional(), attn)
            self.assertIn(enc.mask_mode, ("config", "explicit"))
            print(f"[transformers {transformers.__version__}, attn={attn}] bidirectional via mask_mode={enc.mask_mode!r}")
            causal = ef.TextEncoder(tiny_encoder(0, attn_implementation=attn), ByteTokenizer(), bidirectional=False)
            self.assertFalse(causal.probe_bidirectional(), attn)

    def test_explicit_4d_mask_fallback_works_even_if_flag_says_causal(self):
        for attn in ("sdpa", "eager"):
            bb = tiny_encoder(0, attn_implementation=attn)
            enc = ef.TextEncoder(bb, ByteTokenizer())
            ef.set_bidirectional(bb, False)  # sabotage the config route
            enc.mask_mode = "config"
            self.assertFalse(enc.probe_bidirectional(), attn)
            enc.mask_mode = "explicit"
            self.assertTrue(enc.probe_bidirectional(), attn)

    def test_negotiation_raises_when_nothing_works(self):
        with mock.patch.object(ef.TextEncoder, "probe_bidirectional", return_value=False):
            with self.assertRaises(RuntimeError):
                ef.TextEncoder(tiny_encoder(0), ByteTokenizer())

    def test_padding_does_not_change_embeddings(self):
        for mode in ("config", "explicit"):
            enc = ef.TextEncoder(tiny_encoder(0), ByteTokenizer())
            enc.mask_mode = mode
            if mode == "config" and not enc.probe_bidirectional():
                continue  # config route is not bidirectional on this transformers version (explicit mode covers it)
            alone = enc.encode(["short text"])
            batched = enc.encode(["short text", "a much longer text that forces padding of the first one"])
            self.assertTrue(abs(alone[0] - batched[0]).max() < 1e-5, mode)
            self.assertAlmostEqual(float((alone[0] ** 2).sum()), 1.0, places=4)  # L2-normalised

    def test_embedding_dim_comes_from_model_config(self):
        for hidden, heads, hd in ((48, 2, 24), (80, 4, 20)):
            enc = ef.TextEncoder(tiny_encoder(0, hidden_size=hidden, num_heads=heads, num_kv_heads=heads, head_dim=hd), ByteTokenizer())
            self.assertEqual(enc.dim, hidden)
            self.assertEqual(enc.encode(["abc", "def"]).shape, (2, hidden))


@unittest.skipUnless(HAVE_DEPS, "torch/transformers not installed")
class TestEmbeddingTraining(unittest.TestCase):
    def test_training_reduces_loss_and_improves_held_out_retrieval(self):
        train, held, _ = embedding_data()
        enc = make_encoder()
        before = ef.evaluate_epistemic(enc, held)
        cfg = ef.EmbeddingFTConfig(tau=0.05, lr=2e-3, warmup_steps=3, batch_size=16, epochs=5, seed=0)
        hist = ef.EmbeddingTrainer(enc, cfg).fit(train)
        after = ef.evaluate_epistemic(enc, held)
        first, last = thirds([h["loss"] for h in hist])
        print(f"[embedding smoke] loss {first:.3f} -> {last:.3f}; held-out {before} -> {after}")
        self.assertEqual(len(hist), 5 * (len(train) // 16))
        self.assertLess(last, 0.7 * first)
        self.assertGreater(after["retrieval_top1"], before["retrieval_top1"])
        self.assertGreater(after["epistemic_acc"], before["epistemic_acc"])
        self.assertTrue(enc.probe_bidirectional())  # still bidirectional after training
        self.assertAlmostEqual(hist[0]["lr"], cfg.lr * 1 / 3, places=9)  # warmup starts at 1/warmup_steps

    def test_checkpoint_roundtrip(self):
        enc = make_encoder(seed=1)
        texts = ["[step 3] tool_output: {\"code\": 200}", "another memory"]
        with tempfile.TemporaryDirectory() as d:
            enc.save(d)
            for f in ("helm_embedding.json", "USAGE.md", "config.json", "tiny_tokenizer.json"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
            with open(os.path.join(d, "config.json")) as f:
                self.assertIs(json.load(f)["is_causal"], False)  # the bidirectional flag is persisted
            with open(os.path.join(d, "helm_embedding.json")) as f:
                meta = json.load(f)
            self.assertEqual((meta["pooling"], meta["dim"]), ("weighted", enc.dim))
            loaded = ef.TextEncoder.load(d, dtype=torch.float32)
            self.assertTrue(loaded.probe_bidirectional())
            self.assertEqual(loaded.dim, enc.dim)
            self.assertLess(abs(loaded.encode(texts) - enc.encode(texts)).max(), 1e-6)
            self.assertLess(abs(loaded.encode(texts, instruction="x") - enc.encode(texts, instruction="x")).max(), 1e-6)

    def test_instruction_prefix_is_applied_to_queries_only(self):
        enc = make_encoder()
        q = enc.encode(["hello"], instruction="find logs")
        manual = enc.encode([ef.format_query("find logs", "hello")])
        self.assertLess(abs(q - manual).max(), 1e-6)
        self.assertEqual(ef.format_query("I", "T"), "Instruct: I\nQuery:T")
        self.assertGreater(abs(q - enc.encode(["hello"])).max(), 1e-4)

    def test_gradient_caching_matches_full_batch_gradients(self):
        train, _, _ = embedding_data()
        batch = train[:10]
        enc_a = make_encoder(seed=3)
        enc_b = copy.deepcopy(enc_a)
        base = dict(tau=0.05, batch_size=10, mask_same_seed=True)
        ta = ef.EmbeddingTrainer(enc_a, ef.EmbeddingFTConfig(**base))
        tb = ef.EmbeddingTrainer(enc_b, ef.EmbeddingFTConfig(micro_batch_size=7, **base))
        sa, sb = ta.loss_and_backward(batch), tb.loss_and_backward(batch)
        self.assertAlmostEqual(sa["loss"], sb["loss"], places=5)
        n = 0
        for pa, pb in zip(enc_a.parameters(), enc_b.parameters()):
            if pa.grad is None:
                continue
            n += 1
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-5, rtol=2e-3)
        self.assertGreater(n, 5)

    def test_unfrozen_full_finetuning_updates_every_layer(self):
        train, _, _ = embedding_data()
        enc = make_encoder(seed=4)
        before = {n: p.detach().clone() for n, p in enc.named_parameters()}
        ef.EmbeddingTrainer(enc, ef.EmbeddingFTConfig(tau=0.05, lr=1e-3, warmup_steps=0, batch_size=16, epochs=1)).train_step(train[:16])
        changed = [n for n, p in enc.named_parameters() if not torch.equal(p, before[n])]
        self.assertTrue(any("layers.0." in n for n in changed) and any("layers.1." in n for n in changed))


@unittest.skipUnless(HAVE_DEPS, "torch/transformers not installed")
class TestOPDSmoke(unittest.TestCase):
    def test_teacher_is_nontrivial(self):
        S = controller_setup()
        self.assertLess(S["teacher_losses"][-1], 0.5 * S["teacher_losses"][0])
        self.assertTrue(all(not p.requires_grad for p in S["teacher"].parameters()))

    def test_reverse_kl_decreases(self):
        S = controller_setup()
        teacher, tok = S["teacher"], S["tok"]
        student = copy.deepcopy(S["student"])
        t_before = sum(float(p.sum()) for p in teacher.parameters())
        cfg = opd.OPDConfig(iters=60, prompts_per_iter=8, samples_per_prompt=2, max_new_tokens=60, lr=3e-3, seed=0)
        trainer = opd.OPDTrainer(student, teacher, tok, cfg)
        hist = trainer.fit([r["prompt"] for r in S["records"]])
        kl = [h["mean_reverse_kl"] for h in hist]
        first, last = thirds(kl)
        print(f"[opd smoke] mean reverse KL first third {first:.3f} -> last third {last:.3f}")
        self.assertEqual(len(hist), 60)
        self.assertTrue(all(map(lambda x: x == x, kl)))  # no NaN
        self.assertLess(last, 0.8 * first)
        self.assertEqual(sum(float(p.sum()) for p in teacher.parameters()), t_before)  # teacher is never updated
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all("mean_reverse_kl" in h for h in hist))

    def test_sampling_is_ancestral_not_top_k_truncated(self):
        S = controller_setup()
        model = tiny_causal_lm(0, vocab_size=S["tok"].vocab_size)  # near-uniform next-token distribution
        roll = opd.sample_rollouts(model, S["tok"], [S["records"][0]["prompt"]] * 256, max_new_tokens=1)
        self.assertGreater(len({c[0] for c in roll.completion_ids}), 50)  # transformers' default top_k=50 would cap this

    def test_completion_logprobs_match_a_plain_forward_pass(self):
        S = controller_setup()
        model, tok = S["student"], S["tok"]
        rec = S["records"][:4]
        p = opd.encode_prompts(tok, [r["prompt"] for r in rec])
        c = opd.encode_completions(tok, [r["completion"] for r in rec])
        lp, mask = opd.completion_logprobs(model, p, c, tok.pad_token_id)
        lp_mb, _ = opd.completion_logprobs(model, p, c, tok.pad_token_id, micro_batch_size=3)
        torch.testing.assert_close(lp, lp_mb, atol=1e-4, rtol=1e-4)
        with torch.no_grad():
            for i in range(len(rec)):
                ref = torch.log_softmax(model(input_ids=torch.tensor([p[i] + c[i]])).logits[0].float(), -1)
                for j, t in enumerate(c[i]):
                    self.assertAlmostEqual(float(lp[i, j]), float(ref[len(p[i]) + j - 1, t]), places=4)
                self.assertEqual(int(mask[i].sum()), len(c[i]))

    def test_sampler_and_scorer_agree_on_left_padded_batches(self):
        # on-policy consistency: the log-prob generate() sampled with equals the one the trainer recomputes
        S = controller_setup()
        model, tok = S["student"].eval(), S["tok"]
        prompts = [S["records"][i]["prompt"] for i in (0, 1, 2)]
        p_ids = opd.encode_prompts(tok, prompts)
        self.assertGreater(len({len(p) for p in p_ids}), 1)  # genuinely different lengths -> left padding
        width = max(len(p) for p in p_ids)
        ids = torch.full((3, width), tok.pad_token_id)
        att = torch.zeros((3, width), dtype=torch.long)
        for i, p in enumerate(p_ids):
            ids[i, width - len(p):] = torch.tensor(p)
            att[i, width - len(p):] = 1
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=6, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                 return_dict_in_generate=True, output_logits=True)
        comps = []
        for row in out.sequences[:, width:].tolist():  # cut at the first EOS exactly like sample_rollouts does
            comps.append(row[: row.index(tok.eos_token_id) + 1] if tok.eos_token_id in row else row)
        sampler_lp = torch.stack(
            [torch.log_softmax(l.float(), -1).gather(-1, out.sequences[:, width + t:width + t + 1]).squeeze(-1)
             for t, l in enumerate(out.logits)], dim=1)
        scored, mask = opd.completion_logprobs(model, p_ids, comps, tok.pad_token_id)
        w = scored.size(1)
        torch.testing.assert_close(scored[mask], sampler_lp[:, :w][mask], atol=1e-4, rtol=1e-4)

    def test_teacher_scoring_has_no_gradient_and_matches_student_shape(self):
        S = controller_setup()
        roll = opd.sample_rollouts(S["student"], S["tok"], [r["prompt"] for r in S["records"][:6]], 20)
        t = opd.teacher_logprobs(S["teacher"], roll, S["tok"].pad_token_id, micro_batch_size=4)
        s, mask = opd.completion_logprobs(S["student"], roll.prompt_ids, roll.completion_ids, S["tok"].pad_token_id)
        self.assertFalse(t.requires_grad)
        self.assertEqual(t.shape, s.shape)
        self.assertTrue(bool((t[~mask] == 0).all()))


@unittest.skipUnless(HAVE_DEPS, "torch/transformers not installed")
class TestGSPOSmoke(unittest.TestCase):
    def _trainer(self, **kw):
        S = gspo_setup()
        base = dict(iters=40, prompts_per_iter=4, group_size=16, max_new_tokens=48, lr=1e-3, seed=0)
        base.update(kw)
        policy = copy.deepcopy(S["student"])
        return gspo.GSPOTrainer(policy, S["tok"], make_toy_reward_fn(), gspo.GSPOConfig(**base)), S["records"][:4]

    def test_mean_reward_increases(self):
        trainer, pool = self._trainer()
        hist = trainer.fit(pool)
        first, last = thirds([h["mean_reward"] for h in hist])
        print(f"[gspo smoke] mean hybrid reward first third {first:.3f} -> last third {last:.3f}")
        self.assertGreater(last, first + 0.02)
        self.assertGreater(hist[-1]["format_ok_frac"], 0.9)
        r = hist[0]
        self.assertAlmostEqual(r["mean_reward"], r["mean_r_task"] + 0.05 * r["mean_r_mem"], places=5)  # alpha=1, beta=0.05

    def test_on_policy_step_has_unit_ratio_and_multi_minibatch_clips(self):
        trainer, pool = self._trainer(iters=1)
        rec = trainer.step(pool)
        self.assertAlmostEqual(rec["mean_seq_ratio"], 1.0, places=4)
        self.assertEqual(rec["clip_frac"], 0.0)
        trainer, pool = self._trainer(n_minibatches=4)
        hist = trainer.fit(pool, iters=3)
        self.assertGreater(max(h["n_optimizer_steps"] for h in hist), 1)
        self.assertGreater(max(h["clip_frac"] for h in hist), 0.0)  # later minibatches are off-policy -> tiny clip range binds

    def test_reward_fn_dict_and_float_outputs(self):
        S = gspo_setup()
        pool = S["records"][:2]
        for fn in (lambda rec, text, n: 1.0 if "WRITE" in text else 0.0,
                   lambda rec, text, n: {"reward": float(len(text) % 3), "r_task": 0.0}):
            policy = copy.deepcopy(S["student"])
            tr = gspo.GSPOTrainer(policy, S["tok"], fn, gspo.GSPOConfig(prompts_per_iter=2, group_size=4, max_new_tokens=24, lr=1e-4))
            rec = tr.step(pool)
            self.assertIn("mean_reward", rec)


if __name__ == "__main__":
    unittest.main()
