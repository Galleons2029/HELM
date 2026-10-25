"""Tests for helm.benchmarks (stdlib unittest, MockLLM only: no network, no GPU, no data downloads).

Run from inside ``code/``:  ``python -m unittest tests.test_benchmarks``

Everything here exercises plumbing on tiny hand-built fixtures. Nothing asserts a paper number, and every fixture-driven
run is a ``mock``-labelled run by construction. The ``write_*`` / ``make_toy_env`` helpers are reused by the CLI smoke
checks (``scripts/run_benchmark.py``) in the final report.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from helm.benchmarks import bright, gaia, harness, lifelong, mind2web, qa_sanity, streambench, webarena  # noqa: E402
from helm.benchmarks.base import (  # noqa: E402
    EpisodeResult,
    MissingDataError,
    MissingDependencyError,
    ResultsWriter,
    aggregate_over_seeds,
    embedder_source,
    episode_row,
    llm_source,
    summarize,
)
from helm.benchmarks.harness import Agent, NullBackend, ToolRegistry, ToolResult, make_agent, parse_action, parse_tool_call  # noqa: E402
from helm.config import HelmConfig  # noqa: E402
from helm.embedding import HashingEmbedder  # noqa: E402
from helm.llm.client import MockLLM, OpenAICompatClient  # noqa: E402
from helm.types import Status  # noqa: E402

import run_benchmark  # noqa: E402  (scripts/run_benchmark.py)


# ---------------------------------------------------------------------------------------------- fixtures
def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def write_gaia_fixture(d: Path) -> Path:
    rows = [
        {"task_id": "g1", "Question": "What is 2+2?", "Level": 1, "Final answer": "4", "file_name": ""},
        {"task_id": "g2", "Question": "What is 12*12?", "Level": 2, "Final answer": "144", "file_name": ""},
        {"task_id": "g3", "question": "Name the capital of France.", "level": "3", "final_answer": "Paris", "file_name": "note.txt"},
        {"task_id": "g4", "Question": "Hidden test question", "Level": 2, "Final answer": "?", "file_name": ""},
    ]
    return write_jsonl(d / "metadata.jsonl", rows)


def m2w_task(aid: str, n_steps: int = 2, website: str = "shop.example"):
    """A Mind2Web-schema record. Step i has a gold element with a unique aria_label and two distractors; the positive
    candidate is deliberately listed first so tests can check that ranking never reads that flag."""
    actions = []
    for i in range(n_steps):
        def cand(nid, label, tag="button"):
            return {"tag": tag, "backend_node_id": nid, "attributes": json.dumps({"aria_label": label, "id": f"el{nid}", "bounding_box_rect": "1,2,3,4"})}
        pos = cand(f"{aid}-{i}-p", f"gold-{aid}-{i}")
        negs = [cand(f"{aid}-{i}-n{j}", f"decoy {j}") for j in range(2)]
        op = {"op": "TYPE", "original_op": "TYPE", "value": "red shoes"} if i == 0 else {"op": "CLICK", "original_op": "CLICK", "value": ""}
        actions.append(
            {
                "action_uid": f"{aid}-{i}",
                "operation": op,
                "pos_candidates": [pos],
                "neg_candidates": negs,
                "cleaned_html": f'<div backend_node_id="x"><button backend_node_id="{aid}-{i}-p">Gold {i}</button></div>',
            }
        )
    return {"annotation_id": aid, "website": website, "domain": "Shopping", "confirmed_task": f"find red shoes ({aid})", "actions": actions}


def write_mind2web_fixture(d: Path) -> Path:
    for split, (aid, n) in {"test_task": ("t1", 2), "test_website": ("w1", 3), "test_domain": ("d1", 2)}.items():
        sub = d / split
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"{split}_0.json").write_text(json.dumps([m2w_task(aid, n), m2w_task(aid + "b", 1)]), encoding="utf-8")
    return d


def write_stream_fixture(path: Path) -> Path:
    return write_jsonl(path, [{"id": i, "input": "capital of france", "output": "paris"} for i in range(4)])


def write_lifelong_fixture(path: Path) -> Path:
    return write_jsonl(
        path,
        [{"id": f"l{i}", "instruction": f"set key k{i} to {i}", "label": {"key": f"k{i}", "value": str(i)}, "env": "toykv"} for i in range(3)],
    )


def write_bright_fixture(path: Path) -> Path:
    docs = [
        {"id": "d_quantum", "content": "photons entanglement quantum experiment measurement"},
        {"id": "d_pasta", "content": "cooking pasta recipe boil water salt sauce"},
        {"id": "d_stocks", "content": "stock market bonds yield interest portfolio"},
        {"id": "d_garden", "content": "garden soil compost tomatoes watering seeds"},
    ]
    queries = [
        {"id": "q1", "query": "quantum photons entanglement", "gold_ids": ["d_quantum"], "excluded_ids": ["N/A"]},
        {"id": "q2", "query": "boil pasta with sauce recipe", "gold_ids": ["d_pasta"], "excluded_ids": ["N/A"]},
        {"id": "q3", "query": "bonds yield portfolio market", "gold_ids": ["d_stocks"], "excluded_ids": ["N/A"]},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"queries": queries, "documents": docs}), encoding="utf-8")
    return path


def write_qa_fixtures(d: Path):
    qs = [
        {"id": "m1", "question": "Which gas do plants absorb for photosynthesis?", "choices": ["Oxygen", "Carbon dioxide", "Helium", "Neon"], "answer": 1, "subject": "bio"},
        {"id": "m2", "question": "What is the capital of France?", "choices": ["Rome", "Paris", "Berlin", "Madrid"], "answer": "B", "subject": "geo"},
    ]
    corpus = [
        {"id": "c1", "text": "Plants absorb carbon dioxide gas during photosynthesis and release oxygen."},
        {"id": "c2", "text": "Paris is the capital and largest city of France."},
        {"id": "c3", "text": "Granite is an igneous rock."},
    ]
    return write_jsonl(d / "qa.jsonl", qs), write_jsonl(d / "corpus.jsonl", corpus)


class ToyKVEnv:
    """Tiny LifelongEnv: the goal is to set a key to a value with ``set <key> <value>``."""

    name = "toykv"

    def __init__(self):
        self.store = {}

    def reset(self, task):
        self.store = {}
        return f"Store is empty. Goal: {task.prompt}"

    def step(self, action):
        parts = action.split()
        if len(parts) == 3 and parts[0] == "set":
            self.store[parts[1]] = parts[2]
            return lifelong.StepOutcome(f"ok {parts[1]}={parts[2]}", 200)
        return lifelong.StepOutcome(f"error: bad action {action!r}", 400)

    def verify_label(self, task, answer):
        return self.store.get(task.gold["key"]) == task.gold["value"]


def make_toy_env():
    return ToyKVEnv()


def toy_policy(bad_keys=()):
    """Scripted reasoner for ToyKVEnv: one env call per task (a bad call for keys in ``bad_keys``), then answer."""

    def fn(messages):
        content = messages[-1]["content"]
        latest = content.split("Latest observation:")[1] if "Latest observation:" in content else ""
        if "[step" in latest:  # the env call has been answered
            return "<answer>done</answer>"
        task_block = content.split("Task:\n", 1)[1].split("\n\nRecent observations:")[0].split("\n\nLatest observation:")[0]
        key, val = re.findall(r"set key (\w+) to (\w+)", task_block)[-1]  # replayed examples come first, the live task last
        return '<tool>env {"action": "oops"}</tool>' if key in bad_keys else f'<tool>env {{"action": "set {key} {val}"}}</tool>'

    return MockLLM(fn)


class _CapturingClient:
    """Not a MockLLM and not a known real client: records decoding kwargs."""

    def __init__(self, reply="<answer>x</answer>"):
        self.reply, self.gens = reply, []

    def chat(self, messages, **gen):
        self.gens.append(gen)
        return self.reply


class _BigBackend(NullBackend):
    """Returns far more recall text than the cap, to test the harness-level truncation."""

    def recall(self, query, ctx):
        from helm.backends import RecallResult, RecalledItem

        text = "\n".join(f"fact number {i} says something long enough" for i in range(60))
        return RecallResult(items=[RecalledItem(text.splitlines()[0]), RecalledItem("fact number 59 says something long enough")], text=text, tokens=1)


def agent_for(method, llm, cfg=None, **kw):
    return make_agent(method, cfg or HelmConfig(), llm=llm, seed=0, **kw)


# ---------------------------------------------------------------------------------------------- base / writer
class TestResultsWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "r.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_rows_without_valid_source(self):
        w = ResultsWriter(self.path)
        for bad in ({"a": 1}, {"a": 1, "source": "real"}, {"a": 1, "source": ""}, {"a": 1, "source": None}):
            with self.assertRaises(ValueError):
                w.append(bad)
        self.assertFalse(self.path.exists())

    def test_header_grows_and_source_is_a_leading_column(self):
        w = ResultsWriter(self.path)
        w.append({"score": 1.0, "source": "mock", "benchmark": "x"})
        w.append({"score": 2.0, "source": "mock", "extra_col": "e", "benchmark": "x"})
        with open(self.path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0].keys())[:2], ["benchmark", "source"])
        self.assertEqual(rows[0]["extra_col"], "")
        self.assertEqual(rows[1]["extra_col"], "e")

    def test_refuses_to_mix_into_unlabelled_csv(self):
        self.path.write_text("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            ResultsWriter(self.path).append({"a": 3, "source": "mock"})

    def test_llm_source_labels(self):
        self.assertEqual(llm_source(MockLLM(["x"])), "mock")
        real = object.__new__(OpenAICompatClient)  # no network, no openai import needed
        self.assertEqual(llm_source(real), "measured")
        self.assertEqual(llm_source(_CapturingClient()), "mock")  # unknown clients are never "measured"

        class Declared(_CapturingClient):
            is_real_llm = True

        self.assertEqual(llm_source(Declared()), "measured")

        class MockWrapper(MockLLM):  # a MockLLM subclass claiming to be real is still a mock
            is_real_llm = True

        self.assertEqual(llm_source(MockWrapper(["x"])), "mock")
        self.assertEqual(embedder_source(HashingEmbedder(32)), "mock")

    def test_agent_with_mockllm_is_never_measured(self):
        a = agent_for("flat_rag", MockLLM(["<answer>x</answer>"]))
        self.assertEqual(a.source, "mock")

    def test_summaries_and_seed_aggregation(self):
        rs = [EpisodeResult(1.0, True, ctu=10, steps=2, extra={"epi_consumed": 2, "epi_invalid": 1}),
              EpisodeResult(0.0, False, ctu=30, steps=2, extra={"epi_consumed": 2, "epi_invalid": 0})]
        s = summarize(rs)
        self.assertEqual(s["n"], 2)
        self.assertAlmostEqual(s["score"], 0.5)
        self.assertAlmostEqual(s["ctu_per_task"], 20.0)
        self.assertAlmostEqual(s["ctu_per_step"], 10.0)
        self.assertAlmostEqual(s["epi_err_pct"], 25.0)  # pooled 1/4, not mean of per-episode rates
        agg = aggregate_over_seeds([{"score": 0.2}, {"score": 0.4}, {"score": 0.6}], keys=("score",))
        self.assertAlmostEqual(agg["score"], 0.4)
        self.assertAlmostEqual(agg["score_se"], 0.2 / 3 ** 0.5, places=6)

    def test_episode_row_keeps_scalars_and_nests_the_rest(self):
        r = EpisodeResult(1.0, True, extra={"a": 1, "b": {"c": 2}}, task_id="t")
        row = episode_row(r, source="mock", seed=0)
        self.assertEqual(row["x_a"], 1)
        self.assertEqual(row["extra_json"], {"b": {"c": 2}})


# ---------------------------------------------------------------------------------------------- harness
class TestParsing(unittest.TestCase):
    def test_parse_action(self):
        self.assertEqual(parse_action("<answer> 42 </answer>").payload, "42")
        self.assertEqual(parse_action("hmm <recall>what is x</recall>").kind, "recall")
        self.assertEqual(parse_action('<tool>f {"a": 1}</tool>').kind, "tool")
        self.assertEqual(parse_action("<recall>q</recall> then <answer>no</answer>").kind, "recall")  # earliest wins
        self.assertEqual(parse_action("<think>I will say <answer>wrong</answer></think><answer>right</answer>").payload, "right")
        self.assertEqual(parse_action("reasoning</think><answer>ok</answer>").payload, "ok")  # opening tag stripped by template
        self.assertEqual(parse_action("<answer>cut off").payload, "cut off")
        self.assertEqual(parse_action("just chatting").kind, "none")
        self.assertEqual(parse_action("<think>never closed <answer>x</answer>").kind, "none")

    def test_parse_tool_call(self):
        self.assertEqual(parse_tool_call('search {"q": "a b"}'), ("search", {"q": "a b"}, None))
        self.assertEqual(parse_tool_call("ping"), ("ping", {}, None))
        self.assertIsNotNone(parse_tool_call("search {bad json}")[2])
        self.assertIsNotNone(parse_tool_call("search [1, 2]")[2])
        self.assertIsNotNone(parse_tool_call("")[2])

    def test_truncate_to_tokens(self):
        c = harness.TokenCounter()
        text = "\n".join(["one two three"] * 10)
        out, cut = harness.truncate_to_tokens(text, 7, c)
        self.assertTrue(cut)
        self.assertLessEqual(c.count(out), 7)
        self.assertEqual(harness.truncate_to_tokens("a b", 5, c), ("a b", False))
        one_line, cut = harness.truncate_to_tokens("w " * 50, 5, c)
        self.assertTrue(cut)
        self.assertLessEqual(c.count(one_line), 5)


class TestHarness(unittest.TestCase):
    def kb_tools(self):
        return ToolRegistry().add("kb", lambda args: ToolResult(200, "capital of France = Paris"), "lookup facts")

    def test_recall_round_trip(self):
        llm = MockLLM(['<tool>kb {"q": "capital"}</tool>', "<recall>capital of France</recall>", "<answer>Paris</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("What is the capital of France?", self.kb_tools())
        self.assertEqual((run.answer, run.terminated, run.steps), ("Paris", "answer", 3))
        self.assertEqual(run.n_recalls, 1)
        # the <memory> block reaches the model on the *next* call, with the recalled text inside it
        last_prompt = llm.calls[2][1]["content"]
        self.assertIn("<memory>", last_prompt)
        self.assertIn("capital of France = Paris", last_prompt.split("<memory>")[1])
        rec = [e for e in run.events if e["type"] == "recall"][0]
        self.assertGreater(rec["tokens"], 0)
        self.assertEqual(rec["items"][0]["audit_ids"], ["kb:L1"])
        self.assertEqual(run.epi.consumed, 1)
        self.assertEqual(run.epi.invalid, 0)
        calls = [e for e in run.events if e["type"] == "call"]
        self.assertEqual(calls[2]["retrieved_tokens"], rec["tokens"])  # billed once, in the consuming call

    def test_ctu_accounting_matches_what_is_sent(self):
        llm = MockLLM(['<tool>kb {}</tool>', '<tool>kb {}</tool>', '<tool>kb {}</tool>', '<tool>kb {}</tool>', "<answer>x</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("q", self.kb_tools())
        calls = [e for e in run.events if e["type"] == "call"]
        self.assertEqual(sum(c["ctu"] for c in calls), run.ctu.total)
        self.assertEqual(len(run.ctu.per_step), run.steps)
        sys_t = run.ctu.system_tokens
        for c in calls:
            self.assertEqual(c["ctu"], sys_t + c["hist_tokens"] + c["retrieved_tokens"] + c["tool_obs_tokens"])
            self.assertEqual(c["hist_tokens"], c["hist_rendered_tokens"])  # the history shown == the history billed
        # working window is 3 raw observations: history never exceeds 2 prior + 1 pending observation
        obs_tokens = [e["tokens"] for e in run.events if e["type"] == "tool"]
        self.assertLessEqual(max(c["hist_tokens"] for c in calls), 3 * max(obs_tokens))

    def test_failed_tool_becomes_failed_status_and_is_audited(self):
        tools = ToolRegistry().add("flaky", lambda args: ToolResult(500, "upstream exploded"), "always fails")
        llm = MockLLM(['<tool>flaky {"x": 1}</tool>', "<recall>upstream exploded</recall>", "<answer>gave up</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("do the thing", tools)
        obs = agent.audit.obs["flaky:L1"]
        self.assertEqual(obs.status, Status.FAILED)
        self.assertEqual(obs.error_payload, "upstream exploded")
        self.assertEqual(run.n_failed_tools, 1)
        # the failed trace is recalled by the (ungoverned) baseline and the audit, which is independent of it, flags it
        self.assertEqual(run.epi.consumed, 1)
        self.assertEqual(run.epi.by_reason["failed"], 1)
        self.assertGreater(run.epi.rate, 99.0)
        self.assertEqual([e["code"] for e in run.events if e["type"] == "tool"], [500])

    def test_tool_exception_unknown_tool_and_bad_json_are_failed_observations(self):
        def boom(args):
            raise RuntimeError("kaput")

        tools = ToolRegistry().add("boom", boom)
        llm = MockLLM(["<tool>boom {}</tool>", "<tool>nosuch {}</tool>", "<tool>boom {bad}</tool>", "<answer>x</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("t", tools)
        self.assertEqual([e["code"] for e in run.events if e["type"] == "tool"], [500, 404, 400])
        self.assertTrue(all(o.status == Status.FAILED for o in agent.audit.obs.values()))

    def test_max_steps_termination(self):
        llm = MockLLM(['<tool>kb {}</tool>'])  # never answers
        agent = agent_for("flat_rag", llm, max_steps=4)
        run = agent.run("loop forever", self.kb_tools())
        self.assertEqual((run.terminated, run.answer, run.steps), ("max_steps", None, 4))
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(len(run.ctu.per_step), 4)
        res = run.to_result(0.0, False, "t")
        self.assertEqual(res.extra["terminated"], "max_steps")
        self.assertEqual(res.steps, 4)

    def test_format_error_is_nudged_not_logged_as_memory(self):
        llm = MockLLM(["I am thinking out loud", "<answer>ok</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("t")
        self.assertEqual(run.answer, "ok")
        self.assertIn("Format error", llm.calls[1][1]["content"])
        self.assertEqual(len(agent.audit.obs), 0)

    def test_decoding_is_locked_to_config_and_seed_is_forwarded(self):
        client = _CapturingClient()
        cfg = HelmConfig()
        make_agent("none", cfg, llm=client, seed=7).run("t")
        g = client.gens[0]
        self.assertEqual((g["temperature"], g["top_p"], g["max_tokens"], g["seed"]), (cfg.reasoner.temperature, cfg.reasoner.top_p, cfg.reasoner.max_tokens, 7))
        cfg2 = cfg.override(**{"reasoner.temperature": 0.5})
        client2 = _CapturingClient()
        make_agent("none", cfg2, llm=client2, seed=0).run("t")
        self.assertEqual(client2.gens[0]["temperature"], 0.5)

    def test_matched_budgets_are_enforced(self):
        cfg = HelmConfig()
        backend = harness.build_method("flat_rag", cfg.override(**{"memory.k1": 9}))
        with self.assertRaises(ValueError):
            Agent(backend, MockLLM(["x"]), cfg)
        Agent(backend, MockLLM(["x"]), cfg, strict_budgets=False)  # explicit opt-out is allowed

    def test_recall_token_cap_applies_to_any_backend(self):
        cfg = HelmConfig().override(**{"memory.max_recall_tokens": 25})
        agent = Agent(_BigBackend(), MockLLM(["<recall>x</recall>", "<answer>y</answer>"]), cfg)
        run = agent.run("t")
        self.assertEqual(run.n_truncated_recalls, 1)
        rec = [e for e in run.events if e["type"] == "recall"][0]
        self.assertLessEqual(rec["tokens"], 25)
        self.assertEqual(len(rec["items"]), 1)  # only the item whose text survived truncation counts as consumed

    def test_keep_memory_persists_store_audit_and_step_counter(self):
        llm = MockLLM(["<tool>kb {}</tool>", "<answer>a</answer>", "<recall>capital of France</recall>", "<answer>b</answer>"])
        agent = agent_for("flat_rag", llm)
        agent.run("first task", self.kb_tools(), keep_memory=True)
        run2 = agent.run("second task", keep_memory=True)
        self.assertEqual(run2.epi.consumed, 1)  # recalled the first task's trace
        self.assertIn("kb:L1", agent.audit.obs)
        self.assertGreater(agent._gstep, 2)
        llm.script = ["<answer>z</answer>"]
        agent.run("fresh", keep_memory=False)
        self.assertEqual(len(agent.audit.obs), 0)
        self.assertEqual(agent._gstep, 1)

    def test_version_stamping_makes_old_traces_stale_in_the_audit(self):
        calls = iter([ToolResult(200, "price = 5", version="v1"), ToolResult(200, "stock = 7", version="v2")])
        tools = ToolRegistry().add("api", lambda args: next(calls))
        llm = MockLLM(["<tool>api {}</tool>", "<tool>api {}</tool>", "<recall>price</recall>", "<answer>x</answer>"])
        agent = agent_for("flat_rag", llm)
        run = agent.run("t", tools)
        self.assertEqual(agent.audit.obs["api:L1"].task_version, "v1")
        self.assertEqual(agent.audit.obs["api:L2"].task_version, "v2")
        self.assertGreaterEqual(run.epi.by_reason["stale"], 1)

    def test_context_overflow_is_reported_without_calling_the_model(self):
        cfg = HelmConfig().override(**{"reasoner.context_window": 10})
        llm = MockLLM(["<answer>x</answer>"])
        run = make_agent("none", cfg, llm=llm).run("a long task description " * 5)
        self.assertEqual(run.terminated, "context_overflow")
        self.assertTrue(run.overflow)
        self.assertEqual(llm.calls, [])

    def test_prefetch_recall_injects_memory_into_the_first_call(self):
        llm = MockLLM(["<answer>ok</answer>"])
        agent = agent_for("flat_rag", llm)
        agent.ingest("kb", "capital of France = Paris", Status.SUCCESS)
        run = agent.run("capital of France?", keep_memory=True, prefetch_query="capital of France")
        self.assertIn("<memory>", llm.calls[0][1]["content"])
        self.assertEqual(run.n_recalls, 1)
        self.assertGreater(run.ctu.per_step[0], 0)

    def test_epierr_is_measured_by_the_audit_not_by_the_backend_under_test(self):
        """Same scripted behaviour for every backend: a verified value, then a failed call carrying a different value."""

        def scenario():
            calls = iter([
                ToolResult(200, '{"entity": "acme", "attr": "price", "value": 5}'),
                ToolResult(500, '{"entity": "acme", "attr": "price", "value": 9, "error": "timeout"}'),
            ])
            return ToolRegistry().add("api", lambda args: next(calls))

        def epi(method):
            llm = MockLLM(["<tool>api {}</tool>", "<tool>api {}</tool>", "<recall>acme price</recall>", "<answer>x</answer>"])
            run = agent_for(method, llm).run("price of acme?", scenario())
            return run.epi

        flat, governed, ungoverned = epi("flat_rag"), epi("helm"), epi("helm_no_gov")
        self.assertEqual(flat.by_reason["failed"], 1)  # the failed trace is recalled and the audit flags it
        self.assertEqual(governed.invalid, 0)  # governance filtered it before it reached the model
        self.assertEqual(ungoverned.invalid, 1)  # same store with governance off: the failure is consumed again
        self.assertGreaterEqual(governed.consumed, 1)

    def test_helm_backend_runs_under_the_same_scaffold(self):
        llm = MockLLM(['<tool>kb {}</tool>', "<recall>capital of France</recall>", "<answer>Paris</answer>"])
        agent = agent_for("helm", llm)
        run = agent.run("capital?", self.kb_tools())
        self.assertEqual(run.answer, "Paris")
        self.assertEqual(run.epi.consumed >= 1, True)


# ---------------------------------------------------------------------------------------------- GAIA
class TestGaia(unittest.TestCase):
    def test_scorer_numbers(self):
        m = gaia.gaia_match
        self.assertTrue(m("$89,706.00", "89706.00"))  # the paper's worked example (dollar sign, comma, trailing zeros)
        self.assertTrue(m("50%", "50"))
        self.assertTrue(m("1,234", "1234"))
        self.assertTrue(m(" 3.0 ", "3"))
        self.assertTrue(m("-2", "-2.0"))
        self.assertFalse(m("3.1", "3"))
        self.assertFalse(m("three", "3"))
        self.assertFalse(m("", "3"))
        self.assertFalse(m(None, "3"))
        self.assertFalse(m("12 apples", "12"))

    def test_scorer_strings(self):
        m = gaia.gaia_match
        self.assertTrue(m("  Paris. ", "paris"))
        self.assertTrue(m("New  York", "newyork"))
        self.assertTrue(m("St. Petersburg!", "st petersburg"))
        self.assertFalse(m("Lyon", "Paris"))

    def test_scorer_lists(self):
        m = gaia.gaia_match
        self.assertTrue(m("red, green ;blue", "Red,Green,Blue"))
        self.assertTrue(m("1, 2.0, abc", "1,2,ABC"))
        self.assertFalse(m("red,green", "red,green,blue"))  # length mismatch
        self.assertFalse(m("red,green,blue", "red,green"))
        self.assertFalse(m("1,3", "1,2"))
        self.assertTrue(m("x-y, z", "X-Y,z"))  # list elements keep punctuation (documented judgement call)
        self.assertFalse(m("xy, z", "X-Y,z"))

    def test_clean_answer(self):
        self.assertEqual(gaia.clean_answer("FINAL ANSWER: 42"), "42")
        self.assertEqual(gaia.clean_answer("  final answer :  blue "), "blue")
        self.assertIsNone(gaia.clean_answer(None))

    def test_loader_levels_hub_names_and_unscored_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_gaia_fixture(Path(d))
            tasks = gaia.load_gaia(path)  # default levels (2, 3), answerless test rows dropped
            self.assertEqual([t.task_id for t in tasks], ["g2", "g3"])
            self.assertEqual(tasks[1].gold, "Paris")
            self.assertEqual(tasks[1].meta["file_name"], "note.txt")
            self.assertEqual(len(gaia.load_gaia(path, levels=None)), 3)
            self.assertEqual(len(gaia.load_gaia(path, levels=None, require_gold=False)), 4)
            self.assertEqual(len(gaia.load_gaia(path, levels=(3,))), 1)
            self.assertEqual(len(gaia.load_gaia(path, limit=1)), 1)
            self.assertEqual(len(gaia.load_gaia(Path(d))), 2)  # directory containing metadata.jsonl
            with self.assertRaises(MissingDataError):
                gaia.load_gaia(path, levels=(9,))  # nothing left after filtering
            with self.assertRaises(MissingDataError):
                gaia.load_gaia(Path(d) / "nope.jsonl")

    def test_calculator_and_file_sandbox(self):
        self.assertEqual(gaia.safe_eval("2 + 3 * 4"), 14)
        self.assertAlmostEqual(gaia.safe_eval("sqrt(16) + pi"), 4 + math.pi)
        for bad in ("__import__('os').system('true')", "open('x')", "2 ** 100000", "x + 1"):
            with self.assertRaises((ValueError, SyntaxError)):
                gaia.safe_eval(bad)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("hello")
            tools = gaia.default_tools(d)
            self.assertEqual(tools.call("read_file", {"name": "a.txt"}).output, "hello")
            self.assertEqual(tools.call("read_file", {"name": "../etc/passwd"}).code, 403)
            self.assertEqual(tools.call("read_file", {"name": "missing.txt"}).code, 404)
            self.assertEqual(tools.call("calculator", {"expression": "1/0"}).code, 422)
            self.assertEqual(tools.call("calculator", {"expression": "7*6"}).output, "42")

    def test_adapter_episode_and_level_summary(self):
        with tempfile.TemporaryDirectory() as d:
            tasks = gaia.load_gaia(write_gaia_fixture(Path(d)))
        tasks[0].prompt, tasks[0].gold = "What is 6*7?", "42"

        def policy(messages):
            c = messages[-1]["content"]
            if "Latest observation:" in c:
                return "<answer>FINAL ANSWER: 42</answer>"
            return '<tool>calculator {"expression": "6*7"}</tool>'

        ad = gaia.GaiaAdapter()
        agent = agent_for("flat_rag", MockLLM(policy))
        results = ad.run_all(tasks[:1], agent)
        self.assertEqual(results[0].score, 1.0)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].extra["prediction"], "42")
        self.assertEqual(results[0].extra["level"], 2)
        wrong = ad.run_all(tasks[1:], agent_for("flat_rag", MockLLM(["<answer>London</answer>"])))
        self.assertEqual(wrong[0].score, 0.0)
        rows = ad.summarize(tasks, results + wrong)
        self.assertEqual([r["split"] for r in rows], ["all", "L2", "L3"])
        self.assertEqual(rows[1]["score"], 1.0)
        self.assertEqual(rows[2]["score"], 0.0)


# ---------------------------------------------------------------------------------------------- WebArena
class TestWebArena(unittest.TestCase):
    RAW = {
        "task_id": 7, "intent": "Find the price of the item", "sites": ["shopping"], "start_url": "__SHOPPING__/index",
        "require_login": True, "storage_state": "./.auth/shopping_state.json", "require_reset": False,
        "intent_template": "Find the price of {{x}}",
        "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "$12.00"}},
    }

    def test_parse_config_and_placeholders(self):
        cfg = webarena.parse_config(self.RAW, "7.json", env={"SHOPPING": "http://shop:7770"})
        self.assertEqual((cfg.task_id, cfg.intent, cfg.start_url), (7, "Find the price of the item", "http://shop:7770/index"))
        self.assertEqual(cfg.eval_types, ["string_match"])
        self.assertEqual(cfg.reference_answers, {"exact_match": "$12.00"})
        self.assertTrue(cfg.require_login)
        left = webarena.parse_config(self.RAW, env={}).start_url
        self.assertEqual(left, "__SHOPPING__/index")  # unset variables are left alone, not silently blanked
        with self.assertRaises(MissingDataError):
            webarena.parse_config({"intent": "x"})

    def test_load_configs_directory_and_list(self):
        with tempfile.TemporaryDirectory() as d:
            for i in (10, 2):
                (Path(d) / f"{i}.json").write_text(json.dumps({**self.RAW, "task_id": i}))
            (Path(d) / "test.raw.json").write_text("[]")  # ignored: not an <id>.json task file
            cfgs = webarena.load_task_configs(d, env={})
            self.assertEqual([c.task_id for c in cfgs], [2, 10])
            self.assertTrue(cfgs[0].config_file.endswith("2.json"))
            lst = Path(d) / "list.json"
            lst.write_text(json.dumps([self.RAW, {**self.RAW, "task_id": 8}]))
            from_list = webarena.load_task_configs(lst, limit=1, env={})
            self.assertEqual(len(from_list), 1)
            self.assertIsNone(from_list[0].config_file)
            with self.assertRaises(MissingDataError):
                webarena.load_task_configs(Path(d) / "missing")
            empty = Path(d) / "empty"
            empty.mkdir()
            with self.assertRaises(MissingDataError):
                webarena.load_task_configs(empty)

    def test_action_parsing(self):
        p = webarena.parse_action_string
        self.assertEqual(p("click [1234]"), webarena.WebAction("click", element_id="1234"))
        self.assertEqual(p("hover [5]").kind, "hover")
        a = p("type [12] [red shoes] [0]")
        self.assertEqual((a.element_id, a.text, a.press_enter), ("12", "red shoes", False))
        a = p("type [12] [red shoes]")
        self.assertEqual((a.text, a.press_enter), ("red shoes", True))  # Enter is pressed by default
        a = p("type [3] [a [1]]")
        self.assertEqual(a.text, "a [1]")
        self.assertEqual(p("press [Ctrl+v]").key, "Ctrl+v")
        self.assertEqual(p("scroll [down]").direction, "down")
        self.assertEqual(p("tab_focus [2]").tab_index, 2)
        self.assertEqual(p("goto [http://x.test/a?b=c]").url, "http://x.test/a?b=c")
        self.assertEqual(p("stop [$279.49]").answer, "$279.49")
        self.assertEqual(p("stop [N/A]").answer, "N/A")
        for bare in ("new_tab", "close_tab", "go_back", "go_forward"):
            self.assertEqual(p(bare).kind, bare)
        self.assertEqual(p("In summary, the next action I will perform is ```click [77]```").element_id, "77")
        for bad in ("", "dance [1]", "click 12", "scroll [sideways]", "tab_focus [x]", "go_back [1]", "type [1]"):
            with self.assertRaises(webarena.ActionParseError):
                p(bad)

    def test_evaluate_requires_official_validators_and_says_how_to_get_them(self):
        with mock.patch.dict(sys.modules, {"browser_env": None, "evaluation_harness": None}):
            with self.assertRaises(MissingDependencyError) as cm:
                webarena.evaluate("1.json", [])
        self.assertIn("web-arena-x/webarena", str(cm.exception))
        with self.assertRaises(webarena.BenchmarkError):
            webarena.evaluate("", [])

    def test_check_environment_names_missing_variables(self):
        with self.assertRaises(MissingDependencyError) as cm:
            webarena.check_environment({"SHOPPING": "x"})
        self.assertIn("REDDIT", str(cm.exception))
        self.assertNotIn("SHOPPING,", str(cm.exception))
        webarena.check_environment({v: "x" for v in webarena.WEBARENA_ENV_VARS})

    def test_adapter_control_flow_with_a_scripted_environment(self):
        """Wiring only (a scripted fake env, mock-labelled): the real browser path is Not validated against the real environment."""

        class FakeEnv:
            page = None

            def __init__(self):
                self.trajectory_, self.actions = [], []

            @property
            def trajectory(self):
                return self.trajectory_

            def reset(self, config):
                return "[1] button 'Buy'"

            def step(self, action):
                self.actions.append(action)
                self.trajectory_.append(action)
                return "page after " + action.kind, action.kind == "stop"

        envs = []

        def factory():
            envs.append(FakeEnv())
            return envs[-1]

        seen = {}

        def fake_eval(config_file, trajectory, page):
            seen["traj"], seen["cfg"] = list(trajectory), config_file
            return 1.0 if any(getattr(a, "answer", None) == "$12.00" for a in trajectory) else 0.0

        ad = webarena.WebArenaAdapter(env_factory=factory, evaluate_fn=fake_eval)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "7.json").write_text(json.dumps(self.RAW))
            tasks = ad.load(d)
        llm = MockLLM(['<tool>browser {"action": "click [1]"}</tool>', '<tool>browser {"action": "dance"}</tool>', "<answer>$12.00</answer>"])
        res = ad.run_episode(tasks[0], agent_for("flat_rag", llm))
        self.assertEqual(res.score, 1.0)
        kinds = [a.kind for a in envs[0].actions]
        self.assertEqual(kinds, ["click", "stop"])  # the malformed action never reached the env; stop closes the trajectory
        self.assertEqual(res.extra["n_failed_tools"], 1)  # the parse error is a FAILED observation (code 400)
        self.assertTrue(seen["cfg"].endswith("7.json"))


# ---------------------------------------------------------------------------------------------- Mind2Web
class TestMind2Web(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = write_mind2web_fixture(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loader_splits_limit_and_aliases(self):
        tasks = mind2web.load_mind2web(self.root)
        self.assertEqual(sorted({t.split for t in tasks}), ["test_domain", "test_task", "test_website"])
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len(mind2web.load_mind2web(self.root, limit=1)), 3)  # per split
        cw = mind2web.load_mind2web(self.root, split="CW")
        self.assertEqual({t.split for t in cw}, {"test_website"})
        t = next(t for t in tasks if t.task_id == "w1")
        self.assertEqual(len(t.gold), 3)
        self.assertEqual(t.gold[0].op, "TYPE")
        self.assertEqual(t.gold[0].value, "red shoes")
        self.assertEqual(t.gold[0].pos_ids, {"w1-0-p"})
        self.assertEqual(t.gold[0].candidates[0].inner_text, "Gold 0")  # recovered from cleaned_html via backend_node_id
        self.assertIn("[button] gold-w1-0 -> TYPE red shoes", t.gold[0].action_repr)
        single = mind2web.load_mind2web(self.root / "test_task" / "test_task_0.json")
        self.assertEqual({t.split for t in single}, {"test_task"})
        with self.assertRaises(MissingDataError):
            mind2web.load_mind2web(self.root / "nope")
        with self.assertRaises(MissingDataError):
            mind2web.parse_task({"annotation_id": "x"})

    def test_op_f1(self):
        self.assertEqual(mind2web.op_f1("CLICK", "CLICK"), 1.0)
        self.assertEqual(mind2web.op_f1("TYPE red shoes", "TYPE red shoes"), 1.0)
        self.assertAlmostEqual(mind2web.op_f1("TYPE red shoes", "TYPE red"), 0.8)  # P=2/3, R=1
        self.assertEqual(mind2web.op_f1("CLICK", "TYPE hello"), 0.0)
        self.assertEqual(mind2web.op_f1("", ""), 1.0)
        self.assertEqual(mind2web.op_f1("", "CLICK"), 0.0)
        self.assertAlmostEqual(mind2web.op_f1("a a b", "a b b", multiset=True), 2 / 3)

    def test_step_and_task_scoring(self):
        steps = mind2web.load_mind2web(self.root, split="CT")[0].gold  # TYPE red shoes, then CLICK
        good = [mind2web.Prediction(s.candidates[0].node_id, s.op, s.value) for s in steps]
        m = mind2web.score_task(steps, good)
        self.assertEqual((m["ele_acc"], m["op_f1"], m["step_sr"], m["sr"]), (1.0, 1.0, 1.0, 1.0))
        wrong_op = [mind2web.Prediction(steps[0].candidates[0].node_id, "TYPE", "blue shoes"), good[1]]
        m = mind2web.score_task(steps, wrong_op)
        self.assertEqual(m["ele_acc"], 1.0)
        self.assertLess(m["op_f1"], 1.0)
        self.assertEqual((m["step_sr"], m["sr"]), (0.5, 0.0))  # element right but value wrong: step fails
        none_of_above = [mind2web.Prediction(None, s.op, s.value) for s in steps]
        m = mind2web.score_task(steps, none_of_above)
        self.assertEqual((m["ele_acc"], m["op_f1"], m["step_sr"]), (0.0, 1.0, 0.0))
        m = mind2web.score_task(steps, good[:1])  # missing step counts as wrong
        self.assertEqual((m["step_sr"], m["sr"]), (0.5, 0.0))
        with self.assertRaises(ValueError):
            mind2web.score_task([], [])

    def test_macro_vs_micro_aggregation(self):
        a = {"ele_acc": 1.0, "op_f1": 1.0, "step_sr": 1.0, "sr": 1.0, "n_steps": 1, "n_ele": 1, "f1_sum": 1.0, "n_step_ok": 1}
        b = {"ele_acc": 0.0, "op_f1": 0.0, "step_sr": 0.0, "sr": 0.0, "n_steps": 3, "n_ele": 0, "f1_sum": 0.0, "n_step_ok": 0}
        agg = mind2web.aggregate([a, b])
        self.assertAlmostEqual(agg["step_sr"], 0.5)  # macro over tasks (the paper's convention)
        self.assertAlmostEqual(agg["step_sr_micro"], 0.25)  # micro over steps

    def test_ranker_is_blind_to_gold_flags_and_input_order(self):
        task = mind2web.load_mind2web(self.root, split="CT")[0]
        cands = task.gold[0].candidates
        base = [c.node_id for c in mind2web.lexical_ranker(task, 0, cands, [])]
        self.assertEqual(base, [c.node_id for c in mind2web.lexical_ranker(task, 0, list(reversed(cands)), [])])
        import copy

        flipped = copy.deepcopy(cands)
        for c in flipped:
            c.positive = not c.positive
        self.assertEqual(base, [c.node_id for c in mind2web.lexical_ranker(task, 0, flipped, [])])

    def _oracle(self, task_by_text):
        """Scripted reasoner that reads the prompt and picks the candidate carrying the gold element's aria-label."""

        def fn(messages):
            c = messages[-1]["content"]
            before, latest = c.split("Latest observation:")  # the page is the (pending) observation of every call
            if "All steps are complete" in latest:
                return "<answer>done</answer>"
            task = next(t for text, t in task_by_text.items() if text in before)
            idx = int(re.search(r"Step (\d+)/", latest).group(1)) - 1
            step = task.gold[idx]
            gold_label = next(x for x in step.candidates if x.positive).attrs["aria_label"]
            letter = next(line[0] for line in latest.splitlines() if re.match(r"[B-Z]\. ", line) and gold_label in line)
            return f'<tool>act {{"choice": "{letter}", "operation": "{step.op}", "value": {json.dumps(step.value)}}}</tool>'

        return fn

    def test_oracle_policy_scores_perfectly_and_none_of_above_scores_zero(self):
        tasks = mind2web.load_mind2web(self.root)
        by_text = {t.prompt: t for t in tasks}
        ad = mind2web.Mind2WebAdapter(top_k=10)
        results = ad.run_all(tasks, agent_for("flat_rag", MockLLM(self._oracle(by_text))))
        for r in results:
            self.assertEqual((r.extra["ele_acc"], r.extra["op_f1"], r.extra["step_sr"], r.extra["sr"]), (1.0, 1.0, 1.0, 1.0), r.task_id)
        zeros = ad.run_all(tasks, agent_for("flat_rag", mind2web.mock_llm()))
        self.assertTrue(all(r.extra["step_sr"] == 0.0 and r.extra["sr"] == 0.0 for r in zeros))

    def test_top_k_makes_gold_unreachable_and_history_is_teacher_forced(self):
        tasks = mind2web.load_mind2web(self.root, split="CW")
        t = next(t for t in tasks if t.task_id == "w1")
        env = mind2web._StepEnv(t, top_k=1, ranker=mind2web.lexical_ranker)
        self.assertEqual(sum(1 for line in env.first.splitlines() if re.match(r"[B-Z]\. ", line)), 1)
        res = env.act({"choice": "A", "operation": "CLICK"})  # wrong on purpose
        self.assertEqual(res.code, 200)
        self.assertIn("[button] gold-w1-0 -> TYPE red shoes", res.output)  # next page carries the GROUND-TRUTH history
        self.assertEqual(env.act({"choice": "Z"}).code, 400)  # invalid letter is rejected and does not advance
        self.assertEqual(len(env.preds), 1)

    def test_summary_rows_per_split(self):
        tasks = mind2web.load_mind2web(self.root)
        ad = mind2web.Mind2WebAdapter()
        results = ad.run_all(tasks, agent_for("none", mind2web.mock_llm()))
        rows = ad.summarize(tasks, results)
        self.assertEqual([r["split"] for r in rows], ["CD", "CT", "CW", "all"])
        self.assertTrue(all("step_sr" in r and "ele_acc_micro" in r and "ctu_per_task" in r for r in rows))
        self.assertEqual(rows[-1]["n_tasks"], 6)


# ---------------------------------------------------------------------------------------------- StreamBench
class TestStreamBench(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(streambench.exact_match("The Paris!", "paris"), 1.0)
        self.assertEqual(streambench.exact_match("rome", ["paris", "Rome"]), 1.0)  # any-of
        self.assertEqual(streambench.exact_match("rome", "paris"), 0.0)
        self.assertAlmostEqual(streambench.token_f1("red green blue", "red green pink"), 2 / 3)
        self.assertEqual(streambench.token_f1("", ""), 1.0)
        self.assertEqual(streambench.token_f1("x", ""), 0.0)

    def test_feedback_status_mapping(self):
        self.assertEqual(streambench.feedback_status(True), Status.SUCCESS)
        self.assertEqual(streambench.feedback_status(False), Status.FAILED)
        self.assertEqual(streambench.feedback_status(None), Status.UNKNOWN)

    def test_loader(self):
        with tempfile.TemporaryDirectory() as d:
            tasks = streambench.load_stream(write_stream_fixture(Path(d) / "s.jsonl"), limit=3)
            self.assertEqual([t.gold for t in tasks], ["paris"] * 3)
            bad = write_jsonl(Path(d) / "bad.jsonl", [{"input": "x"}])
            with self.assertRaises(MissingDataError):
                streambench.load_stream(bad)
            with self.assertRaises(MissingDataError):
                streambench.load_stream(Path(d) / "missing.jsonl")

    @staticmethod
    def learner():
        """Wrong the first time; once memory shows an incorrect or a correct trace it answers 'paris'."""

        def fn(messages):
            c = messages[-1]["content"]
            if "<memory>" not in c:
                return "<recall>capital of france</recall>"
            mem = c.split("<memory>")[1]
            return "<answer>paris</answer>" if "feedback:" in mem else "<answer>wrong</answer>"

        return MockLLM(fn)

    def tasks(self):
        with tempfile.TemporaryDirectory() as d:
            return streambench.load_stream(write_stream_fixture(Path(d) / "s.jsonl"))

    def test_continual_stream_improves_and_logs_statuses(self):
        tasks = self.tasks()
        agent = agent_for("flat_rag", self.learner())
        ad = streambench.StreamBenchAdapter(keep_memory=True)
        results = ad.run_all(tasks, agent)
        self.assertEqual([r.score for r in results], [0.0, 1.0, 1.0, 1.0])
        self.assertEqual([round(r.extra["running_score"], 4) for r in results], [0.0, 0.5, 0.6667, 0.75])
        fb = [o for o in agent.audit.obs.values() if o.tool == "stream_feedback"]
        self.assertEqual([o.status for o in sorted(fb, key=lambda o: o.step)], [Status.FAILED, Status.SUCCESS, Status.SUCCESS, Status.SUCCESS])
        self.assertEqual([r.extra["feedback"] for r in results], ["FAILED", "SUCCESS", "SUCCESS", "SUCCESS"])
        row = ad.summarize(tasks, results)[0]
        self.assertAlmostEqual(row["final_metric"], 0.75)
        self.assertGreater(row["score_second_half"], row["score_first_half"])

    def test_without_keep_memory_nothing_is_learned(self):
        tasks = self.tasks()
        agent = agent_for("flat_rag", self.learner())
        results = streambench.StreamBenchAdapter(keep_memory=False).run_all(tasks, agent)
        self.assertEqual([r.score for r in results], [0.0] * 4)
        self.assertEqual(len([o for o in agent.audit.obs.values() if o.tool == "stream_feedback"]), 0)

    def test_unknown_feedback_is_written_as_unknown(self):
        tasks = self.tasks()[:2]
        agent = agent_for("flat_rag", MockLLM(["<answer>paris</answer>"]))
        ad = streambench.StreamBenchAdapter(feedback_fn=lambda task, pred, score: None)
        ad.run_all(tasks, agent)
        self.assertEqual({o.status for o in agent.audit.obs.values()}, {Status.UNKNOWN})

    def test_prefetch_recalls_with_x_t_before_the_first_call(self):
        tasks = self.tasks()[:2]
        llm = MockLLM(["<answer>paris</answer>"])
        agent = agent_for("flat_rag", llm)
        results = streambench.StreamBenchAdapter(prefetch=True).run_all(tasks, agent)
        self.assertIn("(no relevant memory)", llm.calls[0][1]["content"])  # nothing in memory yet
        self.assertNotIn("(no relevant memory)", llm.calls[1][1]["content"])  # item 2 sees item 1's (input, prediction, feedback) trace
        self.assertIn("feedback: correct", llm.calls[1][1]["content"])
        self.assertEqual(results[1].extra["n_recalls"], 1)

    def test_run_all_resets_the_agent_so_streams_do_not_leak(self):
        tasks = self.tasks()
        agent = agent_for("flat_rag", self.learner())
        ad = streambench.StreamBenchAdapter()
        first = ad.run_all(tasks, agent)
        second = ad.run_all(tasks, agent)
        self.assertEqual([r.score for r in first], [r.score for r in second])


# ---------------------------------------------------------------------------------------------- LifelongAgentBench
class TestLifelong(unittest.TestCase):
    def tasks(self):
        with tempfile.TemporaryDirectory() as d:
            return lifelong.load_lifelong(write_lifelong_fixture(Path(d) / "l.jsonl"))

    def test_replay_memory_keeps_k_most_recent_successes_verbatim(self):
        rm = lifelong.ReplayMemory(2)
        for i in range(4):
            rm.add(lifelong.Trajectory(f"t{i}", f"instr {i}", [("a", "o")], "ans", success=(i != 2)))
        self.assertEqual([t.task_id for t in rm.recent()], ["t1", "t3"])  # t2 failed -> never replayed
        text = rm.prefix()
        self.assertIn("Instruction: instr 1", text)
        self.assertIn("Instruction: instr 3", text)
        self.assertNotIn("instr 0", text)
        self.assertNotIn("instr 2", text)
        self.assertEqual(lifelong.ReplayMemory(0).prefix(), "")
        with self.assertRaises(ValueError):
            lifelong.ReplayMemory(-1)

    def test_replay_prompt_grows_linearly_with_k(self):
        counter = harness.TokenCounter()
        sizes = []
        for k in (1, 2, 3, 4):
            rm = lifelong.ReplayMemory(k)
            for i in range(4):
                rm.add(lifelong.Trajectory(f"t{i}", "set key a to 1", [("set a 1", "ok a=1")], "done", True))
            sizes.append(counter.count(rm.prefix()))
        self.assertEqual(sizes[1] - sizes[0], sizes[2] - sizes[1])
        self.assertEqual(sizes[2] - sizes[1], sizes[3] - sizes[2])
        self.assertGreater(sizes[0], 0)

    def test_oom_guard_reports_instead_of_crashing(self):
        tasks = self.tasks()
        rm = lifelong.ReplayMemory(8)
        for i in range(8):
            rm.add(lifelong.Trajectory(f"t{i}", "long instruction " * 20, [("act", "obs " * 20)], "done", True))
        llm = toy_policy()
        cfg = HelmConfig().override(**{"reasoner.context_window": 200})
        ad = lifelong.LifelongAdapter(ToyKVEnv(), rm)
        res = ad.run_episode(tasks[0], make_agent("none", cfg, llm=llm))
        self.assertTrue(res.extra["oom"])
        self.assertEqual((res.score, res.success, res.steps), (0.0, False, 0))
        self.assertEqual(llm.calls, [])  # never reached the model
        self.assertGreater(res.extra["prompt_tokens"], 200)
        # a tighter explicit limit works the same way and the summary carries the flag
        ad2 = lifelong.LifelongAdapter(ToyKVEnv(), None, max_prompt_tokens=5)
        res2 = ad2.run_episode(tasks[0], agent_for("none", toy_policy()))
        self.assertTrue(res2.extra["oom"])
        row = ad2.summarize(tasks[:1], [res2])[0]
        self.assertEqual((row["oom"], row["n_oom"], row["first_oom_index"]), (True, 1, 0))

    def test_sequential_run_persists_memory_logs_failures_and_replays(self):
        tasks = self.tasks()
        rm = lifelong.ReplayMemory(2)
        llm = toy_policy(bad_keys=("k1",))
        ad = lifelong.LifelongAdapter(ToyKVEnv(), rm)
        agent = agent_for("flat_rag", llm)
        results = ad.run_all(tasks, agent)
        self.assertEqual([r.score for r in results], [1.0, 0.0, 1.0])
        self.assertEqual([round(r.extra["running_success"], 3) for r in results], [1.0, 0.5, 0.667])
        # failed command -> FAILED observation in the audit; verified outcomes written as SUCCESS / FAILED
        env_obs = {o.step: o for o in agent.audit.obs.values() if o.tool == "env"}
        self.assertEqual(sorted(o.status.value for o in env_obs.values()), ["FAILED", "SUCCESS", "SUCCESS"])
        outcome = [o.status for o in sorted((o for o in agent.audit.obs.values() if o.tool == "episode_outcome"), key=lambda o: o.step)]
        self.assertEqual(outcome, [Status.SUCCESS, Status.FAILED, Status.SUCCESS])
        self.assertGreaterEqual(len(agent.backend.log), 6)  # memory persisted across the whole sequence
        self.assertEqual([t.task_id for t in rm.recent()], ["l0", "l2"])  # only verified successes are replayed
        third_prompt = llm.calls[-2][1]["content"]
        self.assertIn("Earlier solved tasks", third_prompt)
        self.assertIn("Instruction: set key k0 to 0", third_prompt)

    def test_summary_per_env_and_missing_env_error(self):
        tasks = self.tasks()
        ad = lifelong.LifelongAdapter(ToyKVEnv())
        results = ad.run_all(tasks, agent_for("none", toy_policy()))
        rows = ad.summarize(tasks, results)
        self.assertEqual([r["split"] for r in rows], ["all", "toykv"])
        self.assertEqual(rows[0]["score"], 1.0)
        with self.assertRaises(MissingDataError):
            lifelong.LifelongAdapter().run_episode(tasks[0], agent_for("none", toy_policy()))


# ---------------------------------------------------------------------------------------------- BRIGHT
class TestBright(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = bright.load_bright(write_bright_fixture(Path(self.tmp.name) / "b.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_metrics_hand_computed(self):
        ndcg = bright.ndcg_at_k
        self.assertEqual(ndcg(["a", "b"], {"a"}), 1.0)
        self.assertEqual(ndcg(["a", "b"], {"a", "b"}), 1.0)
        # gold {a, b}, ranking x a y b: DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3)
        want = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg(["x", "a", "y", "b"], {"a", "b"}), want)
        self.assertAlmostEqual(ndcg(["x", "a"], {"a"}), 1 / math.log2(3))
        self.assertEqual(ndcg(["x", "y"], {"a"}), 0.0)
        self.assertEqual(ndcg(["x"] * 10 + ["a"], {"a"}, k=10), 0.0)  # beyond the cutoff
        self.assertTrue(math.isnan(ndcg(["a"], set())))
        self.assertEqual(bright.recall_at_k(["x", "a", "y"], {"a", "b"}, 3), 0.5)
        self.assertEqual(bright.recall_at_k(["a", "b"], {"a", "b"}, 1), 0.5)
        self.assertTrue(math.isnan(bright.recall_at_k(["a"], set())))

    def test_loader_forms_and_validation(self):
        self.assertEqual(len(self.data.doc_ids), 4)
        self.assertEqual(self.data.queries[0].excluded_ids, set())  # "N/A" placeholder means no exclusions
        d = Path(self.tmp.name) / "dir"
        write_jsonl(d / "examples.jsonl", [{"id": "q", "query": "pasta", "gold_ids": ["a"], "excluded_ids": ["b"]}])
        write_jsonl(d / "documents.jsonl", [{"id": "a", "content": "pasta"}, {"id": "b", "text": "pasta pasta"}])
        data = bright.load_bright(d)
        self.assertEqual(data.queries[0].excluded_ids, {"b"})
        self.assertEqual(data.doc_texts[1], "pasta pasta")
        with self.assertRaises(MissingDataError):
            bright.parse_retrieval([{"id": "q", "query": "x", "gold_ids": ["zzz"]}], [{"id": "a", "content": "x"}])
        with self.assertRaises(MissingDataError):
            bright.load_bright(Path(self.tmp.name) / "none.json")
        self.assertEqual(len(bright.load_bright(Path(self.tmp.name) / "b.json", limit=1).queries), 1)

    def test_bm25_ranks_the_matching_document_first_and_weights_rare_terms(self):
        bm = bright.BM25Retriever()
        bm.index(self.data.doc_ids, self.data.doc_texts)
        self.assertEqual(bm.rank("quantum photons entanglement", 3)[0], "d_quantum")
        self.assertEqual(bm.rank("boil pasta with sauce recipe", 1), ["d_pasta"])
        self.assertEqual(bm.rank("xyzzy plugh", 2), ["d_quantum", "d_pasta"])  # no match: stable corpus order
        bm.index(["common", "rare"], ["the the the apple", "zebra apple"])
        self.assertEqual(bm.rank("zebra apple", 1), ["rare"])  # idf(zebra) > idf(apple)
        self.assertEqual(bm.rank("apple", 5, exclude={"common"}), ["rare"])

    def test_every_retriever_hits_the_ideal_ranking_on_the_fixture(self):
        emb = HashingEmbedder(256)
        for name, ret in (("bm25", bright.build_retriever("bm25", emb)), ("dense", bright.build_retriever("dense", emb)),
                          ("shnm", bright.build_retriever("shnm", emb, depth=10))):
            rep = bright.evaluate_retrieval(self.data, ret, k=10)
            self.assertEqual(rep["ndcg@10"], 1.0, name)
            self.assertEqual(rep["recall@10"], 1.0, name)
            self.assertEqual(rep["n_queries"], 3)

    def test_shnm_store_holds_the_corpus_and_respects_exclusions(self):
        ret = bright.SHNMRetriever(HashingEmbedder(256), depth=10)
        ret.index(self.data.doc_ids, self.data.doc_texts)
        self.assertEqual(ret.store.sizes()[1], 4)  # nothing pruned, nothing skipped
        top = ret.rank("quantum photons entanglement", 4)
        self.assertEqual(top[0], "d_quantum")
        self.assertEqual(len(top), 4)
        self.assertNotIn("d_quantum", ret.rank("quantum photons entanglement", 4, exclude={"d_quantum"}))
        with self.assertRaises(KeyError):
            bright.build_retriever("flat_rag_typo", HashingEmbedder(32))

    def test_helm_variants_work_as_retrievers(self):
        ret = bright.build_retriever("helm_no_gov", HashingEmbedder(256), depth=10)
        self.assertEqual(bright.evaluate_retrieval(self.data, ret)["ndcg@10"], 1.0)

    def test_excluded_ids_are_removed_before_scoring(self):
        data = bright.parse_retrieval(
            [{"id": "q", "query": "pasta pasta pasta", "gold_ids": ["g"], "excluded_ids": ["decoy"]}],
            [{"id": "decoy", "content": "pasta pasta pasta pasta"}, {"id": "g", "content": "pasta salt"}, {"id": "o", "content": "other"}],
        )
        rep = bright.evaluate_retrieval(data, bright.BM25Retriever())
        self.assertEqual(rep["ndcg@10"], 1.0)  # without exclusion the decoy would push the gold to rank 2

    def test_adapter_run_all_and_summary(self):
        ad = bright.BrightAdapter()
        path = Path(self.tmp.name) / "b.json"
        tasks = ad.load(str(path))
        results = ad.run_all(tasks, bright.BM25Retriever())
        row = ad.summarize(tasks, results)[0]
        self.assertEqual((row["ndcg@10"], row["recall@10"], row["n"]), (1.0, 1.0, 3))


# ---------------------------------------------------------------------------------------------- QA sanity
class TestQA(unittest.TestCase):
    def test_extract_letter(self):
        x = qa_sanity.extract_letter
        for text, want in {
            "<answer>B</answer>": "B", "<answer>c</answer>": "C", "The answer is C.": "C", "Answer: D": "D", "B": "B",
            "B. because": "B", "(A) is right": "A", "I think (b) then (C)": "C", "Final answer: **B**": "B",
            "A cat is an animal": None, "the answer is a mystery": None, "": None, "no idea": None,
        }.items():
            self.assertEqual(x(text), want, text)
        self.assertIsNone(x("<answer>E</answer>", 4))  # out-of-range letter
        self.assertEqual(x("<answer>E</answer>", 5), "E")

    def test_parse_both_schemas_and_deterministic_gpqa_shuffle(self):
        t = qa_sanity.parse_question({"question": "q", "choices": ["w", "x", "y", "z"], "answer": 2, "subject": "s"}, 0)
        self.assertEqual((t.gold, t.split), ("C", "s"))
        self.assertEqual(qa_sanity.parse_question({"question": "q", "choices": list("wxyz"), "answer": "d"}, 0).gold, "D")
        row = {"Record ID": "r1", "Question": "q", "Correct Answer": "RIGHT", "Incorrect Answer 1": "a", "Incorrect Answer 2": "b", "Incorrect Answer 3": "c"}
        a, b = qa_sanity.parse_question(row, 0, seed=3), qa_sanity.parse_question(row, 0, seed=3)
        self.assertEqual(a.meta["choices"], b.meta["choices"])
        self.assertEqual(a.meta["choices"]["ABCD".index(a.gold)], "RIGHT")  # the tracked gold position is the right option
        positions = {qa_sanity.parse_question({**row, "Record ID": f"r{i}"}, 0).gold for i in range(30)}
        self.assertGreater(len(positions), 1)  # actually shuffled
        with self.assertRaises(MissingDataError):
            qa_sanity.parse_question({"foo": 1}, 0)
        with self.assertRaises(MissingDataError):
            qa_sanity.parse_question({"question": "q", "choices": ["a", "b"], "answer": 5}, 0)

    def test_prompt_builder(self):
        p = qa_sanity.build_prompt("Why?", ["a", "b"], ["first passage", "second passage"])
        self.assertIn("[1] first passage", p)
        self.assertIn("A. a\nB. b", p)
        self.assertIn("<answer>X</answer>", p)
        self.assertNotIn("Context:", qa_sanity.build_prompt("Why?", ["a", "b"]))

    def test_adapter_closed_book_and_with_retrieval(self):
        with tempfile.TemporaryDirectory() as d:
            qpath, cpath = write_qa_fixtures(Path(d))
            tasks = qa_sanity.load_mcq(qpath)
            ids, texts = qa_sanity.load_corpus(cpath)
        closed_llm = MockLLM(["<answer>B</answer>"])
        closed = qa_sanity.QAAdapter().run_all(tasks, agent_for("none", closed_llm))
        self.assertEqual([r.score for r in closed], [1.0, 1.0])  # the scripted reply "B" is right for both fixtures
        self.assertNotIn("Context:", closed_llm.calls[0][1]["content"])
        self.assertEqual(closed[0].extra["n_passages"], 0)

        bm = bright.BM25Retriever()
        llm = MockLLM(["<answer>A</answer>"])
        ad = qa_sanity.QAAdapter(bm, (ids, texts), k=1)
        results = ad.run_all(tasks, agent_for("none", llm))
        self.assertEqual([r.score for r in results], [0.0, 0.0])
        self.assertIn("carbon dioxide", llm.calls[0][1]["content"])  # the photosynthesis passage was retrieved
        self.assertEqual(results[0].extra["retrieved_ids"], ["c1"])
        self.assertEqual(results[1].extra["retrieved_ids"], ["c2"])
        self.assertGreater(results[0].ctu, 0)
        unparsed = qa_sanity.QAAdapter().run_all(tasks[:1], agent_for("none", MockLLM(["no idea"])))
        self.assertEqual((unparsed[0].score, unparsed[0].extra["letter"]), (0.0, None))
        with self.assertRaises(ValueError):
            qa_sanity.QAAdapter(bm)  # a retriever needs a corpus

    def test_qa_decoding_comes_from_config(self):
        client = _CapturingClient("<answer>A</answer>")
        with tempfile.TemporaryDirectory() as d:
            tasks = qa_sanity.load_mcq(write_qa_fixtures(Path(d))[0])
        qa_sanity.QAAdapter().run_all(tasks[:1], agent_for("none", client))
        self.assertEqual(client.gens[0]["temperature"], HelmConfig().reasoner.temperature)


# ---------------------------------------------------------------------------------------------- CLI
class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.out = self.d / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_benchmark.main([*map(str, argv), "--out-dir", str(self.out)])
        return code, out.getvalue(), err.getvalue()

    def rows(self, name):
        with open(self.out / name, newline="") as f:
            return list(csv.DictReader(f))

    def assert_labelled(self, name, source="mock"):
        rows = self.rows(name)
        self.assertTrue(rows)
        self.assertEqual({r["source"] for r in rows}, {source}, name)

    def test_gaia_mock_run_writes_mock_rows_and_prints_config(self):
        data = write_gaia_fixture(self.d)
        code, out, err = self.run_cli("--benchmark", "gaia", "--method", "helm", "--data", data, "--mock", "--seeds", 0, 42, "--limit", 2)
        self.assertEqual(code, 0, err)
        self.assertIn(f'"max_recall_tokens": {HelmConfig().memory.max_recall_tokens}', out)  # resolved config is printed
        self.assert_labelled("bench_gaia.csv")
        self.assert_labelled("bench_gaia_episodes.csv")
        summary = self.rows("bench_gaia.csv")
        self.assertIn("mean", {r["seed"] for r in summary})  # across-seed aggregation row
        self.assertTrue(all(r["method"] == "helm" for r in summary))

    def test_mind2web_streambench_qa_bright_mock_runs(self):
        m2w = write_mind2web_fixture(self.d / "m2w")
        code, _, err = self.run_cli("--benchmark", "mind2web", "--method", "flat_rag", "--data", m2w, "--mock", "--seeds", 0)
        self.assertEqual(code, 0, err)
        self.assert_labelled("bench_mind2web.csv")
        self.assertEqual({r["split"] for r in self.rows("bench_mind2web.csv")}, {"CT", "CW", "CD", "all"})
        stream = write_stream_fixture(self.d / "s.jsonl")
        code, _, err = self.run_cli("--benchmark", "streambench", "--method", "mem0_style", "--data", stream, "--mock", "--seeds", 0)
        self.assertEqual(code, 0, err)
        self.assert_labelled("bench_streambench_episodes.csv")
        qpath, cpath = write_qa_fixtures(self.d)
        code, _, err = self.run_cli("--benchmark", "qa", "--method", "bm25", "--data", qpath, "--corpus", cpath, "--mock", "--seeds", 0)
        self.assertEqual(code, 0, err)
        self.assert_labelled("bench_qa.csv")
        b = write_bright_fixture(self.d / "b.json")
        code, _, err = self.run_cli("--benchmark", "bright", "--method", "shnm", "--data", b, "--mock", "--seeds", 0)
        self.assertEqual(code, 0, err)
        self.assert_labelled("bench_bright.csv")

    def test_lifelong_with_env_factory_and_replay(self):
        data = write_lifelong_fixture(self.d / "l.jsonl")
        code, _, err = self.run_cli(
            "--benchmark", "lifelong", "--method", "none", "--data", data, "--mock", "--seeds", 0,
            "--env-factory", "tests.test_benchmarks:make_toy_env", "--replay-k", 2,
        )
        self.assertEqual(code, 0, err)
        self.assert_labelled("bench_lifelong.csv")

    def test_bright_label_follows_the_components_not_the_data(self):
        b = write_bright_fixture(self.d / "b.json")
        self.assertEqual(self.run_cli("--benchmark", "bright", "--method", "bm25", "--data", b, "--seeds", 0)[0], 0)
        self.assertEqual({r["source"] for r in self.rows("bench_bright.csv")}, {"measured"})  # pure computation, no stand-ins
        os.remove(self.out / "bench_bright.csv")
        self.assertEqual(self.run_cli("--benchmark", "bright", "--method", "dense", "--data", b, "--seeds", 0)[0], 0)
        self.assertEqual({r["source"] for r in self.rows("bench_bright.csv")}, {"mock"})  # hashing embedder is a stand-in

    def test_clear_errors_instead_of_stack_traces(self):
        data = write_gaia_fixture(self.d)
        cases = {
            "no data": (["--benchmark", "gaia", "--method", "helm", "--mock"], "--data is required"),
            "bad path": (["--benchmark", "gaia", "--method", "helm", "--mock", "--data", self.d / "nope"], "does not exist"),
            "no llm": (["--benchmark", "gaia", "--method", "helm", "--data", data], "--mock"),
            "both llm": (["--benchmark", "gaia", "--method", "helm", "--data", data, "--mock", "--llm-base-url", "http://x"], "exactly one"),
            "bad method": (["--benchmark", "gaia", "--method", "nope", "--data", data, "--mock"], "unknown method"),
            "lifelong env": (["--benchmark", "lifelong", "--method", "none", "--data", data, "--mock"], "--env-factory"),
            "bad factory": (["--benchmark", "lifelong", "--method", "none", "--data", data, "--mock", "--env-factory", "no.such:thing"], "--env-factory"),
            "qa needs corpus": (["--benchmark", "qa", "--method", "bm25", "--data", data, "--mock"], "--corpus"),
            "bright none": (["--benchmark", "bright", "--method", "none", "--data", data], "retrieval"),
            "bad config": (["--benchmark", "gaia", "--method", "helm", "--data", data, "--mock", "--config", self.d / "x.yaml"], "config file"),
        }
        for name, (argv, needle) in cases.items():
            code, _, err = self.run_cli(*argv)
            self.assertEqual(code, 2, name)
            self.assertTrue(err.startswith("error:"), (name, err))
            self.assertIn(needle, err, name)
        self.assertFalse(self.out.exists() and any(self.out.iterdir()))  # failed runs wrote nothing

    def test_webarena_cannot_run_without_the_environments(self):
        cfgs = self.d / "wa"
        cfgs.mkdir()
        (cfgs / "1.json").write_text(json.dumps({**TestWebArena.RAW, "task_id": 1}))
        clean = {k: v for k, v in os.environ.items() if k not in webarena.WEBARENA_ENV_VARS}
        with mock.patch.dict(os.environ, clean, clear=True):
            code, _, err = self.run_cli("--benchmark", "webarena", "--method", "helm", "--data", cfgs, "--mock", "--seeds", 0)
        self.assertEqual(code, 2)
        self.assertIn("WebArena sites are not configured", err)

    def test_real_client_path_reports_missing_openai_package_cleanly(self):
        data = write_gaia_fixture(self.d)
        with mock.patch.dict(sys.modules, {"openai": None}):  # no network is ever touched on this path
            code, _, err = self.run_cli("--benchmark", "gaia", "--method", "flat_rag", "--data", data, "--llm-base-url", "http://localhost:1/v1", "--seeds", 0)
        self.assertEqual(code, 2)
        self.assertIn("openai", err)

    def test_episode_csv_does_not_repeat_the_limits_block(self):
        data = write_gaia_fixture(self.d)
        self.assertEqual(self.run_cli("--benchmark", "gaia", "--method", "flat_rag", "--data", data, "--mock", "--seeds", 0)[0], 0)
        for r in self.rows("bench_gaia_episodes.csv"):
            self.assertNotIn("limits", r["extra_json"])

    def test_mock_can_never_produce_measured_rows(self):
        data = write_gaia_fixture(self.d)
        self.assertEqual(self.run_cli("--benchmark", "gaia", "--method", "flat_rag", "--data", data, "--mock", "--seeds", 0)[0], 0)
        for name in ("bench_gaia.csv", "bench_gaia_episodes.csv"):
            self.assertNotIn("measured", {r["source"] for r in self.rows(name)})


if __name__ == "__main__":
    unittest.main()
