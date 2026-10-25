"""Mind2Web adapter: JSON loader, MINDACT-style multiple-choice step prompt, step-level metrics, per-split reports.

SCHEMA FOLLOWS THE DATASET CARD (``osunlp/Mind2Web``) and has only been exercised on fixtures; check it against the real
files before use:

    task:   annotation_id, website, domain, subdomain, confirmed_task, action_reprs[str], actions[]
    action: action_uid, operation{op, original_op, value}, pos_candidates[], neg_candidates[], cleaned_html, raw_html
    cand.:  tag, backend_node_id, attributes (a JSON *string*), is_original_target, is_top_level_target
    splits: test_task (cross-task, CT), test_website (cross-website, CW), test_domain (cross-domain, CD)

What is taken from the paper itself (Deng et al., 2023, Sec. 4.2): Element Accuracy = the selected element is among the
acceptable (positive) elements; Operation F1 = token-level F1 of the predicted operation including its value; a step
succeeds iff element AND operation are correct; a task succeeds iff all steps succeed; step-wise metrics are macro
averages across tasks; evaluation uses the ground-truth action history (teacher forcing); MINDACT selects among
candidates in groups of up to five plus a "None of the above" option.

Deviations / assumptions:
* The benchmark's DeBERTa candidate-ranker outputs are an external artifact we do not have. The default ranker is a
  transparent lexical-overlap ranker over the pooled (positive + negative) candidates, with ties broken by a hash of the
  node id so gold-first ordering in the file cannot leak; candidates that do not reach the top-k are unreachable.
  Plug the official ranking in via ``ranker=``.
* Operation F1 uses token *sets* (MINDACT behaviour); ``op_f1(..., multiset=True)`` gives SQuAD-style counting.
  The two differ only when a token repeats.
* Selection is a single multiple-choice round over the top-k candidates (default 10, plus "None of the above"), not MINDACT's
  iterative rounds of five; recall of the gold element is therefore capped by the ranker and ``top_k``.
* ``limit`` applies per split so a quick run still covers CT, CW and CD.
"""

from __future__ import annotations

import json
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..llm.client import MockLLM
from .base import BaseAdapter, EpisodeResult, MissingDataError, Task, summarize
from .harness import Agent, ToolRegistry, ToolResult

SPLIT_NAMES = {"test_task": "CT", "test_website": "CW", "test_domain": "CD"}
_ALIASES = {
    "ct": "test_task", "cross_task": "test_task", "cross-task": "test_task", "test_task": "test_task",
    "cw": "test_website", "cross_website": "test_website", "cross-website": "test_website", "test_website": "test_website",
    "cd": "test_domain", "cross_domain": "test_domain", "cross-domain": "test_domain", "test_domain": "test_domain",
}
_ATTR_KEYS = ("aria_label", "title", "placeholder", "name", "id", "type", "value", "alt", "href")
_WORD = re.compile(r"[a-z0-9]+")


def canonical_split(name: Optional[str]) -> str:
    if not name:
        return ""
    return _ALIASES.get(name.strip().lower(), name)


# --------------------------------------------------------------------------------------------- data model
@dataclass
class Candidate:
    node_id: str
    tag: str
    attrs: Dict[str, Any]
    inner_text: str
    positive: bool = False

    def render(self, max_chars: int = 160) -> str:
        parts = [self.tag or "node"]
        for k in _ATTR_KEYS:
            v = self.attrs.get(k)
            if v not in (None, ""):
                parts.append(f'{k}="{str(v)[:40]}"')
        out = "<" + " ".join(parts) + ">"
        if self.inner_text:
            out += " " + self.inner_text
        return out[:max_chars]

    def label(self) -> str:
        for k in ("aria_label", "title", "placeholder", "name", "id"):
            if self.attrs.get(k):
                return str(self.attrs[k])[:40]
        return self.inner_text[:40]


@dataclass
class Step:
    op: str
    value: str
    candidates: List[Candidate]
    action_repr: str = ""

    @property
    def pos_ids(self) -> Set[str]:
        return {c.node_id for c in self.candidates if c.positive}


def _element_text(cleaned_html: str, node_id: str, max_chars: int = 80) -> str:
    """Best-effort inner text of the element carrying ``backend_node_id=<id>`` in the cleaned HTML."""
    if not cleaned_html:
        return ""
    m = re.search(r'<\w+\b[^>]*\bbackend_node_id="%s"[^>]*>([^<]*)' % re.escape(node_id), cleaned_html)
    return " ".join(m.group(1).split())[:max_chars] if m else ""


def _attrs(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _candidate(c: Dict[str, Any], html: str, positive: bool) -> Candidate:
    nid = str(c.get("backend_node_id", c.get("node_id", "")))
    return Candidate(nid, str(c.get("tag", "")), _attrs(c.get("attributes")), _element_text(html, nid), positive)


def parse_task(row: Dict[str, Any], split: str = "") -> Task:
    """One Mind2Web record -> ``Task`` (gold = list of ``Step``)."""
    for k in ("annotation_id", "confirmed_task", "actions"):
        if k not in row:
            raise MissingDataError(f"Mind2Web record lacks '{k}' (keys: {sorted(row)}); schema follows the dataset card, check it")
    reprs = row.get("action_reprs") or []
    steps: List[Step] = []
    for i, a in enumerate(row["actions"]):
        op = a.get("operation") or {}
        html = a.get("cleaned_html", "")
        cands = [_candidate(c, html, True) for c in a.get("pos_candidates", [])]
        cands += [_candidate(c, html, False) for c in a.get("neg_candidates", [])]
        pos = next((c for c in cands if c.positive), None)
        value = str(op.get("value", "") or "")
        name = str(op.get("op", "")).upper()
        repr_ = reprs[i] if i < len(reprs) else ""
        if not repr_ and pos is not None:
            repr_ = f"[{pos.tag}] {pos.label()} -> {name}" + (f" {value}" if value else "")
        steps.append(Step(name, value, cands, repr_))
    return Task(
        task_id=str(row["annotation_id"]),
        prompt=str(row["confirmed_task"]),
        gold=steps,
        split=split,
        meta={"website": row.get("website", ""), "domain": row.get("domain", ""), "n_steps": len(steps)},
    )


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise MissingDataError(f"{path}: expected a JSON list of Mind2Web tasks")
    return data


def _split_of(path: Path) -> str:
    for part in (*path.parts[-3:],):
        for name in SPLIT_NAMES:
            if name in part:
                return name
    return ""


def load_mind2web(path: str | Path, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
    """File (json list / jsonl) or directory; in a directory every ``test_task*``/``test_website*``/``test_domain*``
    file (or sub-directory of files) is one split. ``limit`` is applied per split."""
    p = Path(path)
    if not p.exists():
        raise MissingDataError(f"Mind2Web data path does not exist: {p}")
    want = canonical_split(split)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl") and f.is_file())
    by_split: Dict[str, List[Task]] = {}
    for f in files:
        sp = _split_of(f) or want
        if want and sp and sp != want:
            continue
        for row in _read_json_rows(f):
            if not row.get("actions"):
                continue  # nothing to evaluate
            by_split.setdefault(sp, []).append(parse_task(row, sp))
    tasks: List[Task] = []
    for sp in sorted(by_split):
        tasks.extend(by_split[sp][:limit] if limit is not None else by_split[sp])
    if not tasks:
        raise MissingDataError(f"no Mind2Web tasks found under {p} (split={split!r}); expected test_task/test_website/test_domain json files")
    return tasks


# --------------------------------------------------------------------------------------------- metrics
def op_f1(pred: str, gold: str, multiset: bool = False, lowercase: bool = False) -> float:
    """Token-level F1 between two ``"OP value"`` strings (see module docstring for the set-vs-multiset note)."""
    if lowercase:
        pred, gold = pred.lower(), gold.lower()
    p, g = pred.split(), gold.split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    if multiset:
        tp = sum((Counter(p) & Counter(g)).values())
        np_, ng = len(p), len(g)
    else:
        sp, sg = set(p), set(g)
        tp, np_, ng = len(sp & sg), len(sp), len(sg)
    if tp == 0:
        return 0.0
    prec, rec = tp / np_, tp / ng
    return 2 * prec * rec / (prec + rec)


@dataclass
class Prediction:
    node_id: Optional[str]  # None = "None of the above" / no selection
    op: str
    value: str


def score_step(step: Step, pred: Optional[Prediction]) -> Dict[str, float]:
    if pred is None:
        return {"ele": 0.0, "f1": 0.0, "ok": 0.0}
    ele = 1.0 if (pred.node_id is not None and pred.node_id in step.pos_ids) else 0.0
    f1 = op_f1(f"{pred.op} {pred.value}".strip(), f"{step.op} {step.value}".strip())
    return {"ele": ele, "f1": f1, "ok": 1.0 if (ele == 1.0 and f1 == 1.0) else 0.0}


def score_task(steps: Sequence[Step], preds: Sequence[Optional[Prediction]]) -> Dict[str, Any]:
    if not steps:
        raise ValueError("a Mind2Web task needs at least one step")
    per = [score_step(s, preds[i] if i < len(preds) else None) for i, s in enumerate(steps)]
    n = len(per)
    return {
        "ele_acc": sum(x["ele"] for x in per) / n,
        "op_f1": sum(x["f1"] for x in per) / n,
        "step_sr": sum(x["ok"] for x in per) / n,
        "sr": 1.0 if all(x["ok"] == 1.0 for x in per) else 0.0,
        "n_steps": n,
        "n_ele": int(sum(x["ele"] for x in per)),
        "f1_sum": sum(x["f1"] for x in per),
        "n_step_ok": int(sum(x["ok"] for x in per)),
    }


def aggregate(per_task: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Macro (over tasks, the paper's step-wise convention) and micro (over steps) averages."""
    if not per_task:
        return {}
    k = len(per_task)
    steps = sum(t["n_steps"] for t in per_task)
    return {
        "n_tasks": k,
        "n_steps": steps,
        "ele_acc": sum(t["ele_acc"] for t in per_task) / k,
        "op_f1": sum(t["op_f1"] for t in per_task) / k,
        "step_sr": sum(t["step_sr"] for t in per_task) / k,
        "sr": sum(t["sr"] for t in per_task) / k,
        "ele_acc_micro": sum(t["n_ele"] for t in per_task) / steps,
        "op_f1_micro": sum(t["f1_sum"] for t in per_task) / steps,
        "step_sr_micro": sum(t["n_step_ok"] for t in per_task) / steps,
    }


# --------------------------------------------------------------------------------------------- candidates & prompts
def _tokens(text: str) -> Set[str]:
    return set(_WORD.findall(text.lower()))


def lexical_ranker(task: Task, step_idx: int, cands: Sequence[Candidate], history: Sequence[str]) -> List[Candidate]:
    """Rank the pooled candidates by token overlap with task + previous actions. Never reads ``Candidate.positive``."""
    q = _tokens(task.prompt + " " + " ".join(history))

    def key(c: Candidate) -> Tuple[float, int]:
        toks = _tokens(c.render())
        score = len(q & toks) / ((len(toks) + 1) ** 0.5)
        return (-score, zlib.crc32(c.node_id.encode()))

    return sorted(cands, key=key)


Ranker = Callable[[Task, int, Sequence[Candidate], Sequence[str]], List[Candidate]]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_step(task: Task, idx: int, history: Sequence[str], shown: Sequence[Candidate]) -> str:
    lines = [f"Step {idx + 1}/{len(task.gold)} on {task.meta.get('website', 'the website')}."]
    lines.append("Previous actions:" + ("" if history else " (none)"))
    lines += [f"  {h}" for h in history]
    lines.append("What should be the next action? Choose from (if the correct element is not listed choose A):")
    lines.append("A. None of the above")
    lines += [f"{LETTERS[i + 1]}. {c.render()}" for i, c in enumerate(shown)]
    return "\n".join(lines)


def parse_choice(args: Dict[str, Any], n_shown: int) -> Tuple[Optional[int], Optional[str]]:
    """Letter -> candidate index (-1 = None of the above). Returns ``(index, error)``."""
    raw = str(args.get("choice", "")).strip().upper()
    if not raw or raw[0] not in LETTERS:
        return None, "missing 'choice' letter"
    k = LETTERS.index(raw[0])
    if k > n_shown:
        return None, f"choice {raw[0]} is not an option (A..{LETTERS[n_shown]})"
    return k - 1, None


class _StepEnv:
    """Teacher-forced replay of one task: the next page is always the next gold page, whatever the agent chose."""

    def __init__(self, task: Task, top_k: int, ranker: Ranker):
        self.task, self.top_k, self.ranker = task, top_k, ranker
        self.steps: List[Step] = task.gold
        self.idx = 0
        self.history: List[str] = []
        self.preds: List[Optional[Prediction]] = []
        self._shown: List[Candidate] = []
        self.first = self._present()

    def _present(self) -> str:
        step = self.steps[self.idx]
        self._shown = self.ranker(self.task, self.idx, step.candidates, self.history)[: self.top_k]
        return render_step(self.task, self.idx, self.history, self._shown)

    def act(self, args: Dict[str, Any]) -> ToolResult:
        if self.idx >= len(self.steps):
            return ToolResult(200, "All steps are complete. Reply with <answer>done</answer>.")
        k, err = parse_choice(args, len(self._shown))
        if err:
            return ToolResult(400, f"error: {err}")
        op = (str(args.get("operation", args.get("action", "CLICK"))).strip().upper().split() or ["CLICK"])[0]
        value = str(args.get("value", "") or "")
        chosen = self._shown[k] if k >= 0 else None
        self.preds.append(Prediction(chosen.node_id if chosen else None, op, value))
        gold = self.steps[self.idx]
        self.history.append(gold.action_repr or f"{gold.op} {gold.value}".strip())  # ground-truth history
        self.idx += 1
        if self.idx >= len(self.steps):
            return ToolResult(200, "All steps are complete. Reply with <answer>done</answer>.")
        return ToolResult(200, self._present())


# --------------------------------------------------------------------------------------------- adapter
class Mind2WebAdapter(BaseAdapter):
    name = "mind2web"

    def __init__(self, top_k: int = 10, ranker: Ranker = lexical_ranker, slack: int = 2):
        self.top_k = top_k
        self.ranker = ranker
        self.slack = slack  # extra agent calls per step for recalls / format errors

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("Mind2Web needs --data PATH to the downloaded json files (see mind2web.py for the expected schema)")
        return load_mind2web(path, split, limit)

    def score(self, task: Task, prediction: Any) -> float:
        """``prediction`` = list of ``Prediction``; returns the task's Step SR."""
        return score_task(task.gold, prediction)["step_sr"]

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        env = _StepEnv(task, self.top_k, self.ranker)
        tools = ToolRegistry().add(
            "act", env.act, 'submit the next action, args {"choice": "B", "operation": "CLICK|TYPE|SELECT", "value": "<text, if any>"}'
        )
        run = agent.run(
            task.prompt, tools, task_version=task.version, max_steps=len(env.steps) * (1 + self.slack) + 1,
            initial_observation=env.first,
        )
        m = score_task(env.steps, env.preds)
        return run.to_result(m["step_sr"], m["sr"] == 1.0, task.task_id, task.split, **m)

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        groups: Dict[str, List[int]] = {}
        for i, t in enumerate(tasks):
            groups.setdefault(SPLIT_NAMES.get(t.split, t.split or "all"), []).append(i)
        groups["all"] = list(range(len(tasks)))
        for name in sorted(groups):
            idx = groups[name]
            sub = [results[i] for i in idx]
            per_task = [{k: r.extra[k] for k in ("ele_acc", "op_f1", "step_sr", "sr", "n_steps", "n_ele", "f1_sum", "n_step_ok")} for r in sub]
            rows.append({"split": name, **summarize(sub, seed=seed), **aggregate(per_task)})
        return rows


def mock_llm() -> MockLLM:
    """Offline stand-in: always picks 'None of the above' with CLICK until the steps run out. Scores ~0 by design."""

    def fn(messages: List[Dict[str, str]]) -> str:
        if "All steps are complete" in messages[-1]["content"]:
            return "<answer>done</answer>"
        return '<tool>act {"choice": "A", "operation": "CLICK", "value": ""}</tool>'

    return MockLLM(fn)
