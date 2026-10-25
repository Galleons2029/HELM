"""StreamBench adapter: the online input-feedback loop.

Protocol (Wu et al., 2024, Algorithm 1): for t = 1..T the agent receives x_t, predicts y_t, the environment returns a
binary correctness feedback fb_t in {0, 1}, and the agent may update. The final metric is sum_t h(y_t, gold_t) / T.
Here the triple (input, prediction, feedback) is written to memory with ``Status.SUCCESS`` if the feedback is positive,
``Status.FAILED`` if negative and ``Status.UNKNOWN`` otherwise (feedback unavailable), through ``Agent.ingest`` so
the audit log sees exactly what the backend sees. ``keep_memory=True`` (default) is the continual setting: the store
persists across the whole stream; ``False`` gives the memoryless zero-shot reference.

Metrics. ``metric_fn(prediction, gold) -> [0, 1]`` is pluggable. Provided: ``exact_match`` (SQuAD-style normalisation)
and ``token_f1``. The seven StreamBench task metrics are NOT implemented: Spider / CoSQL / BIRD execution accuracy and
DS-1000 pass@1 need databases / code sandboxes, ToolBench needs its tool-call checker, and DDXPlus accuracy / HotpotQA
EM are only approximated by ``exact_match``. Supply your own ``metric_fn`` for anything else.

Data. Local jsonl, one object per line: ``input`` (also ``question`` / ``x``) and ``output`` (also ``answer`` / ``gold`` /
``label``; a list means any-of). This format is a convention of this adapter, not the official release's schema.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..types import Status
from .base import BaseAdapter, EpisodeResult, MissingDataError, Task, summarize
from .harness import Agent, recall_then_answer_llm

MetricFn = Callable[[str, Any], float]
FeedbackFn = Callable[[Task, str, float], Optional[bool]]

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def _as_list(gold: Any) -> List[str]:
    return [str(g) for g in gold] if isinstance(gold, (list, tuple)) else [str(gold)]


def exact_match(pred: str, gold: Any) -> float:
    p = normalize_answer(pred or "")
    return 1.0 if any(p == normalize_answer(g) for g in _as_list(gold)) else 0.0


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred or "").split(), normalize_answer(gold).split()
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, gold: Any) -> float:
    return max(_f1(pred, g) for g in _as_list(gold))


def load_stream(path: str | Path, limit: Optional[int] = None) -> List[Task]:
    p = Path(path)
    if not p.is_file():
        raise MissingDataError(f"StreamBench stream file not found: {p} (expected jsonl with 'input' and 'output' per line)")
    tasks: List[Task] = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        x = next((row[k] for k in ("input", "question", "x") if k in row), None)
        y = next((row[k] for k in ("output", "answer", "gold", "label") if k in row), None)
        if x is None or y is None:
            raise MissingDataError(f"{p}:{n + 1}: need an input field (input/question/x) and a gold field (output/answer/gold/label)")
        tasks.append(Task(task_id=str(row.get("id", n)), prompt=str(x), gold=y, meta={"t": len(tasks)}))
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise MissingDataError(f"{p} contains no stream items")
    return tasks


def feedback_status(positive: Optional[bool]) -> Status:
    return Status.UNKNOWN if positive is None else (Status.SUCCESS if positive else Status.FAILED)


def feedback_text(task: Task, prediction: Optional[str], positive: Optional[bool]) -> str:
    verdict = "none" if positive is None else ("correct" if positive else "incorrect")
    return f"input: {task.prompt}\nprediction: {prediction}\nfeedback: {verdict}"


class StreamBenchAdapter(BaseAdapter):
    name = "streambench"

    def __init__(
        self,
        metric_fn: MetricFn = exact_match,
        feedback_fn: Optional[FeedbackFn] = None,
        keep_memory: bool = True,
        threshold: float = 1.0,
        prefetch: bool = False,
        max_steps: int = 6,
    ):
        self.metric_fn = metric_fn
        self.feedback_fn = feedback_fn or (lambda task, pred, score: score >= threshold)
        self.keep_memory = keep_memory
        self.threshold = threshold
        self.prefetch = prefetch  # recall with x_t before the first call (otherwise the model must emit <recall>)
        self.max_steps = max_steps

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("StreamBench needs --data PATH to a jsonl stream (fields: input, output)")
        return load_stream(path, limit)

    def score(self, task: Task, prediction: Any) -> float:
        return float(self.metric_fn("" if prediction is None else str(prediction), task.gold))

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        run = agent.run(
            task.prompt,
            keep_memory=self.keep_memory,
            task_version=task.version,
            max_steps=self.max_steps,
            prefetch_query=task.prompt if self.prefetch else None,
        )
        pred = run.answer
        s = self.score(task, pred)
        positive = self.feedback_fn(task, pred or "", s)
        if self.keep_memory:  # the stream's update step
            agent.ingest("stream_feedback", feedback_text(task, pred, positive), feedback_status(positive))
        return run.to_result(
            s, s >= self.threshold, task.task_id, task.split,
            prediction=pred, feedback=feedback_status(positive).value,
        )

    def run_all(self, tasks: Sequence[Task], agent: Agent) -> List[EpisodeResult]:
        agent.reset()
        out: List[EpisodeResult] = []
        total = 0.0
        for t, task in enumerate(tasks):
            res = self.run_episode(task, agent)
            total += res.score
            res.extra["t"] = t
            res.extra["running_score"] = total / (t + 1)
            out.append(res)
        return out

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        row = summarize(results, seed=seed)
        scores = [r.score for r in results]
        half = len(scores) // 2
        row["final_metric"] = row["score"]  # sum_t h_t / T
        if half:
            row["score_first_half"] = sum(scores[:half]) / half
            row["score_second_half"] = sum(scores[half:]) / (len(scores) - half)
        return [row]


def mock_llm():
    return recall_then_answer_llm("unknown")
