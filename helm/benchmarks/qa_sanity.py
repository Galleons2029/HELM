"""MMLU / GPQA multiple-choice QA with retrieved context (the appendix's auxiliary sanity check).

Pipeline per question: build a retrieval query (question, optionally with the options) -> top-k passages from a corpus via
any ``Retriever`` from ``bright.py`` (BM25, dense over an ``Embedder``, or an SHNM store) -> prompt -> one LLM call with
decoding locked from ``cfg.reasoner`` -> letter extraction -> accuracy. ``retriever=None`` gives the closed-book reference.
Not implemented: ReasonIR-style retrieval. The 4-way letter format and the extraction rules are this adapter's own.

Data (local jsonl, formats follow the dataset cards, check against real downloads):
* MMLU-like: ``question``, ``choices`` (list), ``answer`` (index or letter), optional ``subject``;
* GPQA-like: ``Question``, ``Correct Answer``, ``Incorrect Answer 1..3``. The options are shuffled deterministically per
  question (seeded) and the position of the correct one is tracked.
The corpus file is jsonl/json of ``{"id", "text" | "content"}``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.client import MockLLM
from .base import BaseAdapter, EpisodeResult, MissingDataError, Task
from .bright import Retriever
from .harness import Agent

LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------------------------------- data
def _gold_letter(ans: Any, n: int) -> str:
    if isinstance(ans, int) or (isinstance(ans, str) and ans.strip().isdigit()):
        i = int(ans)
        if not 0 <= i < n:
            raise MissingDataError(f"answer index {i} out of range for {n} choices")
        return LETTERS[i]
    letter = str(ans).strip().upper()[:1]
    if letter not in LETTERS[:n]:
        raise MissingDataError(f"cannot interpret answer {ans!r} for {n} choices")
    return letter


def parse_question(row: Dict[str, Any], idx: int, seed: int = 0) -> Task:
    if "choices" in row:
        question, choices, gold = str(row["question"]), [str(c) for c in row["choices"]], _gold_letter(row["answer"], len(row["choices"]))
    elif "Correct Answer" in row:
        question = str(row["Question"])
        opts = [str(row["Correct Answer"])] + [str(row[f"Incorrect Answer {i}"]) for i in (1, 2, 3) if f"Incorrect Answer {i}" in row]
        order = list(range(len(opts)))
        random.Random(f"{seed}:{row.get('Record ID', idx)}").shuffle(order)
        choices = [opts[j] for j in order]
        gold = LETTERS[order.index(0)]
    else:
        raise MissingDataError(f"row {idx}: unrecognised multiple-choice schema (keys: {sorted(row)})")
    return Task(
        task_id=str(row.get("id", row.get("Record ID", idx))),
        prompt=question,
        gold=gold,
        split=str(row.get("subject", "")),
        meta={"choices": choices},
    )


def load_mcq(path: str | Path, limit: Optional[int] = None, seed: int = 0) -> List[Task]:
    p = Path(path)
    if not p.is_file():
        raise MissingDataError(f"QA file not found: {p} (expected jsonl of multiple-choice questions)")
    text = p.read_text(encoding="utf-8")
    rows = json.loads(text) if p.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
    tasks = [parse_question(r, i, seed) for i, r in enumerate(rows[:limit] if limit is not None else rows)]
    if not tasks:
        raise MissingDataError(f"{p} contains no questions")
    return tasks


def load_corpus(path: str | Path) -> Tuple[List[str], List[str]]:
    p = Path(path)
    if not p.is_file():
        raise MissingDataError(f"corpus file not found: {p} (jsonl/json of {{id, text}})")
    text = p.read_text(encoding="utf-8")
    rows = json.loads(text) if p.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
    ids, texts = [], []
    for i, r in enumerate(rows):
        t = r.get("text", r.get("content"))
        if t is None:
            raise MissingDataError(f"corpus row {i} needs 'text' or 'content'")
        ids.append(str(r.get("id", i)))
        texts.append(str(t))
    return ids, texts


# --------------------------------------------------------------------------------------------- prompt / answer
def build_prompt(question: str, choices: Sequence[str], passages: Sequence[str] = ()) -> str:
    parts: List[str] = []
    if passages:
        parts.append("Context:\n" + "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)))
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    parts.append(f"Question: {question}\n{opts}")
    parts.append("Answer with the letter of the correct option, written as <answer>X</answer>.")
    return "\n\n".join(parts)


_PATTERNS = (
    re.compile(r"<answer>\s*\(?([A-H])\b", re.IGNORECASE),
    re.compile(r"\b(?i:final\s+answer|correct\s+(?:answer|option)|answer)\s*(?:is|:)?\s*\(?\*{0,2}([A-H])\b"),
    re.compile(r"^\s*\(?([A-H])(?:[\).:]|\n|$)"),
)
_PAREN = re.compile(r"\(([A-H])\)")


def extract_letter(reply: str, n_choices: int = 4) -> Optional[str]:
    """Pick the chosen letter from a free-form reply; ``None`` if nothing unambiguous is found."""
    if not reply:
        return None
    valid = LETTERS[:n_choices]
    for pat in _PATTERNS:
        m = pat.search(reply)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    found = [x for x in _PAREN.findall(reply) if x in valid]
    return found[-1] if found else None


# --------------------------------------------------------------------------------------------- adapter
class QAAdapter(BaseAdapter):
    name = "qa"

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        corpus: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
        k: int = 5,
        query_with_choices: bool = True,
        max_passage_chars: int = 1200,
    ):
        if retriever is not None and corpus is None:
            raise ValueError("a retriever needs a corpus (ids, texts)")
        self.retriever, self.corpus, self.k = retriever, corpus, k
        self.query_with_choices = query_with_choices
        self.max_passage_chars = max_passage_chars
        self._text: Dict[str, str] = dict(zip(*corpus)) if corpus else {}

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("QA sanity check needs --data PATH to a multiple-choice jsonl (MMLU/GPQA-like)")
        return load_mcq(path, limit)

    def score(self, task: Task, prediction: Any) -> float:
        return 1.0 if prediction == task.gold else 0.0

    def _retrieve(self, task: Task) -> List[Tuple[str, str]]:
        if self.retriever is None:
            return []
        q = task.prompt
        if self.query_with_choices:
            q += "\n" + "\n".join(task.meta["choices"])
        ids = self.retriever.rank(q, self.k)
        return [(i, self._text[i][: self.max_passage_chars]) for i in ids]

    def run_all(self, tasks: Sequence[Task], agent: Agent) -> List[EpisodeResult]:
        if self.retriever is not None and self.corpus is not None:
            self.retriever.index(*self.corpus)
        return [self.run_episode(t, agent) for t in tasks]

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        hits = self._retrieve(task)
        user = build_prompt(task.prompt, task.meta["choices"], [t for _, t in hits])
        system = "You are answering a multiple-choice question. Use the context only if it helps."
        reply = agent.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], **agent.limits["decoding"]
        )
        letter = extract_letter(reply, len(task.meta["choices"]))
        s = self.score(task, letter)
        ctu = agent.tokens.count(system) + agent.tokens.count(user)
        return EpisodeResult(
            score=s, success=s == 1.0, ctu=ctu, steps=1, task_id=task.task_id, split=task.split,
            extra={"letter": letter, "n_passages": len(hits), "retrieved_ids": [i for i, _ in hits], "terminated": "answer"},
        )


def mock_llm() -> MockLLM:
    return MockLLM(["<answer>A</answer>"])
