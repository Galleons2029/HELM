"""GAIA adapter: local jsonl loader, quasi-exact-match scorer, Level 2/3 filtering, default offline tools.

Data. GAIA is gated on the Hugging Face hub, so the primary path is a LOCAL ``metadata.jsonl`` (or a ``.json`` list).
Recognised fields (snake_case, or the hub's capitalised names): ``task_id``, ``question`` / ``Question``, ``level`` /
``Level``, ``final_answer`` / ``Final answer``, ``file_name``. Test-split rows whose answer is withheld (``"?"``) cannot
be scored locally and are dropped by default. The hub layout (``<root>/2023/<split>/metadata.jsonl``) follows the dataset card.
``load_gaia_hf`` is an optional lazy ``datasets`` path: UNVERIFIED (needs the ``datasets`` package and an authorised
token; neither is available in the offline test environment).

Scorer. ``gaia_match`` is an independent implementation of the benchmark's *published description* (Mialon et al., 2023,
Sec. 3.2: "quasi exact match ... up to some normalization that is tied to the type of the ground truth"):

* gold is a number   -> strip ``$``, ``%`` and thousands separators from the answer, compare as floats;
* gold is a list (``,`` or ``;``) -> split both, same length, compare element-wise (numbers as floats, strings
  normalised);
* otherwise          -> lowercase and drop whitespace and punctuation, compare.

MUST BE DIFFED AGAINST THE OFFICIAL SCORER BEFORE ANY NUMBER IS REPORTED. Known judgement calls (modelled on the
official behaviour, not stated in the paper): list elements keep their punctuation (only case and whitespace are
normalised); a gold value such as ``1,000`` is treated as a two-element list, not as a number; numeric detection uses a
strict decimal regex (so ``inf``/``nan``/``1_000`` are strings here).

Tools. GAIA needs web browsing, which cannot be provided offline: pass a ``ToolRegistry`` with your search/browse tools
(``tools=``); the default registry only has a safe ``calculator`` and a sandboxed ``read_file``.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import BaseAdapter, EpisodeResult, MissingDataError, MissingDependencyError, Task, summarize
from .harness import Agent, ToolRegistry, ToolResult, recall_then_answer_llm

# --------------------------------------------------------------------------------------------- scorer
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_WS_RE = re.compile(r"\s")
_PUNCT = str.maketrans("", "", string.punctuation)


def _number(s: str, strip_symbols: bool) -> Optional[float]:
    s = s.strip()
    if strip_symbols:
        s = s.replace("$", "").replace("%", "").replace(",", "").strip()
    return float(s) if _NUMBER_RE.match(s) else None


def normalize_string(s: str, remove_punct: bool = True) -> str:
    s = _WS_RE.sub("", s).lower()
    return s.translate(_PUNCT) if remove_punct else s


def _split_list(s: str) -> List[str]:
    return re.split(r"[,;]", s)


def gaia_match(prediction: Optional[str], gold: str, list_elem_remove_punct: bool = False) -> bool:
    """GAIA-style quasi exact match (see module docstring for provenance and caveats)."""
    if prediction is None:
        return False
    pred, gold = str(prediction), str(gold)
    g_num = _number(gold, strip_symbols=False)
    if g_num is not None:
        p_num = _number(pred, strip_symbols=True)
        return p_num is not None and p_num == g_num
    if "," in gold or ";" in gold:
        g_elems, p_elems = _split_list(gold), _split_list(pred)
        if len(g_elems) != len(p_elems):
            return False
        for p, g in zip(p_elems, g_elems):
            g_n = _number(g, strip_symbols=False)
            if g_n is not None:
                p_n = _number(p, strip_symbols=True)
                if p_n is None or p_n != g_n:
                    return False
            elif normalize_string(p, list_elem_remove_punct) != normalize_string(g, list_elem_remove_punct):
                return False
        return True
    return normalize_string(pred) == normalize_string(gold)


_FINAL_RE = re.compile(r"^\s*final\s+answer\s*:\s*", re.IGNORECASE)


def clean_answer(answer: Optional[str]) -> Optional[str]:
    """Strip an echoed ``FINAL ANSWER:`` prefix (the benchmark's own answer template)."""
    return None if answer is None else _FINAL_RE.sub("", answer).strip()


# --------------------------------------------------------------------------------------------- loading
_ABSENT = {"", "?", None}


def _first(row: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _resolve_path(path: Path, split: Optional[str]) -> Path:
    if path.is_file():
        return path
    candidates = [path / "metadata.jsonl"]
    if split:
        candidates += [path / split / "metadata.jsonl", path / "2023" / split / "metadata.jsonl"]
    for c in candidates:
        if c.is_file():
            return c
    raise MissingDataError(
        f"GAIA metadata not found at {path}. Expected a metadata.jsonl file, or a directory containing "
        f"metadata.jsonl / <split>/metadata.jsonl (GAIA is gated on the HF hub: download it yourself and pass --data)."
    )


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise MissingDataError(f"{path}: expected a JSON list of task objects")
        return rows
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise MissingDataError(f"{path}:{n}: invalid JSON line ({e.msg})") from e
    return rows


def rows_to_tasks(
    rows: Sequence[Dict[str, Any]], levels: Optional[Sequence[int]] = (2, 3), limit: Optional[int] = None,
    require_gold: bool = True, split: str = "",
) -> List[Task]:
    tasks: List[Task] = []
    for idx, row in enumerate(rows):
        question = _first(row, "question", "Question")
        if question is None:
            raise MissingDataError(f"row {idx}: no 'question'/'Question' field (found keys: {sorted(row)})")
        level_raw = _first(row, "level", "Level")
        level = int(level_raw) if level_raw is not None else None
        if levels is not None and level not in levels:
            continue
        gold = _first(row, "final_answer", "Final answer")
        gold = None if gold in _ABSENT else str(gold)
        if gold is None and require_gold:
            continue
        tasks.append(
            Task(
                task_id=str(_first(row, "task_id") or f"gaia-{idx}"),
                prompt=str(question),
                gold=gold,
                split=split,
                meta={"level": level, "file_name": _first(row, "file_name") or ""},
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_gaia(
    path: str | Path, split: Optional[str] = None, levels: Optional[Sequence[int]] = (2, 3),
    limit: Optional[int] = None, require_gold: bool = True,
) -> List[Task]:
    p = Path(path)
    if not p.exists():
        raise MissingDataError(f"GAIA data path does not exist: {p}")
    tasks = rows_to_tasks(_read_rows(_resolve_path(p, split)), levels, limit, require_gold, split or "")
    if not tasks:
        raise MissingDataError(f"no GAIA tasks left under {p} after filtering (levels={levels}, require_gold={require_gold})")
    return tasks


def load_gaia_hf(
    config: str = "2023_all", split: str = "validation", levels: Optional[Sequence[int]] = (2, 3),
    limit: Optional[int] = None, token: Optional[str] = None,
) -> List[Task]:
    """UNVERIFIED lazy path through ``datasets`` (gated dataset ``gaia-benchmark/GAIA``; config names follow the dataset card)."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise MissingDependencyError("pip install datasets (and authenticate to the gated GAIA repo), or pass a local path") from e
    ds = load_dataset("gaia-benchmark/GAIA", config, split=split, token=token)
    return rows_to_tasks([dict(r) for r in ds], levels, limit, True, split)


# --------------------------------------------------------------------------------------------- tools
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor,
          "ceil": math.ceil, "abs": abs, "round": round, "min": min, "max": max, "sin": math.sin, "cos": math.cos}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expr: str) -> float:
    """Arithmetic only (no names, attributes or calls except a fixed math whitelist)."""

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    return ev(ast.parse(expr.strip(), mode="eval"))


def default_tools(file_root: Optional[str | Path] = None, max_file_chars: int = 20000) -> ToolRegistry:
    root = Path(file_root).resolve() if file_root else None

    def calculator(args: Dict[str, Any]) -> ToolResult:
        try:
            return ToolResult(200, repr(safe_eval(str(args.get("expression", "")))))
        except ZeroDivisionError:
            return ToolResult(422, "error: division by zero")
        except (ValueError, SyntaxError, TypeError, OverflowError) as e:
            return ToolResult(400, f"error: {e}")

    def read_file(args: Dict[str, Any]) -> ToolResult:
        if root is None:
            return ToolResult(404, "error: no attachment directory configured")
        target = (root / str(args.get("name", ""))).resolve()
        if root != target and root not in target.parents:
            return ToolResult(403, "error: path outside the attachment directory")
        if not target.is_file():
            return ToolResult(404, f"error: no such file '{args.get('name', '')}'")
        try:
            return ToolResult(200, target.read_text(encoding="utf-8")[:max_file_chars])
        except UnicodeDecodeError:
            return ToolResult(415, "error: binary files are not supported by this tool")

    reg = ToolRegistry()
    reg.add("calculator", calculator, 'evaluate arithmetic, args {"expression": "..."}')
    reg.add("read_file", read_file, 'read a text attachment, args {"name": "<file_name>"}')
    return reg


# --------------------------------------------------------------------------------------------- adapter
ANSWER_FORMAT = (
    "Answer with a number, as few words as possible, or a comma-separated list. For numbers do not use commas or "
    "units such as $ or %; for strings do not use articles or abbreviations."
)


class GaiaAdapter(BaseAdapter):
    name = "gaia"

    def __init__(
        self, levels: Optional[Sequence[int]] = (2, 3), file_root: Optional[str] = None,
        tools: Optional[ToolRegistry] = None, max_steps: Optional[int] = None,
    ):
        self.levels = levels
        self.file_root = file_root
        self.tools = tools
        self.max_steps = max_steps

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("GAIA needs --data PATH to a local metadata.jsonl (the dataset is gated and cannot be fetched here)")
        return load_gaia(path, split=split, levels=self.levels, limit=limit)

    def render_prompt(self, task: Task) -> str:
        text = f"{task.prompt}\n\n{ANSWER_FORMAT}"
        if task.meta.get("file_name"):
            text += f"\n\nAttached file: {task.meta['file_name']} (read it with the read_file tool)."
        return text

    def score(self, task: Task, prediction: Any) -> float:
        if task.gold is None:
            raise ValueError(f"task {task.task_id} has no gold answer")
        return 1.0 if gaia_match(prediction, task.gold) else 0.0

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        tools = self.tools if self.tools is not None else default_tools(self.file_root)
        run = agent.run(self.render_prompt(task), tools, task_version=task.version, max_steps=self.max_steps)
        pred = clean_answer(run.answer)
        s = self.score(task, pred) if pred is not None else 0.0
        return run.to_result(s, s == 1.0, task.task_id, task.split, level=task.meta.get("level"), prediction=pred)

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        rows = [{"split": "all", **summarize(results, seed=seed)}]
        for lv in sorted({t.meta.get("level") for t in tasks if t.meta.get("level") is not None}):
            sub = [r for t, r in zip(tasks, results) if t.meta.get("level") == lv]
            rows.append({"split": f"L{lv}", **summarize(sub, seed=seed)})
        return rows


def mock_llm():
    return recall_then_answer_llm("unknown")
