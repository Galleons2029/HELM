"""Unified agent scaffold shared by every method and every benchmark.

Paper: "All methods share the same agent scaffold and tool interfaces; we vary only the memory system and its
controller." Everything method-specific lives behind ``MemoryBackend``; this module owns the rest:

* a reasoner loop over a ``ChatClient`` with decoding locked from ``cfg.reasoner``;
* a ``ToolRegistry`` whose wrappers turn return codes into ``Status`` (``Status.from_code``);
* event-driven recall: ``<recall>q</recall>`` -> ``backend.recall`` -> a ``<memory>`` block shown to the model;
* every tool observation becomes an ``Observation`` (step, timestamp, tool, log_id, text, status, task_version), is given
  to ``backend.observe`` and, independently, to an ``AuditLog`` that EpiErr is computed from;
* CTU accounting per LLM call: ``system + backend.prompt_history_tokens() + retrieved + tool_obs``;
* one set of memory limits (``cfg.memory``): ``check_matched_budgets`` rejects a backend built from different budgets and
  the recall block is capped at ``cfg.memory.max_recall_tokens`` here as well, so no backend can exceed it.

Prompt model. The prompt is rebuilt statelessly at every call from: system prompt, task, the raw history window the
backend declares (the observations whose tokens make up ``backend.prompt_history_tokens()``), the ``<memory>`` block
retrieved in the previous turn, and the latest tool observation. Recalled memory and tool output are therefore billed
exactly once, in the call that consumes them, which is what makes the CTU sum equal the tokens actually sent. Headers
(``[step i] tool args -> code``) and the model's own generations are not part of CTU by the paper's definition.

EpiErr counts every item in ``RecallResult.items`` that survives the recall-token cap (each is audited against the raw
observation log). For ``LongContextBackend`` that list is the whole history, so its EpiErr is an exposure rate, not a
retrieval rate; and because that backend clears its per-episode token history on ``reset_episode(keep_memory=True)``, a
continual run carries no cross-task history in the rendered prompt or in CTU.

Action grammar: one action per reply, the *earliest* of ``<tool>name {json}</tool>``, ``<recall>...</recall>``,
``<answer>...</answer>``; text after it is discarded so the model cannot hallucinate the result. ``<think>`` blocks are
stripped before parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..backends import MemoryBackend, RecalledItem, RecallResult
from ..config import HelmConfig
from ..llm.client import ChatClient, MockLLM
from ..memory.tokens import TokenCounter
from ..methods import build_method
from ..metrics import AuditLog, CTUMeter, EpiErrMeter
from ..types import Observation, ReadContext, SourceType, Status
from .base import EpisodeResult, llm_source

# ----------------------------------------------------------------------------------------------- tools
@dataclass
class ToolResult:
    """What a tool wrapper returns. ``code`` is an HTTP-style return code (``Status.from_code``); ``None`` -> UNKNOWN."""

    code: Optional[int] = 200
    output: str = ""
    version: Optional[str] = None  # environment/tool version after this call (becomes ``task_version``)
    fields: Dict[str, str] = field(default_factory=dict)  # extra raw-log fields kept for provenance pointers


@dataclass
class Tool:
    name: str
    fn: Callable[[Dict[str, Any]], Any]
    description: str = ""

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        try:
            res = self.fn(args)
        except Exception as e:  # tool wrappers must never crash the episode: a crash is a FAILED observation
            return ToolResult(500, f"{type(e).__name__}: {e}")
        if isinstance(res, ToolResult):
            return res
        return ToolResult(200, str(res))  # lenient: a bare return value counts as a successful call


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def add(self, name: str, fn: Callable[[Dict[str, Any]], Any], description: str = "") -> "ToolRegistry":
        return self.register(Tool(name, fn, description))

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools)

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(404, f"unknown tool '{name}'. available: {self.names()}")
        return tool(args)

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}".rstrip(": ") for t in self._tools.values())


# ----------------------------------------------------------------------------------------------- parsing
_ACTION_RE = re.compile(r"<(answer|recall|tool)>(.*?)</\1>", re.DOTALL)
_OPEN_ANSWER_RE = re.compile(r"<answer>(.*)\Z", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_HEAD_RE = re.compile(r"\s*([A-Za-z_][\w.\-]*)\s*(.*)\Z", re.DOTALL)
_CODE_RE = re.compile(r"""["']?\bcode["']?\s*[:=]\s*\d{3}""")


def strip_think(text: str) -> str:
    """Drop reasoning blocks (Qwen3 emits ``<think>...</think>``) so tags inside them are not mistaken for actions."""
    if "</think>" in text and "<think>" not in text:  # template stripped the opening tag
        text = text.rsplit("</think>", 1)[1]
    text = _THINK_RE.sub("", text)
    i = text.find("<think>")  # unterminated block = truncated generation
    return text[:i] if i >= 0 else text


@dataclass
class Action:
    kind: str  # "tool" | "recall" | "answer" | "none"
    payload: str = ""


def parse_action(reply: str) -> Action:
    text = strip_think(reply)
    m = _ACTION_RE.search(text)
    if m:
        return Action(m.group(1), m.group(2).strip())
    m = _OPEN_ANSWER_RE.search(text)  # generation cut off before </answer>
    if m:
        return Action("answer", m.group(1).strip())
    return Action("none", text.strip())


def parse_tool_call(payload: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """``'name {"a": 1}'`` -> ``(name, {"a": 1}, None)``; on malformed input ``(name_or_None, {}, error)``."""
    m = _TOOL_HEAD_RE.match(payload)
    if not m:
        return None, {}, "malformed tool call; expected: <tool>name {json args}</tool>"
    name, rest = m.group(1), m.group(2).strip()
    if not rest:
        return name, {}, None
    try:
        args = json.loads(rest)
    except json.JSONDecodeError as e:
        return name, {}, f"arguments are not valid JSON ({e.msg})"
    if not isinstance(args, dict):
        return name, {}, "arguments must be a JSON object"
    return name, args, None


# ----------------------------------------------------------------------------------------------- helpers
def truncate_to_tokens(text: str, limit: int, counter: TokenCounter) -> Tuple[str, bool]:
    """Cut ``text`` to at most ``limit`` tokens (line boundaries first, then words). Returns ``(text, was_truncated)``."""
    if counter.count(text) <= limit:
        return text, False
    out: List[str] = []
    used = 0
    for line in text.splitlines():
        c = counter.count(line)
        if used + c > limit:
            break
        out.append(line)
        used += c
    if out:
        return "\n".join(out), True
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(" ".join(words[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]), True


def observation_text(res: ToolResult, counter: TokenCounter, max_tokens: int) -> str:
    """Tool output (bounded identically for every method) plus its return code, so the code is part of the raw log."""
    out, _ = truncate_to_tokens(res.output, max_tokens, counter)
    if res.code is not None and not _CODE_RE.search(out):
        out = f"{out}\ncode={res.code}" if out else f"code={res.code}"
    return out


_MATCHED_BUDGETS = (
    "budget_m1", "budget_m2", "budget_m3", "k1", "k2", "k3",
    "max_recall_tokens", "max_writes_per_step", "max_consolidations_per_episode",
)


def check_matched_budgets(backend: MemoryBackend, cfg: HelmConfig) -> None:
    """Every backend must have been built from the same memory budgets as the harness (paper: matched budgets)."""
    bcfg = getattr(backend, "cfg", None)
    if bcfg is None:
        return
    diffs = {
        f: (getattr(bcfg.memory, f), getattr(cfg.memory, f))
        for f in _MATCHED_BUDGETS
        if getattr(bcfg.memory, f) != getattr(cfg.memory, f)
    }
    if diffs:
        raise ValueError(f"backend memory budgets differ from the harness config (backend, harness): {diffs}")


class NullBackend(MemoryBackend):
    """No external memory. Only the raw history window of the scaffold remains (used by replay-only baselines)."""

    name = "No memory"

    def observe(self, obs: Observation, ctx: ReadContext) -> bool:
        self._note_obs(obs)
        return False

    def recall(self, query: str, ctx: ReadContext) -> RecallResult:
        return RecallResult(path="none")


# ----------------------------------------------------------------------------------------------- run record
@dataclass
class AgentRun:
    answer: Optional[str]
    terminated: str  # "answer" | "max_steps" | "context_overflow"
    steps: int  # LLM calls made
    ctu: CTUMeter
    epi: EpiErrMeter
    events: List[Dict[str, Any]]
    n_recalls: int = 0
    n_tool_calls: int = 0
    n_failed_tools: int = 0
    n_truncated_recalls: int = 0
    limits: Dict[str, Any] = field(default_factory=dict)

    @property
    def overflow(self) -> bool:
        return self.terminated == "context_overflow"

    def to_result(self, score: float, success: bool, task_id: str = "", split: str = "", **extra: Any) -> EpisodeResult:
        info: Dict[str, Any] = {
            "terminated": self.terminated,
            "n_recalls": self.n_recalls,
            "n_tool_calls": self.n_tool_calls,
            "n_failed_tools": self.n_failed_tools,
            "n_truncated_recalls": self.n_truncated_recalls,
            "epi_consumed": self.epi.consumed,
            "epi_invalid": self.epi.invalid,
            "epi_by_reason": dict(self.epi.by_reason),
            "limits": self.limits,
        }
        info.update(extra)
        return EpisodeResult(
            score=float(score),
            success=bool(success),
            ctu=self.ctu.total,
            epi_err=self.epi.rate,
            steps=self.steps,
            logs=self.events,
            extra=info,
            task_id=task_id,
            split=split,
        )


@dataclass
class _Entry:
    header: str
    text: str
    tokens: int

    def render(self) -> str:
        return f"{self.header}\n{self.text}"


DEFAULT_SYSTEM = (
    "You are a tool-using agent. In every turn emit exactly ONE action, then stop:\n"
    "  <tool>NAME {{json arguments}}</tool>    call a tool\n"
    "{recall_line}"
    "  <answer>FINAL ANSWER</answer>    finish the task\n"
    "Retrieved memory is context, not ground truth.\n"
    "{tools}"
)
_RECALL_LINE = "  <recall>what you need to remember</recall>    query long-term memory; the result arrives as a <memory> block\n"


def _default_clock(step: int) -> float:
    """Deterministic simulated wall clock (1 s per step) so runs are reproducible; staleness by days needs a custom clock."""
    return 1.7e9 + float(step)


# ----------------------------------------------------------------------------------------------- agent
class Agent:
    """The reasoner scaffold. Build one per (method, seed); call ``run`` per task."""

    def __init__(
        self,
        backend: MemoryBackend,
        llm: ChatClient,
        cfg: Optional[HelmConfig] = None,
        *,
        max_steps: int = 30,  # ASSUMPTION: scaffold knob, identical across methods
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
        allow_recall: bool = True,
        clock: Callable[[int], float] = _default_clock,
        max_obs_tokens: int = 1500,  # ASSUMPTION: same bound on every tool output, for every method
        strict_budgets: bool = True,
    ):
        self.backend = backend
        self.llm = llm
        self.cfg = cfg or HelmConfig()
        if strict_budgets:
            check_matched_budgets(backend, self.cfg)
        self.max_steps = max_steps
        self.seed = seed
        self.system_prompt = system_prompt
        self.allow_recall = allow_recall
        self.clock = clock
        self.max_obs_tokens = max_obs_tokens
        self.tokens: TokenCounter = backend.tokens
        self.audit = AuditLog(self.cfg.governance)  # independent of the backend under test
        self.source = llm_source(llm)
        self.last_messages: List[Dict[str, str]] = []
        self._gstep = 0
        self._hist: List[_Entry] = []
        self._version: Optional[str] = None
        r = self.cfg.reasoner
        self._gen: Dict[str, Any] = dict(temperature=r.temperature, top_p=r.top_p, max_tokens=r.max_tokens)
        if seed is not None:
            self._gen["seed"] = seed

    # ------------------------------------------------------------------ introspection
    @property
    def limits(self) -> Dict[str, Any]:
        m, r = self.cfg.memory, self.cfg.reasoner
        return {
            "max_recall_tokens": m.max_recall_tokens,
            "budgets": [m.budget_m1, m.budget_m2, m.budget_m3],
            "k": [m.k1, m.k2, m.k3],
            "context_window": r.context_window,
            "decoding": dict(self._gen),
        }

    def _ctx(self, step: int) -> ReadContext:
        return ReadContext(step=step, time=self.clock(step), task_version=self._version)

    def reset(self) -> None:
        """Forget everything (store, audit log, step counter): the start of a fresh stream / sequence / seed."""
        self.backend.reset_episode(keep_memory=False)
        self.audit.reset()
        self._gstep = 0
        self._hist = []
        self._version = None

    # ------------------------------------------------------------------ external observations
    def ingest(
        self,
        tool: str,
        text: str,
        status: Status = Status.UNKNOWN,
        *,
        version: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        source: SourceType = SourceType.TOOL,
    ) -> Observation:
        """Log an observation that did not come from a tool call (e.g. StreamBench feedback, a task outcome)."""
        self._gstep += 1
        step = self._gstep
        obs = Observation(
            step=step,
            timestamp=self.clock(step),
            tool=tool,
            log_id=f"L{step}",
            text=text,
            status=status,
            source=source,
            task_version=version if version is not None else self._version,
            fields={**(fields or {}), "text": text},
            error_payload=text if status == Status.FAILED else None,
        )
        self.audit.add(obs)
        self.backend.observe(obs, ReadContext(step=step, time=obs.timestamp, task_version=obs.task_version))
        return obs

    # ------------------------------------------------------------------ prompt
    def _system(self, tools: ToolRegistry) -> str:
        if self.system_prompt is not None:
            return self.system_prompt
        listing = f"Tools:\n{tools.describe()}\n" if tools.names() else "No tools are available.\n"
        return DEFAULT_SYSTEM.format(recall_line=_RECALL_LINE if self.allow_recall else "", tools=listing)

    @staticmethod
    def _window(entries: Sequence[_Entry], target_tokens: int) -> List[_Entry]:
        """Most recent entries whose token total fits the backend-declared raw-history size."""
        out: List[_Entry] = []
        used = 0
        for e in reversed(entries):
            if used + e.tokens > target_tokens:
                break
            out.append(e)
            used += e.tokens
        return out[::-1]

    def _messages(
        self, system: str, task_text: str, hist: Sequence[_Entry], memory: str, obs: Optional[_Entry], note: str
    ) -> List[Dict[str, str]]:
        parts = [f"Task:\n{task_text}"]
        if hist:
            parts.append("Recent observations:\n" + "\n".join(e.render() for e in hist))
        if memory:
            parts.append(memory)
        if obs is not None:
            parts.append("Latest observation:\n" + obs.render())
        if note:
            parts.append(note)
        parts.append("Next action?")
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]

    # ------------------------------------------------------------------ recall
    def _audit_item(self, it: RecalledItem, ctx: ReadContext) -> List[str]:
        reasons: Set[str] = set()
        for key, value in it.claims or [("", "")]:
            reasons.update(self.audit.classify(it.audit_ids, key, value, ctx))
        return sorted(reasons)

    def _recall(self, query: str, step: int, epi: EpiErrMeter, events: List[Dict[str, Any]]) -> Tuple[str, int, bool]:
        """Run ``backend.recall``; returns ``(memory_block, payload_tokens, truncated)`` and updates EpiErr."""
        ctx = self._ctx(step)
        res = self.backend.recall(query, ctx)
        payload, truncated = truncate_to_tokens(res.text, self.cfg.memory.max_recall_tokens, self.tokens)
        consumed = res.items if not truncated else [it for it in res.items if it.text in payload]
        meta = []
        for it in consumed:
            reasons = self._audit_item(it, ctx)
            epi.add(reasons)
            meta.append(
                {
                    "text": it.text[:200],
                    "layer": it.layer,
                    "status": it.status.value if it.status else None,
                    "step_id": it.step_id,
                    "audit_ids": list(it.audit_ids),
                    "flagged_invalid": it.flagged_invalid,
                    "audit_reasons": reasons,
                }
            )
        tokens = self.tokens.count(payload) if payload else 0
        events.append(
            {
                "type": "recall",
                "step": step,
                "query": query,
                "tokens": tokens,
                "path": res.path,
                "truncated": truncated,
                "fallback_used": res.fallback_used,
                "interventions": len(res.interventions),
                "items": meta,
            }
        )
        block = "<memory>\n" + (payload or "(no relevant memory)") + "\n</memory>"
        return block, tokens, truncated

    # ------------------------------------------------------------------ main loop
    def run(
        self,
        task_text: str,
        tools: Optional[ToolRegistry] = None,
        *,
        keep_memory: bool = False,
        task_version: Optional[str] = None,
        max_steps: Optional[int] = None,
        prefetch_query: Optional[str] = None,
        initial_observation: Optional[str] = None,
    ) -> AgentRun:
        """One episode. ``keep_memory=True`` is the continual setting: the store, the audit log and the step counter
        persist across episodes (so staleness and contradiction span tasks); otherwise everything is reset.

        ``initial_observation`` is the environment's first page/state (WebArena, Mind2Web, LifelongAgentBench): it is
        logged like any tool observation (so memory and the audit see it) and shown with the first call, instead of being
        baked into the task text where it would stay in every prompt."""
        tools = tools or ToolRegistry()
        self.backend.reset_episode(keep_memory=keep_memory)
        if not keep_memory:
            self.audit.reset()
            self._gstep = 0
        self._hist = []
        self._version = task_version
        system = self._system(tools)
        window = self.cfg.reasoner.context_window
        ctu = CTUMeter(system_tokens=self.tokens.count(system) + self.tokens.count(task_text))
        epi = EpiErrMeter()
        events: List[Dict[str, Any]] = []
        run = AgentRun(None, "max_steps", 0, ctu, epi, events, limits=self.limits)

        memory, mem_tokens = "", 0
        pending: Optional[_Entry] = None
        note, note_tokens = "", 0
        if prefetch_query:
            memory, mem_tokens, trunc = self._recall(prefetch_query, self._gstep, epi, events)
            run.n_recalls += 1
            run.n_truncated_recalls += int(trunc)

        if initial_observation is not None:
            self._gstep += 1
            pending, _ = self._log_observation(
                "env_init", {}, ToolResult(200, initial_observation), self._gstep, "[start] environment"
            )
            events.append({"type": "init_observation", "step": self._gstep, "tokens": pending.tokens})

        for i in range(max_steps if max_steps is not None else self.max_steps):
            self._gstep += 1
            step = self._gstep
            hist_tokens = self.backend.prompt_history_tokens()
            if pending is not None:  # the pending observation is billed as tool_obs, not as history
                hist_tokens = max(0, hist_tokens - pending.tokens)
            obs_tokens = (pending.tokens if pending else 0) + note_tokens
            if ctu.system_tokens + hist_tokens + mem_tokens + obs_tokens > window:
                run.terminated = "context_overflow"
                events.append({"type": "overflow", "step": step, "tokens": ctu.system_tokens + hist_tokens + mem_tokens + obs_tokens})
                break
            step_tokens = ctu.step(hist_tokens, mem_tokens, obs_tokens)
            shown = self._window(self._hist[:-1] if pending is not None else self._hist, hist_tokens)
            messages = self._messages(system, task_text, shown, memory, pending, note)
            self.last_messages = messages
            reply = self.llm.chat(messages, **self._gen)
            run.steps += 1
            ev: Dict[str, Any] = {
                "type": "call",
                "i": i,
                "step": step,
                "ctu": step_tokens,
                "hist_tokens": hist_tokens,
                "retrieved_tokens": mem_tokens,
                "tool_obs_tokens": obs_tokens,
                "hist_rendered_tokens": sum(e.tokens for e in shown),
            }
            memory, mem_tokens, pending, note, note_tokens = "", 0, None, "", 0
            act = parse_action(reply)

            if act.kind == "answer":
                ev["action"] = "answer"
                events.append(ev)
                run.answer, run.terminated = act.payload, "answer"
                break
            if act.kind == "recall" and self.allow_recall:
                ev["action"] = "recall"
                events.append(ev)
                memory, mem_tokens, trunc = self._recall(act.payload, step, epi, events)
                run.n_recalls += 1
                run.n_truncated_recalls += int(trunc)
                continue
            if act.kind == "tool":
                ev["action"] = "tool"
                events.append(ev)
                pending = self._run_tool(act.payload, tools, step, i, run, events)
                continue
            ev["action"] = "none"
            events.append(ev)
            note = "Format error: reply with exactly one <tool>, <recall> or <answer> action."
            note_tokens = self.tokens.count(note)
        return run

    def _log_observation(
        self, name: str, args: Dict[str, Any], res: ToolResult, step: int, header: str
    ) -> Tuple[_Entry, Observation]:
        """Wrap a tool result as an ``Observation`` and give it to the audit log and to the backend, identically."""
        status = Status.from_code(res.code)
        if res.version:
            self._version = res.version
        text = observation_text(res, self.tokens, self.max_obs_tokens)
        obs = Observation(
            step=step,
            timestamp=self.clock(step),
            tool=name,
            log_id=f"L{step}",
            text=text,
            status=status,
            source=SourceType.TOOL,
            task_version=self._version,
            fields={**res.fields, "args": json.dumps(args, sort_keys=True), "text": text},
            error_payload=res.output if status == Status.FAILED else None,
        )
        self.audit.add(obs)
        self.backend.observe(obs, ReadContext(step=step, time=obs.timestamp, task_version=obs.task_version))
        entry = _Entry(header, text, self.tokens.count(text))
        self._hist.append(entry)
        return entry, obs

    def _run_tool(
        self, payload: str, tools: ToolRegistry, step: int, i: int, run: AgentRun, events: List[Dict[str, Any]]
    ) -> _Entry:
        name, args, err = parse_tool_call(payload)
        res = ToolResult(400, f"error: {err}") if err is not None else tools.call(name or "", args)
        name = name or "invalid_call"
        header = f"[step {i + 1}] {name} {json.dumps(args, sort_keys=True)} -> code={res.code}"
        entry, obs = self._log_observation(name, args, res, step, header)
        run.n_tool_calls += 1
        run.n_failed_tools += int(obs.status == Status.FAILED)
        events.append(
            {"type": "tool", "step": step, "tool": name, "code": res.code, "status": obs.status.value, "obs_id": obs.obs_id, "tokens": entry.tokens}
        )
        return entry


def make_agent(method: str, cfg: Optional[HelmConfig] = None, *, llm: ChatClient, seed: int = 0, **agent_kw: Any) -> Agent:
    """Build the backend for a method name from ``helm.methods.available()`` (or ``"none"``) and wrap it in an ``Agent``."""
    cfg = cfg or HelmConfig()
    backend = NullBackend() if method == "none" else build_method(method, cfg, seed=seed, llm=llm)
    return Agent(backend, llm, cfg, seed=seed, **agent_kw)


def recall_then_answer_llm(answer: str = "unknown") -> MockLLM:
    """Offline stand-in reasoner: recalls once, then answers with a constant. Exercises the loop; solves nothing."""

    def fn(messages: List[Dict[str, str]]) -> str:
        if "<memory>" not in messages[-1]["content"]:
            return "<recall>relevant facts</recall>"
        return f"<answer>{answer}</answer>"

    return MockLLM(fn)
