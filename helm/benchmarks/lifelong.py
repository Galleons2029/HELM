"""LifelongAgentBench bridge.  Not validated against the real environment against the official DB / OS / KG environments.

What is implemented, from the benchmark paper (Zheng et al., 2025): tasks are executed *strictly sequentially* (memory
and experience persist across tasks); a task is ``<env, o_0, g>`` and yields a single reward on submitting a final
answer (success = 1); success is decided by an automatic label verifier of the environment; the replay baseline "Exp=k"
provides the k most recent *successful* trajectories to the agent.

What is NOT implemented: the official environments themselves (Docker DB / OS sandboxes, the KG SPARQL service) and
their verifiers. You adapt them to the ``LifelongEnv`` protocol below (reset / step / verify_label); this module only
drives them. The jsonl task format used by ``load_lifelong`` (``id``, ``instruction``, ``label``, ``env``) is this
adapter's own convention, not the official data schema.

``ReplayMemory(k)`` prepends the k most recent successful trajectories VERBATIM to the task prompt, so the prompt grows
linearly with k (and with trajectory length); that is the cost the paper contrasts with consolidation. Because the
replay text is part of every call's prompt it is billed in CTU at every step. ``oom=True`` is reported (instead of a
crash) when the prompt would exceed the context window -- the paper's "OOM" is a GPU out-of-memory event on its
own hardware, which a context-window check only approximates; ``max_prompt_tokens`` lets you emulate a tighter limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ..types import Status
from .base import BaseAdapter, EpisodeResult, MissingDataError, Task, summarize
from .harness import Agent, ToolRegistry, ToolResult, recall_then_answer_llm


# --------------------------------------------------------------------------------------------- env protocol
@dataclass
class StepOutcome:
    observation: str
    code: Optional[int] = 200  # HTTP-style; a failed SQL / shell command should be >= 400 so its status is FAILED
    done: bool = False


class LifelongEnv(Protocol):
    name: str

    def reset(self, task: Task) -> str:
        """Prepare the environment for ``task`` and return the initial observation o_0 (instruction, schema, ...)."""

    def step(self, action: str) -> StepOutcome:
        """Execute one natural-language/SQL/shell/SPARQL action."""

    def verify_label(self, task: Task, answer: Optional[str]) -> bool:
        """The environment's automatic label verification of the submitted answer / final state."""


# --------------------------------------------------------------------------------------------- replay baseline
@dataclass
class Trajectory:
    task_id: str
    instruction: str
    steps: List[Tuple[str, str]] = field(default_factory=list)  # (action, observation)
    answer: Optional[str] = None
    success: bool = False

    def render(self, i: int) -> str:
        lines = [f"Example {i} (solved earlier)", f"Instruction: {self.instruction}"]
        for a, o in self.steps:
            lines += [f"Action: {a}", f"Observation: {o}"]
        lines.append(f"Final answer: {self.answer}")
        return "\n".join(lines)


class ReplayMemory:
    """Exp=k replay: the k most recent successful trajectories, rendered verbatim."""

    def __init__(self, k: int):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self._traj: List[Trajectory] = []

    def reset(self) -> None:
        self._traj = []

    def add(self, traj: Trajectory) -> None:
        if traj.success:  # only successful trajectories are replayed
            self._traj.append(traj)

    def recent(self) -> List[Trajectory]:
        return self._traj[-self.k :] if self.k else []

    def prefix(self) -> str:
        rec = self.recent()
        if not rec:
            return ""
        return "Earlier solved tasks:\n\n" + "\n\n".join(t.render(i + 1) for i, t in enumerate(rec)) + "\n\nNow solve the new task.\n\n"

    def __len__(self) -> int:
        return len(self._traj)


# --------------------------------------------------------------------------------------------- data
def load_lifelong(path: str | Path, limit: Optional[int] = None) -> List[Task]:
    p = Path(path)
    if not p.is_file():
        raise MissingDataError(f"LifelongAgentBench task file not found: {p} (expected jsonl: id, instruction, label, env)")
    tasks: List[Task] = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        ins = next((row[k] for k in ("instruction", "question", "prompt") if k in row), None)
        if ins is None:
            raise MissingDataError(f"{p}:{n + 1}: no instruction/question/prompt field")
        tasks.append(
            Task(
                task_id=str(row.get("id", n)),
                prompt=str(ins),
                gold=row.get("label", row.get("answer")),
                meta={"env": row.get("env", ""), "t": len(tasks)},
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    if not tasks:
        raise MissingDataError(f"{p} contains no tasks")
    return tasks


# --------------------------------------------------------------------------------------------- adapter
class LifelongAdapter(BaseAdapter):
    name = "lifelong"

    def __init__(
        self,
        env: Optional[LifelongEnv] = None,
        replay: Optional[ReplayMemory] = None,
        keep_memory: bool = True,
        record_outcome: bool = True,
        max_steps: int = 15,
        max_prompt_tokens: Optional[int] = None,
    ):
        self.env = env
        self.replay = replay
        self.keep_memory = keep_memory
        self.record_outcome = record_outcome  # write the verified outcome to memory (SUCCESS / FAILED), like replay's success flag
        self.max_steps = max_steps
        self.max_prompt_tokens = max_prompt_tokens

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("LifelongAgentBench needs --data PATH to a jsonl task file (and an adapted LifelongEnv, see lifelong.py)")
        return load_lifelong(path, limit)

    def score(self, task: Task, prediction: Any) -> float:
        raise NotImplementedError("success is decided by the environment's verify_label()")

    def _tools(self, env: LifelongEnv, traj: Trajectory, state: Dict[str, Any]) -> ToolRegistry:
        def act(args: Dict[str, Any]) -> ToolResult:
            if state["done"]:
                return ToolResult(409, "the environment has finished; submit your final answer with <answer>")
            action = str(args.get("action", ""))
            out = env.step(action)
            state["done"] = out.done
            traj.steps.append((action, out.observation))
            return ToolResult(out.code, out.observation)

        return ToolRegistry().add("env", act, 'send one action to the environment, args {"action": "<sql | shell | query>"}')

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        if self.env is None:
            raise MissingDataError("no LifelongEnv given: adapt the official DB/OS/KG environment to the LifelongEnv protocol")
        env = self.env
        first = env.reset(task)
        prefix = self.replay.prefix() if self.replay is not None else ""
        text = f"{prefix}{task.prompt}"
        window = self.max_prompt_tokens or agent.cfg.reasoner.context_window
        need = agent.tokens.count(text) + agent.tokens.count(first)
        info: Dict[str, Any] = {"replay_tokens": agent.tokens.count(prefix), "n_replayed": len(self.replay.recent()) if self.replay else 0}
        if need > window:  # report, do not crash
            return EpisodeResult(0.0, False, ctu=0, steps=0, task_id=task.task_id, split=task.split,
                                 extra={**info, "oom": True, "prompt_tokens": need, "terminated": "oom"})
        traj = Trajectory(task.task_id, task.prompt)
        state: Dict[str, Any] = {"done": False}
        run = agent.run(
            text, self._tools(env, traj, state), keep_memory=self.keep_memory, task_version=task.version,
            max_steps=self.max_steps, initial_observation=first,
        )
        ok = bool(env.verify_label(task, run.answer)) if not run.overflow else False
        traj.answer, traj.success = run.answer, ok
        if self.replay is not None:
            self.replay.add(traj)
        if self.keep_memory and self.record_outcome and not run.overflow:
            agent.ingest(
                "episode_outcome",
                f"task: {task.prompt}\nfinal answer: {run.answer}\nverified: {ok}",
                Status.SUCCESS if ok else Status.FAILED,
            )
        return run.to_result(1.0 if ok else 0.0, ok, task.task_id, task.split, oom=run.overflow, answer=run.answer, **info)

    def run_all(self, tasks: Sequence[Task], agent: Agent) -> List[EpisodeResult]:
        """Strict sequential execution: tasks in the given order, memory (and replay buffer) persisting across them."""
        agent.reset()
        if self.replay is not None:
            self.replay.reset()
        out: List[EpisodeResult] = []
        done = 0.0
        for t, task in enumerate(tasks):
            res = self.run_episode(task, agent)
            done += res.score
            res.extra["t"] = t
            res.extra["running_success"] = done / (t + 1)
            out.append(res)
        return out

    def summarize(self, tasks: Sequence[Task], results: Sequence[EpisodeResult], seed: int = 0) -> List[Dict[str, Any]]:
        def row(name: str, idx: Sequence[int]) -> Dict[str, Any]:
            sub = [results[i] for i in idx]
            r = summarize(sub, seed=seed)
            oom = [bool(x.extra.get("oom")) for x in sub]
            r.update(split=name, n_oom=sum(oom), oom=any(oom), first_oom_index=next((i for i, o in enumerate(oom) if o), -1))
            return r

        groups: Dict[str, List[int]] = {}
        for i, t in enumerate(tasks):
            if t.meta.get("env"):
                groups.setdefault(str(t.meta["env"]), []).append(i)
        rows = [row("all", range(len(tasks)))]
        rows += [row(name, idx) for name, idx in sorted(groups.items())]
        return rows


def mock_llm():
    return recall_then_answer_llm("unknown")
