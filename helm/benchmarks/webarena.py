"""WebArena bridge.  Not validated against the real environment: it needs the official Docker web
environments, which are outside the offline test suite. Only config parsing and action-string parsing are unit-tested;
everything that touches a browser is an
interface contract written from the WebArena paper (action space, Fig. 4 / Fig. 7) and the official repo.

What is delegated, deliberately: task success is decided by the OFFICIAL functional validators
(``evaluation_harness.evaluator_router``), never by anything in this file. ``evaluate`` raises an actionable
``MissingDependencyError`` when they are not installed. Note that the official ``fuzzy_match`` validator calls an OpenAI
judge model, so evaluation also needs network access and an API key.

Assumptions about the official release (check against the checkout you use):

* task config JSON keys: ``task_id``, ``intent``, ``sites``, ``start_url``, ``require_login``, ``storage_state``,
  ``require_reset``, ``intent_template(_id)``, ``instantiation_dict`` and ``eval`` (``eval_types``,
  ``reference_answers``, ``reference_url``, ``program_html``, ...); ``config_files/test.raw.json`` holds ``__SHOPPING__``-
  style placeholders that ``scripts/generate_test_data.py`` fills from environment variables;
* environment variables ``SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, HOMEPAGE`` point at the sites;
* modules ``browser_env`` (``ScriptBrowserEnv``, ``create_id_based_action``, ``create_stop_action``) and
  ``evaluation_harness`` (``evaluator_router``); ``env.reset(options={"config_file": ...})`` and
  ``env.step(action) -> (obs, reward, terminated, truncated, info)`` with ``obs["text"]`` the accessibility tree.

Action grammar (paper, Fig. 7): ``click [id]``, ``hover [id]``, ``type [id] [content] [press_enter_after=0|1]`` (Enter is
pressed by default), ``press [key_comb]``, ``scroll [down|up]``, ``new_tab``, ``tab_focus [index]``, ``close_tab``,
``goto [url]``, ``go_back``, ``go_forward``, ``stop [answer]``.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .base import BaseAdapter, BenchmarkError, EpisodeResult, MissingDataError, MissingDependencyError, Task
from .harness import Agent, ToolRegistry, ToolResult, recall_then_answer_llm

WEBARENA_ENV_VARS = ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "MAP", "WIKIPEDIA", "HOMEPAGE")


# --------------------------------------------------------------------------------------------- task configs
@dataclass
class WebArenaConfig:
    task_id: int
    intent: str
    sites: List[str] = field(default_factory=list)
    start_url: str = ""
    require_login: bool = False
    storage_state: Optional[str] = None
    require_reset: bool = False
    eval: Dict[str, Any] = field(default_factory=dict)
    intent_template: str = ""
    config_file: Optional[str] = None  # per-task JSON on disk (the official evaluators read it)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def eval_types(self) -> List[str]:
        return list(self.eval.get("eval_types", []))

    @property
    def reference_answers(self) -> Any:
        return self.eval.get("reference_answers")


def substitute_placeholders(obj: Any, env: Optional[Mapping[str, str]] = None) -> Any:
    """Replace ``__NAME__`` with ``env[NAME]`` for the known site variables (unset variables are left untouched)."""
    env = os.environ if env is None else env
    if isinstance(obj, str):
        for name in WEBARENA_ENV_VARS:
            if name in env:
                obj = obj.replace(f"__{name}__", env[name])
        return obj
    if isinstance(obj, list):
        return [substitute_placeholders(x, env) for x in obj]
    if isinstance(obj, dict):
        return {k: substitute_placeholders(v, env) for k, v in obj.items()}
    return obj


def parse_config(raw: Mapping[str, Any], config_file: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> WebArenaConfig:
    if "intent" not in raw or "task_id" not in raw:
        raise MissingDataError(f"WebArena config lacks 'intent'/'task_id' (keys: {sorted(raw)})")
    raw = substitute_placeholders(dict(raw), env)
    return WebArenaConfig(
        task_id=int(raw["task_id"]),
        intent=str(raw["intent"]),
        sites=list(raw.get("sites", [])),
        start_url=str(raw.get("start_url", "")),
        require_login=bool(raw.get("require_login", False)),
        storage_state=raw.get("storage_state"),
        require_reset=bool(raw.get("require_reset", False)),
        eval=dict(raw.get("eval", {})),
        intent_template=str(raw.get("intent_template", "")),
        config_file=config_file,
        raw=dict(raw),
    )


def load_task_configs(path: str | Path, limit: Optional[int] = None, env: Optional[Mapping[str, str]] = None) -> List[WebArenaConfig]:
    """A directory of per-task ``<id>.json`` files, one such file, or a JSON list (e.g. ``test.raw.json``)."""
    p = Path(path)
    if not p.exists():
        raise MissingDataError(f"WebArena config path does not exist: {p}")
    out: List[WebArenaConfig] = []
    if p.is_dir():
        files = sorted((f for f in p.glob("*.json") if f.stem.isdigit()), key=lambda f: int(f.stem))
        for f in files:
            out.append(parse_config(json.loads(f.read_text(encoding="utf-8")), str(f), env))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            out.extend(parse_config(d, None, env) for d in data)
        else:
            out.append(parse_config(data, str(p), env))
    if not out:
        raise MissingDataError(f"no WebArena task configs found under {p} (expected <id>.json files or a JSON list)")
    return out[:limit] if limit is not None else out


# --------------------------------------------------------------------------------------------- action strings
class ActionParseError(ValueError):
    pass


@dataclass
class WebAction:
    kind: str
    element_id: Optional[str] = None
    text: Optional[str] = None
    press_enter: bool = True
    key: Optional[str] = None
    direction: Optional[str] = None
    url: Optional[str] = None
    tab_index: Optional[int] = None
    answer: Optional[str] = None


_NO_ARG = {"new_tab", "close_tab", "go_back", "go_forward"}
_ONE_ARG = {"click": "element_id", "hover": "element_id", "press": "key", "goto": "url", "scroll": "direction",
            "tab_focus": "tab_index", "stop": "answer"}
_TYPE_FLAG = re.compile(r"^\[(?P<id>[^\]]+)\]\s*\[(?P<text>.*)\]\s*\[(?P<flag>[01])\]\s*$", re.DOTALL)
_TYPE_PLAIN = re.compile(r"^\[(?P<id>[^\]]+)\]\s*\[(?P<text>.*)\]\s*$", re.DOTALL)
_ONE_BRACKET = re.compile(r"^\[(?P<v>.*)\]\s*$", re.DOTALL)
_FENCE = re.compile(r"```(?:[a-zA-Z]*\n)?(.*?)```", re.DOTALL)


def extract_action(reply: str) -> str:
    """The official prompt asks for the action inside triple backticks; fall back to the whole string."""
    m = _FENCE.search(reply)
    return (m.group(1) if m else reply).strip()


def parse_action_string(s: str) -> WebAction:
    s = extract_action(s)
    m = re.match(r"^\s*([a-z_]+)\s*(.*)$", s, re.DOTALL)
    if not m:
        raise ActionParseError(f"not an action: {s!r}")
    name, rest = m.group(1), m.group(2).strip()
    if name in _NO_ARG:
        if rest:
            raise ActionParseError(f"'{name}' takes no arguments, got {rest!r}")
        return WebAction(name)
    if name == "type":
        tm = _TYPE_FLAG.match(rest) or _TYPE_PLAIN.match(rest)
        if not tm:
            raise ActionParseError("expected: type [id] [content] [press_enter_after=0|1]")
        flag = tm.groupdict().get("flag")
        return WebAction("type", element_id=tm.group("id").strip(), text=tm.group("text"), press_enter=(flag != "0"))
    if name in _ONE_ARG:
        bm = _ONE_BRACKET.match(rest)
        if not bm:
            raise ActionParseError(f"expected: {name} [value]")
        v = bm.group("v").strip()
        if name == "scroll" and v not in ("up", "down"):
            raise ActionParseError("scroll direction must be 'up' or 'down'")
        if name == "tab_focus":
            if not v.isdigit():
                raise ActionParseError("tab_focus needs an integer tab index")
            return WebAction(name, tab_index=int(v))
        return WebAction(name, **{_ONE_ARG[name]: v})
    raise ActionParseError(f"unknown action '{name}'")


# --------------------------------------------------------------------------------------------- official env bridge
class WebArenaEnv(Protocol):
    """What the adapter needs from an environment. ``OfficialWebArenaEnv`` implements it over the official package."""

    page: Any

    def reset(self, config: WebArenaConfig) -> str: ...

    def step(self, action: WebAction) -> Tuple[str, bool]: ...  # (observation text, episode terminated)

    @property
    def trajectory(self) -> List[Any]: ...  # official-format trajectory (states and actions) for the validators


def check_environment(env: Optional[Mapping[str, str]] = None) -> None:
    env = os.environ if env is None else env
    missing = [v for v in WEBARENA_ENV_VARS if not env.get(v)]
    if missing:
        raise MissingDependencyError(
            "WebArena sites are not configured: set " + ", ".join(missing) + " to the URLs of your running web environments "
            "(see the WebArena README for the Docker images and ports)."
        )


def import_official() -> Dict[str, Any]:
    """Lazy import of the official packages. UNVERIFIED names; see the module docstring."""
    try:
        import browser_env  # type: ignore
        import evaluation_harness  # type: ignore
    except ImportError as e:
        raise MissingDependencyError(
            "the official WebArena packages are not importable (" + str(e) + "). Clone https://github.com/web-arena-x/webarena, "
            "`pip install -e .`, run `playwright install`, start the Docker sites, and set " + ", ".join(WEBARENA_ENV_VARS) + "."
        ) from e
    return {"browser_env": browser_env, "evaluation_harness": evaluation_harness}


def evaluate(config_file: str | Path, trajectory: Sequence[Any], page: Any = None, client: Any = None) -> float:
    """Delegate to the official functional validators; returns their 0/1 score."""
    if not config_file:
        raise BenchmarkError("the official validators read a per-task config file; generate <id>.json files first (scripts/generate_test_data.py)")
    off = import_official()
    evaluator = off["evaluation_harness"].evaluator_router(str(config_file))
    kwargs: Dict[str, Any] = {"trajectory": list(trajectory), "config_file": str(config_file), "page": page}
    if client is not None:
        kwargs["client"] = client
    return float(evaluator(**kwargs))


class OfficialWebArenaEnv:
    """Thin wrapper over ``browser_env.ScriptBrowserEnv`` (id-based accessibility-tree observations). UNVERIFIED."""

    def __init__(self, headless: bool = True):
        check_environment()
        self._off = import_official()["browser_env"]
        self._env = self._off.ScriptBrowserEnv(
            headless=headless, observation_type="accessibility_tree", current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
        )
        self._traj: List[Any] = []
        self.page: Any = None

    @property
    def trajectory(self) -> List[Any]:
        return self._traj

    def reset(self, config: WebArenaConfig) -> str:
        if not config.config_file:
            raise BenchmarkError("per-task config file required to reset the official environment")
        obs, info = self._env.reset(options={"config_file": config.config_file})
        self._traj = [{"observation": obs, "info": info}]
        self.page = self._env.page
        return str(obs["text"])

    def step(self, action: WebAction) -> Tuple[str, bool]:
        b = self._off
        if action.kind == "stop":
            act = b.create_stop_action(action.answer or "")
        elif action.kind in ("click", "hover"):
            act = b.create_id_based_action(f"{action.kind} [{action.element_id}]")
        elif action.kind == "type":
            act = b.create_id_based_action(f"type [{action.element_id}] [{action.text}] [{1 if action.press_enter else 0}]")
        else:
            arg = {"press": action.key, "scroll": action.direction, "goto": action.url, "tab_focus": action.tab_index}.get(action.kind)
            act = b.create_id_based_action(action.kind if arg is None else f"{action.kind} [{arg}]")
        self._traj.append(act)
        obs, _reward, terminated, truncated, info = self._env.step(act)
        self._traj.append({"observation": obs, "info": info})
        self.page = self._env.page
        return str(obs["text"]), bool(terminated or truncated or action.kind == "stop")


# --------------------------------------------------------------------------------------------- adapter
class WebArenaAdapter(BaseAdapter):
    name = "webarena"

    def __init__(
        self,
        env_factory: Optional[Callable[[], WebArenaEnv]] = None,
        evaluate_fn: Callable[..., float] = evaluate,
        max_steps: int = 30,
    ):
        self.env_factory = env_factory
        self.evaluate_fn = evaluate_fn
        self.max_steps = max_steps

    def load(self, path: Optional[str] = None, split: Optional[str] = None, limit: Optional[int] = None) -> List[Task]:
        if path is None:
            raise MissingDataError("WebArena needs --data PATH to a directory of <id>.json task configs (or a JSON list)")
        cfgs = load_task_configs(path, limit=limit)
        return [
            Task(task_id=str(c.task_id), prompt=c.intent, gold=None, split=split or "", meta={"config": c})
            for c in cfgs
        ]

    def score(self, task: Task, prediction: Any) -> float:
        raise NotImplementedError("WebArena success comes from the official validators; call evaluate() on a finished trajectory")

    def _tools(self, env: WebArenaEnv, state: Dict[str, Any]) -> ToolRegistry:
        def browser(args: Dict[str, Any]) -> ToolResult:
            try:
                action = parse_action_string(str(args.get("action", "")))
            except ActionParseError as e:
                return ToolResult(400, f"error: {e}")
            text, done = env.step(action)
            state["done"] = state["done"] or done
            if action.kind == "stop":
                state["stopped"] = True
                return ToolResult(200, "episode ended; now reply with <answer>done</answer>")
            return ToolResult(200, text)

        reg = ToolRegistry()
        reg.add("browser", browser, 'perform one WebArena action, args {"action": "click [12]"}; actions: click, hover, type [id] [text] [0|1], '
                "press, scroll [up|down], new_tab, tab_focus, close_tab, goto, go_back, go_forward, stop [answer]")
        return reg

    def run_episode(self, task: Task, agent: Agent) -> EpisodeResult:
        if self.env_factory is None:
            check_environment()
            self.env_factory = OfficialWebArenaEnv
        cfg: WebArenaConfig = task.meta["config"]
        env = self.env_factory()
        first = env.reset(cfg)
        state: Dict[str, Any] = {"done": False, "stopped": False}
        prompt = f"{cfg.intent}\n\nStart URL: {cfg.start_url}"
        run = agent.run(prompt, self._tools(env, state), task_version=task.version, max_steps=self.max_steps, initial_observation=first)
        if not state["stopped"]:  # close the trajectory the way the official agent does: a final stop action
            env.step(WebAction("stop", answer=run.answer or ""))
        score = float(self.evaluate_fn(cfg.config_file, env.trajectory, env.page))
        return run.to_result(score, score >= 1.0, task.task_id, task.split, answer=run.answer)


def materialize_configs(configs: Sequence[WebArenaConfig], out_dir: Optional[str] = None) -> List[WebArenaConfig]:
    """Write configs that came from a JSON list to ``<out_dir>/<id>.json`` so the official validators can read them."""
    d = Path(out_dir or tempfile.mkdtemp(prefix="webarena_cfg_"))
    d.mkdir(parents=True, exist_ok=True)
    for c in configs:
        f = d / f"{c.task_id}.json"
        f.write_text(json.dumps(c.raw), encoding="utf-8")
        c.config_file = str(f)
    return list(configs)


def mock_llm():
    return recall_then_answer_llm("N/A")
