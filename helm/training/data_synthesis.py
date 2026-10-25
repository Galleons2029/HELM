"""Training-data synthesis (Appendix "Nested-memory synthesis details").

Three generators share one record schema (JSONL, one record per line):

``generate_offline_dataset``      fully offline, template based, built-in seed list, no downloads. Used by smoke runs and tests.
``generate_teacher_dataset``      teacher LLM (any ``helm.llm.client.ChatClient``) writes the M1/M2/M3 triplet for a seed
                                  document/interaction; verified here only with ``MockLLM``.
``load_fineweb_edu_seeds``        optional FineWeb-Edu seed loader (edu score >= 3.0). NOT VERIFIED in the offline test suite (the
                                  ``datasets`` package is not installed and there is no network); imported lazily.

``make_synthetic_controller_dataset`` is unrelated to the retriever data: it builds random ``ControllerState`` prompts
with the ``HeuristicController`` decision as the teacher completion, used by the controller smoke tests.

Record schema (retriever data)::

    id, kind ("short" | "long"), seed_id, layer (layer of the positive, 1..3), instruction, query,
    positive, negatives=[{text, violation, layer}] (exactly ``n_counterfactuals``), triplet={m1, m2, m3}, meta={...}

Paper-stated (Appendix): hierarchical triplet M1/M2/M3 per seed; queries whose positive can come from any layer;
needle-in-a-haystack expansion to 512-8192 tokens with distractor episodes; epistemic counterfactuals (mismatched
timestamps, failed tool status, inconsistent entity state); FineWeb-Edu seeds with educational score >= 3.0;
M3 keys of about 8 tokens. Per-query count of 3 counterfactuals is from the task spec.

ASSUMPTIONS (everything below is NOT stated in the paper):
  * Trace format of M1 (``[step N] user/tool_call/tool_output`` lines with a JSON payload holding entity, attr, value,
    code, ts, version) and of M2 (procedure + "Outcome: entity.attr = value as of ..."); chosen to match the fields the
    claim extractor and the tool-aware hashing embedder read.
  * One counterfactual per violation kind (stale, failed, inconsistent) per query. Stale: 8-40 days older, previous
    version, step >= 250 older (beyond the 200-step / 7-day windows of the repro table). Failed: HTTP-style error
    code, no value. Inconsistent: different value, 20-360 minutes older.
  * M3 entries carry no validity state, so for queries whose positive is an M3 entry each counterfactual is rendered
    randomly as an M1 or M2 trace.
  * Long-horizon records place the haystack (needle + distractor episodes) on the *document* side: the positive is the
    haystack containing the valid needle, the negatives are the same haystack with a counterfactual needle. Their
    queries are either evidence-level (M1 style) or thematic (M3 style) cues. Length is drawn log-uniformly in
    [min_ctx_tokens, max_ctx_tokens] and measured with ``TokenCounter`` (regex word/punctuation proxy).
  * Distractors never share the needle's (entity, attr) key; 80% are successful traces, 20% failed ones.
  * Instruction strings per target layer, query paraphrase templates, built-in seed list, layer sampling (uniform).
  * Teacher prompt wording and the ``<<placeholder>>`` convention (the teacher writes M1/M2 with placeholders for
    ts/code/value/version/step so validity fields can be controlled exactly when the counterfactuals are rendered).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..controller.base import ControllerState, format_ops, render_state
from ..controller.heuristic import HeuristicController
from ..controller.llm import SYSTEM_PROMPT
from ..llm.client import ChatClient
from ..memory.tokens import DEFAULT_COUNTER, TokenCounter
from ..types import Status

VIOLATIONS: Tuple[str, ...] = ("stale", "failed", "inconsistent")
NOW = datetime(2025, 6, 30, 12, 0, 0, tzinfo=timezone.utc)  # ASSUMPTION: fixed reference "current time" for ts fields

INSTRUCTIONS: Dict[int, str] = {  # ASSUMPTION: wording of the retrieval-granularity instruction per target layer
    1: "Retrieve the valid, current tool log evidence for this query",
    2: "Retrieve the valid, current procedural recall for this query",
    3: "Retrieve the thematic index entry for this intent",
}
_FAILED_CODES = (404, 429, 500, 502, 503)  # ASSUMPTION: HTTP-style failure codes (Status.from_code -> FAILED)
_ERRORS = ("upstream timeout", "service unavailable", "rate limit exceeded", "resource not found", "bad gateway")


# --------------------------------------------------------------------------------------------- seeds and facts
@dataclass(frozen=True)
class SeedSpec:
    """One built-in seed interaction family (tool + attribute); entities and values are drawn at generation time."""

    topic: str
    tool: str
    attr: str
    entities: Tuple[str, ...]
    intent: str
    theme: str
    user: str
    steps: Tuple[str, str]
    why: str
    int_range: Optional[Tuple[int, int]] = None
    choices: Tuple[str, ...] = ()


# ASSUMPTION: the built-in seed list (tool families, entities, value ranges, wording) is invented for offline use.
BUILTIN_SEEDS: Tuple[SeedSpec, ...] = (
    SeedSpec("inventory", "inventory_api.get_stock", "stock", ("widget", "gadget", "bracket", "sprocket"),
             "check the stock level", "inventory stock check",
             "How many units of {entity} are in the warehouse right now?",
             ("query the inventory API for {entity}", "read the stock field of the response"),
             "the reorder decision depends on current stock", int_range=(0, 500)),
    SeedSpec("crm", "crm.lookup_contact", "email", ("customer", "vendor", "partner", "intern"),
             "look up the contact email", "crm contact email lookup",
             "What is the email address on file for {entity}?",
             ("search the CRM for {entity}", "extract the email field of the contact record"),
             "outreach must reach the verified contact",
             choices=("ana@example.org", "bo@example.org", "cy@example.org", "di@example.org", "ed@example.org")),
    SeedSpec("shipping", "carrier_api.track_parcel", "status", ("parcel", "pallet", "crate", "envelope"),
             "track the parcel status", "shipping parcel status tracking",
             "Where is {entity} in the delivery pipeline?",
             ("call the carrier tracking API with the id of {entity}", "read the latest status event"),
             "the customer notice depends on the delivery state",
             choices=("in_transit", "delivered", "held_at_customs", "out_for_delivery")),
    SeedSpec("database", "sql.count_rows", "row_count", ("orders", "customers", "invoices", "sessions"),
             "count the table rows", "database table row count",
             "How many rows does the {entity} table hold?",
             ("run a COUNT query on the {entity} table", "read the single numeric result"),
             "the capacity plan depends on table growth", int_range=(1000, 90000)),
    SeedSpec("deploy", "ci.get_deployed_version", "build", ("svc_auth", "svc_billing", "svc_search", "svc_feed"),
             "check the deployed build", "deployment build version check",
             "Which build of {entity} is running in production?",
             ("query the CI deploy registry for {entity}", "read the build field of the latest release"),
             "rollbacks need the exact running build",
             choices=("1.4.2", "1.5.0", "2.0.1", "2.1.3", "3.0.0")),
    SeedSpec("weather", "weather_api.get_temperature", "temp_c", ("lisbon", "oslo", "kyoto", "lima"),
             "read the current temperature", "weather temperature reading",
             "What is the temperature in {entity} at the moment?",
             ("call the weather API for {entity}", "read the temp_c field of the observation"),
             "the travel advice depends on the current weather", int_range=(-5, 38)),
    SeedSpec("flights", "airfare_api.quote", "price_usd", ("sfo_jfk", "lhr_cdg", "nrt_sin", "gru_mia"),
             "get the fare quote", "flight fare quote lookup",
             "What is the cheapest fare on the {entity} route today?",
             ("request a quote for the {entity} route", "read the lowest price_usd in the quotes"),
             "the booking decision depends on the quoted fare", int_range=(80, 1200)),
    SeedSpec("calendar", "calendar.get_event", "start_time", ("standup", "retro", "planning", "review"),
             "look up the meeting time", "calendar meeting start time",
             "When does the {entity} meeting start?",
             ("fetch the {entity} event from the calendar", "read the start_time of the event"),
             "attendees need the confirmed slot", choices=("09:00", "10:30", "14:00", "16:45")),
    SeedSpec("flags", "flags.get", "enabled", ("dark_mode", "beta_ui", "fast_checkout", "new_search"),
             "read the feature flag", "feature flag state lookup",
             "Is the {entity} feature flag switched on?",
             ("ask the flag service for {entity}", "read the enabled field"),
             "the rollout plan depends on the flag state", choices=("true", "false")),
    SeedSpec("filesystem", "fs.stat", "mode", ("app_conf", "db_conf", "nginx_conf", "cron_conf"),
             "inspect the file permissions", "filesystem file permission check",
             "What permission mode does the {entity} file have?",
             ("stat the {entity} file", "read the octal mode"),
             "the hardening audit depends on exact permissions", choices=("0644", "0600", "0755", "0640")),
)


@dataclass
class Facts:
    """Structured content of one tool episode. Rendered into M1/M2 by a ``Template``."""

    tool: str
    entity: str
    attr: str
    value: Optional[str]
    code: int
    ts: datetime
    version: str
    step: int
    user: str = ""
    error: Optional[str] = None

    @property
    def status(self) -> Status:
        return Status.from_code(self.code)

    @property
    def key(self) -> str:
        return f"{self.entity}.{self.attr}"

    def ts_iso(self) -> str:
        return self.ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_value(v: Optional[str]) -> Any:
    if v is None:
        return None
    try:
        return int(v) if str(int(v)) == v else v  # "0644" stays a string
    except ValueError:
        return v


def _draw_value(spec: SeedSpec, rng: random.Random, exclude: Optional[str] = None) -> str:
    for _ in range(50):
        if spec.choices:
            v = rng.choice(spec.choices)
        else:
            lo, hi = spec.int_range or (0, 1000)
            v = str(rng.randint(lo, hi))
        if v != exclude:
            return v
    return v + "0"


def _draw_entity(spec: SeedSpec, rng: random.Random) -> str:
    return f"{rng.choice(spec.entities)}_{rng.randint(100, 9999)}"  # numeric suffix keeps (entity, attr) keys unique


def draw_facts(spec: SeedSpec, rng: random.Random, step: Optional[int] = None) -> Facts:
    """A successful, current episode (the positive's evidence)."""
    entity = _draw_entity(spec, rng)
    return Facts(
        tool=spec.tool,
        entity=entity,
        attr=spec.attr,
        value=_draw_value(spec, rng),
        code=rng.choice((200, 200, 200, 201)),
        ts=NOW - timedelta(minutes=rng.randint(60, 20 * 60)),
        version=f"v{rng.randint(3, 9)}",
        step=step if step is not None else rng.randint(300, 900),
        user=spec.user.format(entity=entity),
    )


def _prev_version(v: str, rng: random.Random) -> str:
    m = re.match(r"v(\d+)$", v)
    n = int(m.group(1)) if m else 3
    return f"v{max(0, n - rng.randint(1, 2))}"


def make_counterfactual(
    f: Facts, kind: str, rng: random.Random, spec: Optional[SeedSpec] = None, keep_step: bool = False
) -> Facts:
    """Same topic/entity/attribute/lexical content as ``f`` but violating one validity condition."""
    g = replace(f)
    # ASSUMPTION: the magnitudes below (days/minutes/steps/codes) are chosen to violate the repro-table windows
    # (200 steps, 7 days); the paper only names the violation types.
    if kind == "stale":
        g.ts = f.ts - timedelta(days=rng.uniform(8, 40))
        g.version = _prev_version(f.version, rng)
        if not keep_step:
            g.step = max(0, f.step - rng.randint(250, 600))
        if spec is not None and rng.random() < 0.5:
            g.value = _draw_value(spec, rng, exclude=f.value)
    elif kind == "failed":
        g.code = rng.choice(_FAILED_CODES)
        g.value = None
        g.error = f"{rng.choice(_ERRORS)} while reading {f.attr} of {f.entity}"
    elif kind == "inconsistent":
        g.ts = f.ts - timedelta(minutes=rng.randint(20, 360))
        g.value = _draw_value(spec, rng, exclude=f.value) if spec is not None else (f.value or "") + "7"
    else:
        raise ValueError(f"unknown violation kind: {kind}")
    return g


# --------------------------------------------------------------------------------------------- templates
class Template:
    """Renders ``Facts`` into the three memory layers."""

    def m1(self, f: Facts) -> str:
        raise NotImplementedError

    def m2(self, f: Facts) -> str:
        raise NotImplementedError

    def m3(self, f: Facts) -> str:
        raise NotImplementedError


class SeedTemplate(Template):
    """Offline template for a ``SeedSpec`` (the synthesis a teacher LLM would otherwise provide)."""

    def __init__(self, spec: SeedSpec):
        self.spec = spec

    def m1(self, f: Facts) -> str:
        payload: Dict[str, Any] = {
            "entity": f.entity, "attr": f.attr, "value": _json_value(f.value), "code": f.code,
            "ts": f.ts_iso(), "version": f.version,
        }
        if f.error:
            payload["error"] = f.error
        return "\n".join(
            [
                f"[step {f.step}] user: {f.user}",
                f'[step {f.step}] tool_call: {f.tool}(entity="{f.entity}", attr="{f.attr}")',
                f"[step {f.step}] tool_output: {json.dumps(payload)}",
            ]
        )

    def m2(self, f: Facts) -> str:
        s = self.spec
        head = (
            f"To {s.intent} for {f.entity}: first {s.steps[0].format(entity=f.entity)}, "
            f"then {s.steps[1].format(entity=f.entity)}, because {s.why}. "
        )
        if f.status == Status.FAILED:
            tail = (
                f"Outcome: the call failed with code {f.code} ({f.error}), so no value for {f.key} was obtained "
                f"as of {f.ts:%Y-%m-%d %H:%M} (step {f.step}, version {f.version}, status FAILED)."
            )
        else:
            tail = (
                f"Outcome: {f.key} = {f.value} as of {f.ts:%Y-%m-%d %H:%M} "
                f"(step {f.step}, version {f.version}, status {f.status.value})."
            )
        return head + tail

    def m3(self, f: Facts) -> str:
        return f"{self.spec.theme} for {f.entity} ({f.attr})"


_PLACEHOLDERS = ("ts", "code", "value", "version", "step")


class TeacherTemplate(Template):
    """M1/M2 strings written by a teacher LLM with ``<<ts>>``-style placeholders for the validity fields."""

    def __init__(self, m1: str, m2: str, m3: str):
        self.m1_tpl, self.m2_tpl, self.m3_text = m1, m2, m3

    @staticmethod
    def _fill(tpl: str, f: Facts) -> str:
        vals = {
            "ts": f.ts_iso(), "code": str(f.code), "value": "null" if f.value is None else str(f.value),
            "version": f.version, "step": str(f.step),
        }
        for k, v in vals.items():
            tpl = tpl.replace(f"<<{k}>>", v)
        return tpl

    def m1(self, f: Facts) -> str:
        out = self._fill(self.m1_tpl, f)
        return out + (f"\n[step {f.step}] tool_error: {f.error}" if f.error else "")

    def m2(self, f: Facts) -> str:
        out = self._fill(self.m2_tpl, f)
        if f.status == Status.FAILED:
            out += f" (status FAILED, code {f.code})"
        return out

    def m3(self, f: Facts) -> str:
        return self.m3_text


# ASSUMPTION: query paraphrase templates per target layer.
_QUERY_TEMPLATES: Dict[int, Tuple[str, ...]] = {
    1: (
        "exact {attr} of {entity} returned by {tool} with its return code",
        "{tool} output for {entity}: {attr} value and status",
        "raw tool log showing the {attr} of {entity}",
    ),
    2: (
        "how to {intent} for {entity} and what the outcome was",
        "procedure used to {intent} of {entity} and its result",
        "what was learned when we tried to {intent} for {entity}",
    ),
    3: (
        "{topic} {attr} lookup",
        "memory about {theme}",
        "{topic} task: {intent}",
    ),
}


def sample_query(spec: SeedSpec, f: Facts, layer: int, rng: random.Random) -> str:
    tpl = rng.choice(_QUERY_TEMPLATES[layer])
    return tpl.format(attr=f.attr, entity=f.entity, tool=f.tool, intent=spec.intent, topic=spec.topic, theme=spec.theme)


def generic_query(f: Facts, layer: int) -> str:
    """Query for triplets without a ``SeedSpec`` (teacher path)."""
    if layer == 1:
        return f"exact {f.attr} of {f.entity} returned by {f.tool} with its return code"
    if layer == 2:
        return f"how to obtain the {f.attr} of {f.entity} and what the outcome was"
    return f"{f.tool.split('.')[0]} {f.attr} lookup"


# --------------------------------------------------------------------------------------------- long-horizon expansion
def distractor_episode(
    seeds: Sequence[SeedSpec], rng: random.Random, exclude_key: str, step: int, ts: datetime
) -> str:
    """A plausible episode on another key (80% successful, 20% failed; ASSUMPTION: the 80/20 split)."""
    for _ in range(20):
        spec = rng.choice(seeds)
        f = draw_facts(spec, rng, step=step)
        if f.key != exclude_key:
            break
    f.ts = ts
    if rng.random() < 0.2:
        f = make_counterfactual(f, "failed", rng, spec, keep_step=True)
    return SeedTemplate(spec).m1(f)


def build_haystack(
    needle_variants: Dict[str, Facts],
    template: Template,
    seeds: Sequence[SeedSpec],
    rng: random.Random,
    target_tokens: int,
    max_tokens: int,
    counter: TokenCounter = DEFAULT_COUNTER,
    extra_distractors: Sequence[str] = (),
) -> Tuple[Dict[str, str], int, int]:
    """Interleave the needle with distractor episodes until ~``target_tokens``.

    ``needle_variants`` maps a label ("positive", violation kinds) to the needle's ``Facts``; every variant is
    rendered into the *same* distractor sequence so the only difference between positive and negatives is the needle.
    Returns ``({label: text}, needle_index, n_distractors)``.
    """
    pos = needle_variants["positive"]
    texts = {k: template.m1(v) for k, v in needle_variants.items()}
    counts = [counter.count(t) for t in texts.values()]
    lo, hi = min(counts), max(counts)  # every variant must land inside [target, max_tokens]
    episodes: List[str] = []
    step0 = max(0, pos.step - rng.randint(0, 40))
    added = 0
    while lo + added < target_tokens:
        for _ in range(5):
            if extra_distractors and rng.random() < 0.3:
                ep = rng.choice(list(extra_distractors))
            else:
                ts = NOW - timedelta(minutes=rng.randint(1, 2000))
                ep = distractor_episode(seeds, rng, pos.key, step0 + len(episodes), ts)
            n = counter.count(ep)
            if hi + added + n <= max_tokens:
                break
        else:
            break
        episodes.append(ep)
        added += n
    idx = rng.randint(0, len(episodes))
    out = {k: "\n\n".join(episodes[:idx] + [t] + episodes[idx:]) for k, t in texts.items()}
    return out, idx, len(episodes)


# --------------------------------------------------------------------------------------------- dataset assembly
@dataclass
class SynthConfig:
    n: int = 1000  # ASSUMPTION: default size (paper reports no value usable here)
    seed: int = 0
    n_counterfactuals: int = 3  # task spec: 3 counterfactuals per query
    layer_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    long_frac: float = 0.25  # ASSUMPTION
    min_ctx_tokens: int = 512  # paper: 512-8192 tokens
    max_ctx_tokens: int = 8192
    violation_kinds: Tuple[str, ...] = VIOLATIONS


def _violation_sequence(cfg: SynthConfig, rng: random.Random) -> List[str]:
    kinds = list(cfg.violation_kinds)
    rng.shuffle(kinds)
    return [kinds[j % len(kinds)] for j in range(cfg.n_counterfactuals)]


def _render_at(template: Template, f: Facts, layer: int) -> str:
    return template.m1(f) if layer == 1 else template.m2(f)


def build_record(
    idx: int,
    spec: Optional[SeedSpec],
    facts: Facts,
    template: Template,
    cfg: SynthConfig,
    rng: random.Random,
    seeds: Sequence[SeedSpec] = BUILTIN_SEEDS,
    counter: TokenCounter = DEFAULT_COUNTER,
    query: Optional[str] = None,
    extra_distractors: Sequence[str] = (),
    generator: str = "offline_template",
) -> Dict[str, Any]:
    """One training record from one triplet (shared by the offline and teacher generators)."""
    m1, m2, m3 = template.m1(facts), template.m2(facts), template.m3(facts)
    long = rng.random() < cfg.long_frac
    # ASSUMPTION: uniform layer sampling; long-horizon records target the M1 needle with an M1- or M3-style cue.
    layer = 1 if long else rng.choices((1, 2, 3), weights=cfg.layer_weights)[0]
    q_layer = rng.choice((1, 3)) if long else layer
    if query is None:
        query = sample_query(spec, facts, q_layer, rng) if spec else generic_query(facts, q_layer)
    kinds = _violation_sequence(cfg, rng)
    cfs = [(k, make_counterfactual(facts, k, rng, spec, keep_step=long)) for k in kinds]
    # ASSUMPTION: M3 entries carry no validity state, so counterfactuals of an M3 positive are rendered as M1 or M2.
    neg_layers = [layer if layer in (1, 2) else rng.choice((1, 2)) for _ in cfs]
    meta: Dict[str, Any] = {"generator": generator, "topic": spec.topic if spec else "", "entity": facts.entity,
                            "query_layer": q_layer}
    if long:
        # ASSUMPTION: log-uniform length in [min_ctx_tokens, max_ctx_tokens]; haystack on the document side.
        target = int(math.exp(rng.uniform(math.log(cfg.min_ctx_tokens), math.log(cfg.max_ctx_tokens))))
        variants = {"positive": facts, **{f"neg{j}": g for j, (_, g) in enumerate(cfs)}}
        hay, needle_idx, n_dis = build_haystack(
            variants, template, seeds, rng, target, cfg.max_ctx_tokens, counter, extra_distractors
        )
        positive = hay["positive"]
        negatives = [{"text": hay[f"neg{j}"], "violation": k, "layer": 1} for j, (k, _) in enumerate(cfs)]
        meta.update(context_tokens=counter.count(positive), needle_index=needle_idx, n_distractors=n_dis,
                    target_tokens=target)
    else:
        positive = {1: m1, 2: m2, 3: m3}[layer]
        negatives = [{"text": _render_at(template, g, nl), "violation": k, "layer": nl}
                     for nl, (k, g) in zip(neg_layers, cfs)]
    meta["positive_tokens"] = counter.count(positive)
    return {
        "id": f"synth-{idx:06d}",
        "kind": "long" if long else "short",
        "seed_id": f"{meta['topic'] or 'seed'}/{facts.entity}/{idx}",
        "layer": layer,
        "instruction": INSTRUCTIONS[layer],
        "query": query,
        "positive": positive,
        "negatives": negatives,
        "triplet": {"m1": m1, "m2": m2, "m3": m3},
        "meta": meta,
    }


def generate_offline_dataset(
    cfg: SynthConfig, seeds: Sequence[SeedSpec] = BUILTIN_SEEDS, counter: TokenCounter = DEFAULT_COUNTER
) -> List[Dict[str, Any]]:
    """Fully offline, deterministic given ``cfg.seed``."""
    rng = random.Random(cfg.seed)
    out = []
    for i in range(cfg.n):
        spec = rng.choice(seeds)
        out.append(build_record(i, spec, draw_facts(spec, rng), SeedTemplate(spec), cfg, rng, seeds, counter))
    return out


# --------------------------------------------------------------------------------------------- teacher-LLM generator
# ASSUMPTION: teacher prompt wording and the <<placeholder>> convention.
TEACHER_SYSTEM_PROMPT = (
    "You build training data for an agent-memory retriever. Given a seed (a document excerpt or a tool-use "
    "interaction), invent ONE plausible tool-augmented episode that could have produced the seed information and "
    "reply with a single JSON object and nothing else. Keys: "
    '"tool" (tool name), "entity", "attr", "value" (the fact the tool returned), "user" (the user request), '
    '"m1" (raw episodic trace: the tool call, the tool output as JSON and optional dialogue; use the literal '
    "placeholders <<ts>>, <<code>>, <<value>>, <<version>> and <<step>> in place of timestamp, return code, value, "
    'version and step), "m2" (procedural summary, at most 60 words, preserving the causal structure and containing '
    '<<value>>, <<step>> and <<version>>), "m3" (a thematic index key of about 8 tokens capturing the intent).'
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_teacher_messages(seed_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Seed:\n{seed_text.strip()}"},
    ]


def parse_teacher_reply(reply: str) -> Optional[Dict[str, Any]]:
    """Extract and validate the teacher's JSON object; ``None`` if unusable."""
    m = _JSON_RE.search(reply or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    for k in ("tool", "entity", "attr", "value", "m1", "m2", "m3"):
        if k not in d or d[k] in (None, ""):
            return None
    if not all(f"<<{p}>>" in str(d["m1"]) for p in ("ts", "code", "value")):
        return None
    if "<<value>>" not in str(d["m2"]):
        return None
    return d


def synthesize_triplet_with_teacher(
    client: ChatClient, seed_text: str, max_retries: int = 2, **gen: Any
) -> Optional[Tuple[Facts, TeacherTemplate]]:
    """Ask the teacher for one M1/M2/M3 triplet; retries on unparsable output, returns ``None`` if all fail."""
    for _ in range(1 + max_retries):
        d = parse_teacher_reply(client.chat(build_teacher_messages(seed_text), **gen))
        if d is None:
            continue
        facts = Facts(
            tool=str(d["tool"]), entity=str(d["entity"]), attr=str(d["attr"]), value=str(d["value"]), code=200,
            ts=NOW - timedelta(hours=3), version="v5", step=500, user=str(d.get("user", "")),
        )
        return facts, TeacherTemplate(str(d["m1"]), str(d["m2"]), str(d["m3"]))
    return None


def generate_teacher_dataset(
    client: ChatClient,
    seed_texts: Sequence[str],
    cfg: SynthConfig,
    counter: TokenCounter = DEFAULT_COUNTER,
    **gen: Any,
) -> Tuple[List[Dict[str, Any]], int]:
    """Teacher-written triplets for each seed; positive/negative validity fields are then rendered programmatically.

    Returns ``(records, n_failed)``. Long-horizon distractors come from the built-in seeds plus the other teacher
    triplets' valid M1 traces.
    """
    rng = random.Random(cfg.seed)
    built: List[Tuple[Facts, TeacherTemplate]] = []
    failed = 0
    for text in seed_texts:
        got = synthesize_triplet_with_teacher(client, text, **gen)
        if got is None:
            failed += 1
            continue
        f, t = got
        f.ts = NOW - timedelta(minutes=rng.randint(60, 20 * 60))
        f.version = f"v{rng.randint(3, 9)}"
        f.step = rng.randint(300, 900)
        built.append((f, t))
    pool = [t.m1(f) for f, t in built]
    records = []
    for i, (f, t) in enumerate(built[: cfg.n] if cfg.n else built):
        others = [p for j, p in enumerate(pool) if j != i]
        records.append(build_record(i, None, f, t, cfg, rng, BUILTIN_SEEDS, counter, extra_distractors=others,
                                    generator="teacher_llm"))
    return records, failed


# --------------------------------------------------------------------------------------------- FineWeb-Edu seeds
# ASSUMPTION: regex heuristic for "procedural / tool-use structure" seeds (the paper gives no operational definition).
_PROC_RE = re.compile(r"(\bstep\s*\d+|\bfirst\b|\bthen\b|\bnext\b|\bfinally\b|^\s*\d+[.)]\s|```|\bapi\b|\bcommand\b)",
                      re.IGNORECASE | re.MULTILINE)


def looks_procedural(text: str, min_markers: int = 3) -> bool:
    """Heuristic for "multi-step reasoning or tool-use structure rather than factual recitation" (ASSUMPTION)."""
    return len(_PROC_RE.findall(text)) >= min_markers


def load_fineweb_edu_seeds(
    n: int,
    min_score: float = 3.0,
    config_name: str = "sample-10BT",
    prefer_procedural: bool = True,
    max_chars: int = 2000,
    scan_limit: int = 200_000,
) -> List[str]:
    """Stream FineWeb-Edu documents with educational score >= ``min_score`` (paper: 3.0).

    UNVERIFIED in the offline test suite: needs the ``datasets`` package and network access, neither available here. The dataset
    id, the ``sample-10BT`` config name and the ``score`` / ``text`` column names follow the public dataset
    card and must be checked before use.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets (and network access) to load FineWeb-Edu seeds") from e
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=config_name, split="train", streaming=True)
    out: List[str] = []
    for i, row in enumerate(ds):
        if i >= scan_limit or len(out) >= n:
            break
        if float(row.get("score", 0.0)) < min_score:
            continue
        text = str(row.get("text", ""))[:max_chars]
        if prefer_procedural and not looks_procedural(text):
            continue
        out.append(text)
    return out


# --------------------------------------------------------------------------------------------- io and stats
def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def dataset_stats(records: Sequence[Dict[str, Any]], counter: TokenCounter = DEFAULT_COUNTER) -> Dict[str, Any]:
    """Counts and token statistics measured on ``records`` (no paper numbers involved)."""

    def mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    short = [r for r in records if r["kind"] == "short"]
    long = [r for r in records if r["kind"] == "long"]
    ctx = [r["meta"]["context_tokens"] for r in long]
    viol: Dict[str, int] = {}
    for r in records:
        for n in r["negatives"]:
            viol[n["violation"]] = viol.get(n["violation"], 0) + 1
    return {
        "n_records": len(records),
        "n_short": len(short),
        "n_long": len(long),
        "positive_layer_counts": {str(l): sum(1 for r in records if r["layer"] == l) for l in (1, 2, 3)},
        "violation_counts": viol,
        "mean_tokens": {k: mean([counter.count(r["triplet"][k]) for r in records]) for k in ("m1", "m2", "m3")},
        "long_context_tokens": {"min": min(ctx) if ctx else 0, "max": max(ctx) if ctx else 0, "mean": mean(ctx)},
        "negatives_per_query": sorted({len(r["negatives"]) for r in records}),
    }


# --------------------------------------------------------------------------------------------- controller data
_RECALL_PHRASES = (
    "stock level of widget_12", "email of customer_7", "build of svc_auth", "status of parcel_88",
    "row count of orders", "start time of standup",
)
_NOISE_OBS = ("page loaded", "ok", "cursor moved", "tab switched", "scrolled down")


def controller_prompt(state: ControllerState, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Exactly the prompt ``helm.controller.llm.HFController`` builds (system prompt + rendered state + newline)."""
    return system_prompt + "\n\n" + render_state(state) + "\n"


def wrap_controller_prompt(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Put a bare ``render_state`` prompt into the ``HFController`` framing (no-op if it already carries the system prompt)."""
    return prompt if prompt.startswith(system_prompt) else system_prompt + "\n\n" + prompt.rstrip("\n") + "\n"


def state_to_dict(s: ControllerState) -> Dict[str, Any]:
    d = {fl.name: getattr(s, fl.name) for fl in fields(ControllerState)}
    d["obs_status"] = s.obs_status.value
    d["sizes"] = {str(k): v for k, v in s.sizes.items()}
    d["budgets"] = {str(k): v for k, v in s.budgets.items()}
    return d


def state_from_dict(d: Dict[str, Any]) -> ControllerState:
    d = dict(d)
    d["obs_status"] = Status(d["obs_status"])
    d["sizes"] = {int(k): v for k, v in d.get("sizes", {}).items()}
    d["budgets"] = {int(k): v for k, v in d.get("budgets", {}).items()}
    return ControllerState(**d)


def random_controller_state(rng: random.Random, recall_prob: float = 0.35) -> ControllerState:
    budgets = {1: 200, 2: 50, 3: 20}
    status = rng.choices((Status.SUCCESS, Status.FAILED, Status.UNKNOWN), weights=(0.6, 0.25, 0.15))[0]
    has_obs = rng.random() < 0.9
    claimy = has_obs and rng.random() < 0.6
    if claimy:
        ent = f"{rng.choice(('widget', 'customer', 'parcel'))}_{rng.randint(1, 99)}"
        code = {Status.SUCCESS: 200, Status.FAILED: 500, Status.UNKNOWN: 204}[status]
        obs = f'{{"entity": "{ent}", "attr": "stock", "value": {rng.randint(0, 500)}, "code": {code}}}'
        tool = "inventory_api.get_stock"
    elif has_obs:
        obs, tool = rng.choice(_NOISE_OBS), "browser.click"
    else:
        obs, tool = "", ""
    near_full = rng.random() < 0.2  # all probabilities here are arbitrary smoke-data choices
    sizes = {
        1: rng.randint(int(0.9 * budgets[1]), budgets[1]) if near_full else rng.randint(0, int(0.8 * budgets[1])),
        2: rng.randint(0, budgets[2]),
        3: rng.randint(0, budgets[3] - 1),
    }
    return ControllerState(
        step=rng.randint(1, 400),
        obs_tool=tool,
        obs_status=status if has_obs else Status.UNKNOWN,
        obs_text=obs,
        obs_has_claims=claimy and status != Status.FAILED,
        version_changed=rng.random() < 0.1,
        recall_request=rng.choice(_RECALL_PHRASES) if rng.random() < recall_prob else None,
        sizes=sizes,
        budgets=budgets,
        consolidations=rng.randint(0, 5),
        max_consolidations=5,
        steps_since_consolidate=rng.randint(0, 30),
    )


def make_synthetic_controller_dataset(
    n: int, seed: int = 0, recall_prob: float = 0.35, system_prompt: str = SYSTEM_PROMPT
) -> List[Dict[str, Any]]:
    """Random controller prompts with the ``HeuristicController`` decision as the completion.

    Records: ``{"prompt", "completion", "state"}``. Does not depend on ``helm.sim``; the prompts are NOT drawn from real
    agent trajectories, so results on them say nothing about the real controller.
    """
    rng = random.Random(seed)
    teacher = HeuristicController()
    out = []
    for _ in range(n):
        s = random_controller_state(rng, recall_prob)
        out.append(
            {
                "prompt": controller_prompt(s, system_prompt),
                "completion": format_ops(teacher.decide(s)),
                "state": state_to_dict(s),
            }
        )
    return out
