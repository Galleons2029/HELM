"""Synthetic long-horizon world with *epistemic traps*.

The world is a set of (entity, attribute) keys whose true values change over time. An agent learns about them by calling a
`lookup` tool. Three kinds of stored traces become semantically-relevant-but-invalid, which is exactly the failure mode
the paper targets:

* STALE          the value was true when observed, but the environment version bumped or the key drifted since;
* FAILED         the tool returned an error code yet its payload still carries a plausible (wrong) value;
* CONTRADICTORY  two successful observations of the same key disagree (an older one is superseded by a newer one).

Everything is generated from an integer seed, so every memory method faces the *identical* episode.
THESE ARE SYNTHETIC EPISODES: results on them say how a memory mechanism behaves on a controlled trap structure, not how
HELM performs on GAIA/WebArena/etc., and must never be mixed with the paper's benchmark numbers.
"""

from __future__ import annotations

import bisect
import json
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..types import Status

Key = Tuple[str, str]

NOUNS = ["widget", "gadget", "sprocket", "bracket", "valve", "relay", "sensor", "bearing", "gasket", "filter",
         "rotor", "coupler", "piston", "spindle", "nozzle", "flange"]
STATUS_WORDS = ["ok", "degraded", "down", "maintenance"]
_LOG_VOCAB = ("trace span retry backoff shard queue replica gateway cache latency handshake payload checksum token "
              "batch index flush lease quorum buffer socket thread mutex region zone cluster node pod mount volume "
              "rotate expire renew audit policy route proxy tunnel sync merge commit rollback snapshot compact").split()


@dataclass
class SimConfig:
    n_entities: int = 30
    attrs: Tuple[str, ...] = ("stock", "price", "status")
    horizon: int = 128
    horizon_std: int = 24
    n_queries: int = 8
    hot_keys: int = 16
    p_hot: float = 0.85  # share of lookups (and drifts) that hit a hot key
    p_lookup: float = 0.55  # chance a non-query step is a lookup (else a distractor log)
    p_fail: float = 0.20
    p_unknown: float = 0.05
    version_bumps: int = 2
    bump_change_frac: float = 0.4
    p_silent_drift: float = 0.03  # per step: a hot key changes without a version bump (not detectable by version)
    query_trap_bias: float = 0.7
    query_extra_tokens: int = 0  # filler words appended to every recall query ("query verbosity")
    log_tokens: int = 40  # raw-log verbosity: filler words in every tool trace (paper premise: M1 traces are long raw logs)
    seed_offset: int = 0


@dataclass
class LookupResult:
    code: int
    payload: Dict[str, object]

    @property
    def text(self) -> str:
        return json.dumps(self.payload)

    @property
    def status(self) -> Status:
        return Status.from_code(self.code)


@dataclass
class Event:
    step: int
    kind: str  # "lookup" | "distract" | "query"
    key: Optional[Key] = None
    text: str = ""  # distractor log text
    result: Optional[LookupResult] = None
    version: str = "v1"


def _rng(*parts: object) -> random.Random:
    return random.Random(zlib.crc32(repr(parts).encode("utf-8")))


class Episode:
    """One generated episode: a deterministic event stream plus the ground-truth world timeline."""

    def __init__(self, cfg: SimConfig, seed: int):
        self.cfg, self.seed = cfg, seed
        rng = _rng("episode", seed, cfg.seed_offset)
        self.horizon = max(24, int(rng.gauss(cfg.horizon, cfg.horizon_std)))
        ents = [f"{NOUNS[i % len(NOUNS)]}-{i + 1}" for i in range(cfg.n_entities)]
        self.keys: List[Key] = [(e, a) for e in ents for a in cfg.attrs]
        rng.shuffle(self.keys)
        self.hot = self.keys[: cfg.hot_keys]
        self.cold = self.keys[cfg.hot_keys :]
        self._values: Dict[Key, List[Tuple[int, object]]] = {k: [(-1, self._draw(k, rng))] for k in self.keys}
        self._versions: List[Tuple[int, str]] = [(-1, "v1")]
        self.bump_steps = sorted(rng.sample(range(int(0.25 * self.horizon), int(0.85 * self.horizon)), cfg.version_bumps)) \
            if cfg.version_bumps and self.horizon > 8 else []
        q_lo = int(0.35 * self.horizon)
        self.query_steps = sorted(rng.sample(range(q_lo, self.horizon), min(cfg.n_queries, self.horizon - q_lo)))
        self.events: List[Event] = self._build(rng)

    # ------------------------------------------------------------------ truth
    @staticmethod
    def _draw(key: Key, rng: random.Random) -> object:
        if key[1] == "status":
            return rng.choice(STATUS_WORDS)
        if key[1] == "price":
            return rng.randint(1, 500)
        return rng.randint(0, 200)

    def _change(self, key: Key, step: int, rng: random.Random) -> None:
        old = self._values[key][-1][1]
        new = self._draw(key, rng)
        while new == old:
            new = self._draw(key, rng)
        self._values[key].append((step, new))

    def value_at(self, key: Key, step: int) -> object:
        hist = self._values[key]
        i = bisect.bisect_right([s for s, _ in hist], step) - 1
        return hist[max(i, 0)][1]

    def previous_value(self, key: Key, step: int) -> Optional[object]:
        hist = [v for s, v in self._values[key] if s <= step]
        return hist[-2] if len(hist) >= 2 else None

    def version_at(self, step: int) -> str:
        i = bisect.bisect_right([s for s, _ in self._versions], step) - 1
        return self._versions[max(i, 0)][1]

    # ------------------------------------------------------------------ tools
    def lookup(self, key: Key, step: int, attempt: int = 0) -> LookupResult:
        """Deterministic tool call outcome for (key, step, attempt)."""
        r = _rng("lookup", self.seed, self.cfg.seed_offset, key, step, attempt)
        truth = self.value_at(key, step)
        ver = self.version_at(step)
        base = {"entity": key[0], "attr": key[1], "version": ver}
        if self.cfg.log_tokens:
            base["log"] = " ".join(r.choice(_LOG_VOCAB) for _ in range(self.cfg.log_tokens))
        u = r.random()
        if u < self.cfg.p_fail:
            prev = self.previous_value(key, step)
            wrong = prev if (prev is not None and r.random() < 0.5) else self._draw(key, r)
            if wrong == truth:
                wrong = self._draw(key, r)
            code = r.choice([500, 503, 504])
            return LookupResult(code, {**base, "value": wrong, "code": code, "error": "upstream timeout; served cached value"})
        if u < self.cfg.p_fail + self.cfg.p_unknown:
            val = truth if r.random() < 0.8 else self._draw(key, r)
            return LookupResult(204, {**base, "value": val, "code": 204})
        return LookupResult(200, {**base, "value": truth, "code": 200})

    # ------------------------------------------------------------------ event stream
    def _pick_key(self, rng: random.Random) -> Key:
        return rng.choice(self.hot) if rng.random() < self.cfg.p_hot else rng.choice(self.cold or self.hot)

    def _distractor(self, rng: random.Random) -> str:
        e = rng.choice(self.keys)
        templates = [
            "Customer asked about {e} {a} again; see ticket {n}, no figures attached.",
            "search: {n} documents mention {e} {a} in shipping notes, none authoritative.",
            "cron: nightly sync for {e} finished in {n} ms, {a} not touched.",
            "forum thread {n} discusses how {e} {a} is usually reported, opinions differ.",
        ]
        text = rng.choice(templates).format(e=e[0], a=e[1], n=rng.randint(1000, 9999))
        if self.cfg.log_tokens:
            text += " log: " + " ".join(rng.choice(_LOG_VOCAB) for _ in range(self.cfg.log_tokens))
        return text

    def _build(self, rng: random.Random) -> List[Event]:
        events: List[Event] = []
        observed: Dict[Key, List[Tuple[int, object, Status]]] = {}
        pending_bump = list(self.bump_steps)
        ver_n = 1
        for t in range(self.horizon):
            if pending_bump and t == pending_bump[0]:
                pending_bump.pop(0)
                ver_n += 1
                self._versions.append((t, f"v{ver_n}"))
                for k in self.hot + rng.sample(self.cold, min(len(self.cold), 6)):
                    if rng.random() < self.cfg.bump_change_frac:
                        self._change(k, t, rng)
            elif rng.random() < self.cfg.p_silent_drift:
                self._change(rng.choice(self.hot), t, rng)
            ver = self.version_at(t)
            if t in self.query_steps:
                events.append(Event(t, "query", self._query_key(t, observed, rng), version=ver))
            elif rng.random() < self.cfg.p_lookup:
                key = self._pick_key(rng)
                res = self.lookup(key, t, 0)
                observed.setdefault(key, []).append((t, res.payload["value"], res.status))
                events.append(Event(t, "lookup", key, result=res, version=ver))
            else:
                events.append(Event(t, "distract", text=self._distractor(rng), version=ver))
        return events

    def _query_key(self, t: int, observed: Dict[Key, List[Tuple[int, object, Status]]], rng: random.Random) -> Key:
        if not observed:
            return self._pick_key(rng)
        traps = []
        for k, obs in observed.items():
            truth = self.value_at(k, t)
            ok = [(s, v) for s, v, st in obs if st == Status.SUCCESS]
            stale = bool(ok) and ok[-1][1] != truth
            failed = obs[-1][2] == Status.FAILED
            contra = len({v for _, v in ok}) > 1
            if stale or failed or contra:
                traps.append(k)
        pool = traps if (traps and rng.random() < self.cfg.query_trap_bias) else list(observed)
        return rng.choice(pool)
