"""Configuration for HELM. Defaults mirror the paper's reproducibility table (Appendix, ``tab:repro_hparams``).

Every value the paper does NOT specify is marked ``ASSUMPTION`` below and listed in README.md.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class GovernanceConfig:
    """Epistemic governance (Eq. 1 and the validity conditions)."""

    enabled: bool = True  # False => "-Gov": pure similarity ranking, no validity filtering.
    # Fixed weights (paper: lambda_sim=1.0, lambda_time=1.0, lambda_status=10.0, lambda_src=1.0).
    lambda_sim: float = 1.0
    lambda_time: float = 1.0
    lambda_status: float = 10.0
    lambda_src: float = 1.0
    # Per-signal switches for the leave-one-out ablation.
    use_status: bool = True
    use_freshness: bool = True
    use_consistency: bool = True
    use_source: bool = True
    # Staleness window (paper: 200 steps, 7 days, plus snapshot/version check).
    step_window: int = 200
    time_window_days: float = 7.0
    check_version: bool = True
    # Hard filter (Alg. 1 subtracts infinity from invalid items) with soft fallback (readmit if nothing valid).
    hard_filter: bool = True
    fallback_readmit: bool = True
    fallback_k: int = 2  # ASSUMPTION: at most this many invalid items are readmitted (flagged) per layer search.
    # ASSUMPTION: recency kernel g(dt) = exp(-dt / recency_scale), dt in steps. Paper: "monotone recency kernel".
    recency_scale: float = 50.0
    # ASSUMPTION: w(status) / w(src) value tables (paper does not give them).
    status_weights: Dict[str, float] = field(
        default_factory=lambda: {"SUCCESS": 0.0, "UNKNOWN": -0.1, "FAILED": -1.0}
    )
    source_weights: Dict[str, float] = field(
        default_factory=lambda: {"tool": 1.0, "user": 0.8, "summary": 0.6, "index": 0.4, "reasoner": 0.2}
    )
    # Diagnostic-exemption variant: suspend the status penalty when the query is itself diagnostic.
    diagnostic_exemption: bool = False
    diagnostic_keywords: tuple = ("error", "fail", "failed", "why", "debug", "traceback", "exception")


@dataclass
class MemoryConfig:
    """SHNM store settings."""

    budget_m1: int = 200
    budget_m2: int = 50
    budget_m3: int = 20
    k1: int = 5
    k2: int = 5
    k3: int = 3
    pool_mult: int = 4  # ASSUMPTION: candidate pool c_l = pool_mult * k_l (paper: "pool=c_l", value not given).
    index_kind: str = "hnsw"  # hnsw | flat | hnswlib
    hnsw_M: int = 16
    ef_construction: int = 200  # ASSUMPTION: not given in paper.
    ef_search: int = 64
    max_recall_tokens: int = 1500
    max_writes_per_step: int = 1
    max_consolidations_per_episode: int = 5
    # Traversal: "coarse_to_fine" (text of Sec. 3.2: M3 -> M2 -> grounding in M1) or
    # "independent" (literal Alg. 1: query each layer, TopK per layer).
    traversal: str = "coarse_to_fine"
    ground_top_n: int = 2  # ASSUMPTION: number of top M2 recalls expanded to supporting M1 spans.
    prov_depth: int = 2  # max depth D for ProvenanceExpansion.
    conflict_tau: float = 0.5  # ASSUMPTION: Alg. 3 score-gap threshold tau.
    keep_failed_per_key: int = 1  # Prune keeps the minimal failed evidence per key.
    consolidate_every: int = 16  # heuristic controller schedule.
    # Layer / provenance ablations.
    use_m2: bool = True
    use_m3: bool = True
    ground_to_m1: bool = True
    provenance: bool = True  # False => "-Prov": M2/M3 cannot inherit epistemic state from supporting traces.


@dataclass
class EmbeddingConfig:
    backend: str = "hashing"  # hashing | sentence_transformers
    dim: int = 512  # hashing backend only; real models report their own dim.
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    tool_aware: bool = True  # hashing backend: expose structured tool fields (proxy for the tool-aware fine-tune)
    seed: int = 0


@dataclass
class ReasonerConfig:
    model: str = "Qwen/Qwen3-8B"
    context_window: int = 128_000
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class ControllerConfig:
    kind: str = "heuristic"  # heuristic | rule | llm | learned
    model: str = "Qwen/Qwen3-4B"
    alpha: float = 1.0  # reward weight on r_task
    beta: float = 0.05  # reward weight on r_mem


@dataclass
class HelmConfig:
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    seeds: tuple = (0, 42, 123)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HelmConfig":
        cfg = cls()
        _merge(cfg, d or {})
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, *extra: str | Path) -> "HelmConfig":
        """Load ``path`` and then overlay each of ``extra`` (e.g. an ablation file) on top."""
        cfg = cls()
        for p in (path, *extra):
            with open(p, "r", encoding="utf-8") as f:
                _merge(cfg, yaml.safe_load(f) or {})
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def override(self, **dotted: Any) -> "HelmConfig":
        """Return a copy with dotted-path overrides, e.g. ``cfg.override(**{"governance.lambda_status": 0.0})``."""
        new = copy.deepcopy(self)
        for path, value in dotted.items():
            obj = new
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise AttributeError(f"unknown config field: {path}")
            setattr(obj, parts[-1], value)
        return new


def _merge(obj: Any, d: Dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key '{k}' for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        elif isinstance(cur, dict) and isinstance(v, dict):
            cur.update(v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)
