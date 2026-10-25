"""Registry of memory methods: baselines, HELM, and every ablation, all built from one ``HelmConfig``.

Ablations are pure config overrides (dotted paths into ``HelmConfig``), so they are inspectable and also available as
YAML under ``configs/ablations``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .backends import HelmBackend, MemoryBackend
from .baselines import FlatRAGBackend, LongContextBackend, Mem0StyleBackend, MemGPTStyleBackend
from .config import HelmConfig
from .controller.base import Controller
from .controller.recall_interface import MemoryGenerator
from .embedding.base import build_embedder
from .llm.client import ChatClient

# name -> (display name, config overrides, generator variant)
_HELM_VARIANTS: Dict[str, Tuple[str, Dict[str, Any], str]] = {
    "helm": ("HELM (full)", {}, "A0"),
    # component / system ablations (Table: component and per-signal ablations)
    "helm_no_gov": ("HELM -Gov", {"governance.enabled": False}, "A0"),
    "helm_no_prov": ("HELM -Prov", {"memory.provenance": False}, "A0"),
    "helm_no_emb": ("HELM -Emb", {"embedding.tool_aware": False}, "A0"),
    "helm_no_ctrl": ("HELM -Ctrl", {"controller.kind": "rule"}, "A0"),
    # per-signal leave-one-out
    "helm_no_status": ("HELM -status", {"governance.use_status": False}, "A0"),
    "helm_no_freshness": ("HELM -freshness", {"governance.use_freshness": False}, "A0"),
    "helm_no_consistency": ("HELM -consistency", {"governance.use_consistency": False}, "A0"),
    # layer ablations
    "helm_m1_only": ("HELM M1-only", {"memory.use_m2": False, "memory.use_m3": False}, "A0"),
    "helm_no_m3": ("HELM -M3", {"memory.use_m3": False}, "A0"),
    "helm_no_grounding": ("HELM no-grounding", {"memory.ground_to_m1": False}, "A0"),
    # literal Alg. 1 traversal
    "helm_independent": ("HELM independent-traversal", {"memory.traversal": "independent"}, "A0"),
    # faithfulness controls
    "helm_A1": ("HELM A1 (cited)", {}, "A1"),
    "helm_A2": ("HELM A2 (extractive-first)", {}, "A2"),
    # in-harness comparison systems built from the same store
    "helm_lite": (
        "HELM-lite",
        {"memory.use_m2": False, "memory.use_m3": False, "governance.use_consistency": False},
        "A0",
    ),
    "archrag_style": (
        "ArchRAG-style",
        {"governance.enabled": False, "memory.provenance": False},
        "A0",
    ),
}

OFF_THE_SHELF = {"long_context", "flat_rag", "memgpt_style", "mem0_style", "archrag_style"}
BASELINES = ["long_context", "flat_rag", "memgpt_style", "mem0_style", "archrag_style", "helm_lite"]
ABLATIONS = [
    "helm_no_gov", "helm_no_prov", "helm_no_emb", "helm_no_ctrl",
    "helm_no_status", "helm_no_freshness", "helm_no_consistency",
    "helm_m1_only", "helm_no_m3", "helm_no_grounding", "helm_independent",
]
FAITHFULNESS = ["helm", "helm_A1", "helm_A2"]


def variant_overrides(name: str) -> Dict[str, Any]:
    """Dotted-path config overrides that define a HELM variant (used to export ``configs/ablations/*.yaml``)."""
    return dict(_HELM_VARIANTS[name][1])


def variant_generator(name: str) -> str:
    return _HELM_VARIANTS[name][2]


def available() -> List[str]:
    return ["long_context", "flat_rag", "memgpt_style", "mem0_style", *_HELM_VARIANTS]


def build_method(
    name: str,
    cfg: Optional[HelmConfig] = None,
    seed: int = 0,
    controller: Optional[Controller] = None,
    llm: Optional[ChatClient] = None,
) -> MemoryBackend:
    cfg = cfg or HelmConfig()
    # Comparison systems use the shared *off-the-shelf* retriever (paper: "every memory-augmented method ... same
    # encoder"); a "+emb" suffix gives them HELM's tool-aware embedding, which fills the 2x2 retriever comparison.
    # HELM variants keep ``cfg.embedding.tool_aware`` (True by default = the fine-tuned encoder's proxy).
    with_emb = name.endswith("+emb")
    name = name[: -len("+emb")] if with_emb else name
    if name in OFF_THE_SHELF:
        cfg = cfg.override(**{"embedding.tool_aware": with_emb})
    elif with_emb:
        cfg = cfg.override(**{"embedding.tool_aware": True})
    if name in _HELM_VARIANTS:
        label, overrides, variant = _HELM_VARIANTS[name]
        c = cfg.override(**overrides) if overrides else cfg
        emb = build_embedder(c.embedding)
        return HelmBackend(
            c, emb, controller=controller, generator=MemoryGenerator(variant, client=llm),
            name=label + (" + tool-aware emb" if with_emb and name in OFF_THE_SHELF else ""), seed=seed,
        )
    emb = build_embedder(cfg.embedding)
    ctors = {
        "flat_rag": lambda: FlatRAGBackend(cfg, emb),
        "long_context": lambda: LongContextBackend(cfg),
        "memgpt_style": lambda: MemGPTStyleBackend(cfg, emb),
        "mem0_style": lambda: Mem0StyleBackend(cfg, emb),
    }
    if name in ctors:
        be = ctors[name]()
        if with_emb:
            be.name += " + tool-aware emb"
        return be
    raise KeyError(f"unknown method '{name}'. available: {available()}")
