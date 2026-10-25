"""Real-benchmark adapters on top of the unified agent scaffold (``harness.py``).

Adapters are imported lazily (``get_adapter_module``) so that importing the package costs nothing and an adapter with a
missing optional dependency only fails when it is actually selected.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

BENCHMARKS: Dict[str, str] = {
    "gaia": "helm.benchmarks.gaia",
    "webarena": "helm.benchmarks.webarena",
    "mind2web": "helm.benchmarks.mind2web",
    "streambench": "helm.benchmarks.streambench",
    "lifelong": "helm.benchmarks.lifelong",
    "bright": "helm.benchmarks.bright",
    "qa": "helm.benchmarks.qa_sanity",
}

# Benchmarks that run an LLM agent loop (the others are retrieval / single-call evaluations).
RETRIEVAL_ONLY = ("bright",)


def get_adapter_module(name: str) -> Any:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}'. choose from {sorted(BENCHMARKS)}")
    return importlib.import_module(BENCHMARKS[name])
