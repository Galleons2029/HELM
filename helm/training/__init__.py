"""Training pipelines: data synthesis, tool-aware embedding fine-tuning, controller Stage 1 (OPD) and Stage 2 (GSPO).

Submodules are imported explicitly (``from helm.training import data_synthesis``); nothing heavy is imported here, so the
pure-Python parts (data synthesis, rewards) work without torch/transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

SOURCES = ("smoke", "measured")


def write_metrics(path: str | Path, source: str, **payload: Any) -> Path:
    """Write a metrics JSON whose first key is ``"source"`` ("smoke" for tiny CPU runs, "measured" for real runs)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"source": source, **payload}, indent=2, default=str))
    return p


def thirds(values: Sequence[float]) -> Tuple[float, float]:
    """Mean of the first and of the last third of ``values`` (smoothed start/end comparison)."""
    n = max(1, len(values) // 3)
    return sum(values[:n]) / n, sum(values[-n:]) / n


def dtype_kwargs(dtype: Any) -> Dict[str, Any]:
    """``from_pretrained`` keyword for the dtype: ``dtype`` on recent transformers, ``torch_dtype`` on older ones.

    The 4.56 threshold is approximate (``dtype`` is verified to work on 5.9.0); on versions in between pass it yourself.
    """
    import transformers
    from packaging.version import Version

    return {("dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"): dtype}


def resolve_device(pref: str = "auto") -> str:
    """``auto`` -> cuda, then mps, then cpu."""
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_config(title: str, cfg: Dict[str, Any], note: Optional[str] = None) -> None:
    """Print the fully resolved configuration (scripts never train with settings the user cannot see)."""
    print(f"== {title}")
    if note:
        print(note)
    print(json.dumps(cfg, indent=2, default=str))
