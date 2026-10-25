"""Test helpers: make `helm` importable from tests/ and build observations quickly."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helm.config import HelmConfig  # noqa: E402
from helm.embedding import HashingEmbedder  # noqa: E402
from helm.memory import SHNM  # noqa: E402
from helm.types import Observation, ReadContext, Status  # noqa: E402


def mkobs(step, entity, attr, value, code=200, ver="v1", tool="lookup"):
    text = json.dumps({"entity": entity, "attr": attr, "value": value, "code": code, "version": ver})
    return Observation(step, float(step), tool, f"L{step}", text, Status.from_code(code), task_version=ver)


def mkstore(cfg=None, **overrides):
    cfg = (cfg or HelmConfig()).override(**overrides) if overrides else (cfg or HelmConfig())
    return SHNM(cfg, HashingEmbedder(cfg.embedding.dim, tool_aware=cfg.embedding.tool_aware))


def ctx(step, ver="v1"):
    return ReadContext(step=step, time=float(step), task_version=ver)
