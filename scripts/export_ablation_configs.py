#!/usr/bin/env python3
"""Write configs/ablations/<variant>.yaml from the method registry (single source of truth: helm/methods.py)."""
from pathlib import Path

import yaml
from _common import ROOT  # noqa: F401
from helm.methods import _HELM_VARIANTS, variant_overrides


def nest(dotted):
    out = {}
    for k, v in dotted.items():
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def main() -> None:
    out = ROOT / "configs" / "ablations"
    out.mkdir(parents=True, exist_ok=True)
    for name, (label, _, gen) in _HELM_VARIANTS.items():
        body = nest(variant_overrides(name))
        header = f"# {label}  (memory-generator variant: {gen})\n"
        (out / f"{name}.yaml").write_text(header + (yaml.safe_dump(body, sort_keys=False) if body else "{}\n"))
    print(f"wrote {len(_HELM_VARIANTS)} files to {out}")


if __name__ == "__main__":
    main()
