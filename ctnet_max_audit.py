#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoría de invariantes CTNet 3.0 MAX."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import torch

from ctnet_max_profiles import get_profile
from ctnet_omega_cubo6d_plegado_ctnet26 import FoldLayout, FoldedCTNetOmegaCubo26, count_params
from ctnet_omega_mthd_integrated import CTNetOmegaMTHD26


def layout_from_profile(profile_name: str) -> FoldLayout:
    p = get_profile(profile_name)
    return FoldLayout(
        N=p.N,
        d=p.d,
        z_tokens=p.z_tokens,
        z_dim=p.z_dim,
        mem_slots=p.mem_slots,
        mem_dim=p.mem_dim,
        rel_edges=p.rel_edges,
        rel_dim=p.rel_dim,
    )


def audit_profile(profile_name: str, *, batch: int | None = None, steps: int = 1, fp64: bool = False, cuda: bool = False) -> Dict[str, Any]:
    p = get_profile(profile_name)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if fp64 else torch.float32
    layout = layout_from_profile(profile_name)
    model = FoldedCTNetOmegaCubo26(
        layout=layout,
        fractal_steps=p.fractal_steps,
        latent_steps=p.latent_steps,
        cubo_shear=p.cubo_shear,
    ).to(device=device, dtype=dtype)
    audit = model.audit(batch=batch or p.batch, dtype=dtype, device=device, steps=steps, seed=0)
    audit.update(
        profile=p.to_dict(),
        device=str(device),
        dtype=str(dtype),
        params=count_params(model),
        reversible_ok=bool(audit["packed_mae"] < (1e-5 if not fp64 else 1e-10)),
        layout_ok=bool(layout.pad_size >= 0 and layout.semantic_size + layout.pad_size == layout.capacity),
    )
    return audit


def audit_mthd(profile_name: str, *, batch: int | None = None, steps: int = 1, fp64: bool = False, cuda: bool = False) -> Dict[str, Any]:
    p = get_profile(profile_name)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if fp64 else torch.float32
    model = CTNetOmegaMTHD26(
        layout=layout_from_profile(profile_name),
        fractal_steps=p.fractal_steps,
        latent_steps=p.latent_steps,
        cubo_shear=p.cubo_shear,
        mthd_omega_words=1024 if profile_name in {"xl", "max"} else 256,
    ).to(device=device, dtype=dtype)
    audit = model.audit_mthd(batch=batch or p.batch, dtype=dtype, device=device, seed=0, steps=steps)
    audit.update(profile=p.to_dict(), device=str(device), dtype=str(dtype), params=count_params(model))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CTNet MAX invariants")
    parser.add_argument("--profile", choices=["smoke", "base", "xl", "max"], default="smoke")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--mthd", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp64", action="store_true")
    args = parser.parse_args()
    fn = audit_mthd if args.mthd else audit_profile
    print(json.dumps(fn(args.profile, batch=args.batch, steps=args.steps, fp64=args.fp64, cuda=args.cuda), indent=2, default=str))


if __name__ == "__main__":
    main()
