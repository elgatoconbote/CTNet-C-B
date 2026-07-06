#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark mínimo de memoria funcional por reactividad para CTNet MAX."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from ctnet_max_profiles import get_profile
from ctnet_max_audit import layout_from_profile
from ctnet_omega_cubo6d_plegado_ctnet26 import FoldedCTNetOmegaCubo26, count_params
from train_vram_up_coherence_ctnet import Observador, all_perspective_up_loss, batch_to_state


def _distance_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1).to(torch.float32)
    return torch.cdist(x, x)


def _functional_spread(x: torch.Tensor) -> float:
    d = _distance_matrix(x)
    if d.numel() <= 1:
        return 0.0
    return float(d[d > 0].mean().detach().cpu()) if (d > 0).any() else 0.0


def benchmark(profile_name: str, *, steps: int, batch: int, seed: int, cuda: bool) -> Dict:
    torch.manual_seed(seed)
    p = get_profile(profile_name)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model = FoldedCTNetOmegaCubo26(
        layout=layout_from_profile(profile_name),
        fractal_steps=p.fractal_steps,
        latent_steps=p.latent_steps,
        cubo_shear=p.cubo_shear,
    ).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)

    samples: List[Observador] = [
        Observador(x=f"family affine rule input={i} output={2*i+3}", y=f"{2*i+3}", source="ctnet://functional_benchmark", regime="affine_family")
        for i in range(max(batch, 8))
    ]
    probe_samples = samples[:batch]

    def encode_forward(rows: List[Observador]) -> torch.Tensor:
        st, _, _ = batch_to_state(model, rows, device=device, dtype=dtype, max_bytes=p.max_bytes)
        return model.pack(model.forward_state(st))

    with torch.no_grad():
        before = encode_forward(probe_samples)
        before_spread = _functional_spread(before)

    last_loss = None
    for step in range(steps):
        rows = [samples[(step * batch + j) % len(samples)] for j in range(batch)]
        state, target_z, _ = batch_to_state(model, rows, device=device, dtype=dtype, max_bytes=p.max_bytes)
        out = model.forward_state(state)
        xi = model.pack(out)
        loss_up, _ = all_perspective_up_loss(model, state, out)
        loss_anchor = F.mse_loss(out.z, target_z)
        obs = model.cubo_observation(out)
        loss = loss_up + p.lambda_anchor * loss_anchor + p.lambda_omega * obs["omega"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        after = encode_forward(probe_samples)
        after_spread = _functional_spread(after)
        delta = float((after - before).abs().mean().detach().cpu())

    return {
        "profile": profile_name,
        "device": str(device),
        "params": count_params(model),
        "steps": steps,
        "batch": batch,
        "loss_last": last_loss,
        "functional_spread_before": before_spread,
        "functional_spread_after": after_spread,
        "mean_response_delta": delta,
        "reactivity_changed": bool(delta > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet functional reactivity benchmark")
    parser.add_argument("--profile", choices=["smoke", "base", "xl", "max"], default="smoke")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.profile, steps=args.steps, batch=args.batch, seed=args.seed, cuda=args.cuda), indent=2))


if __name__ == "__main__":
    main()
