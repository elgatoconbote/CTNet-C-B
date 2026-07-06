#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet 3.0 MAX profiles.

A profile is not only a bigger tensor. It fixes the complete state geometry:
Xi=[B,N,d], Z, M, R, C6 and pad. The invariant is:

    z_size + memory_size + relations_size + 29 + pad_size == N*d

The MAX profile is intentionally a large-state configuration. Use smoke/base for CPU
checks, xl/max for serious GPU runs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

CUBO_SIZE = 29


@dataclass(frozen=True)
class CTNetProfile:
    name: str
    N: int
    d: int
    z_tokens: int
    z_dim: int
    mem_slots: int
    mem_dim: int
    rel_edges: int
    rel_dim: int
    fractal_steps: int
    latent_steps: int
    cubo_shear: float = 0.05
    max_bytes: int = 2048
    batch: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    lambda_up: float = 1.0
    lambda_anchor: float = 0.10
    lambda_coh: float = 0.05
    lambda_omega: float = 0.25
    lambda_cubo: float = 0.05
    lambda_structure: float = 0.10
    lambda_rev: float = 0.10
    lambda_self_observation: float = 0.25
    lambda_efector: float = 0.25
    reversibility_loss_every: int = 10
    self_observation_every: int = 1
    efector_every: int = 1
    min_slot_var: float = 1e-8

    @property
    def capacity(self) -> int:
        return self.N * self.d

    @property
    def z_size(self) -> int:
        return self.z_tokens * self.z_dim

    @property
    def memory_size(self) -> int:
        return self.mem_slots * self.mem_dim

    @property
    def relations_size(self) -> int:
        return self.rel_edges * self.rel_dim

    @property
    def semantic_size(self) -> int:
        return self.z_size + self.memory_size + self.relations_size + CUBO_SIZE

    @property
    def pad_size(self) -> int:
        return self.capacity - self.semantic_size

    def validate(self) -> None:
        if self.d % 2 != 0:
            raise ValueError(f"profile {self.name}: d must be even")
        if self.z_dim != self.d or self.mem_dim != self.d or self.rel_dim != self.d:
            raise ValueError(f"profile {self.name}: z_dim/mem_dim/rel_dim must equal d")
        if self.pad_size < 0:
            raise ValueError(
                f"profile {self.name}: semantic_size={self.semantic_size} exceeds capacity={self.capacity}"
            )
        if min(self.N, self.d, self.z_tokens, self.mem_slots, self.rel_edges) <= 0:
            raise ValueError(f"profile {self.name}: dimensions must be positive")

    def as_train_argv(self) -> List[str]:
        self.validate()
        pairs = {
            "N": self.N,
            "d": self.d,
            "z-tokens": self.z_tokens,
            "z-dim": self.z_dim,
            "mem-slots": self.mem_slots,
            "mem-dim": self.mem_dim,
            "rel-edges": self.rel_edges,
            "rel-dim": self.rel_dim,
            "fractal-steps": self.fractal_steps,
            "latent-steps": self.latent_steps,
            "cubo-shear": self.cubo_shear,
            "max-bytes": self.max_bytes,
            "batch": self.batch,
            "lr": self.lr,
            "weight-decay": self.weight_decay,
            "grad-clip": self.grad_clip,
            "lambda-up": self.lambda_up,
            "lambda-anchor": self.lambda_anchor,
            "lambda-coh": self.lambda_coh,
            "lambda-omega": self.lambda_omega,
            "lambda-cubo": self.lambda_cubo,
            "lambda-structure": self.lambda_structure,
            "lambda-rev": self.lambda_rev,
            "lambda-self-observation": self.lambda_self_observation,
            "lambda-efector": self.lambda_efector,
            "reversibility-loss-every": self.reversibility_loss_every,
            "self-observation-every": self.self_observation_every,
            "efector-every": self.efector_every,
            "min-slot-var": self.min_slot_var,
        }
        argv: List[str] = []
        for key, value in pairs.items():
            argv.extend([f"--{key}", str(value)])
        return argv

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d.update(
            capacity=self.capacity,
            semantic_size=self.semantic_size,
            pad_size=self.pad_size,
            z_size=self.z_size,
            memory_size=self.memory_size,
            relations_size=self.relations_size,
        )
        return d


PROFILES: Dict[str, CTNetProfile] = {
    # Exact local smoke profile. It should run on CPU in seconds.
    "smoke": CTNetProfile(
        name="smoke",
        N=64,
        d=16,
        z_tokens=32,
        z_dim=16,
        mem_slots=8,
        mem_dim=16,
        rel_edges=8,
        rel_dim=16,
        fractal_steps=1,
        latent_steps=1,
        batch=1,
        max_bytes=1024,
    ),
    # Bigger than the original while still reasonable on CPU/small GPU.
    "base": CTNetProfile(
        name="base",
        N=256,
        d=64,
        z_tokens=160,
        z_dim=64,
        mem_slots=32,
        mem_dim=64,
        rel_edges=32,
        rel_dim=64,
        fractal_steps=4,
        latent_steps=2,
        batch=2,
        max_bytes=4096,
        lr=2e-4,
    ),
    # Serious GPU profile. State capacity 131072 coordinates per sample.
    "xl": CTNetProfile(
        name="xl",
        N=1024,
        d=128,
        z_tokens=768,
        z_dim=128,
        mem_slots=96,
        mem_dim=128,
        rel_edges=96,
        rel_dim=128,
        fractal_steps=8,
        latent_steps=4,
        batch=2,
        max_bytes=8192,
        lr=1.5e-4,
        grad_clip=0.75,
    ),
    # Maximum shipped geometry. Requires a large GPU for training.
    "max": CTNetProfile(
        name="max",
        N=4096,
        d=256,
        z_tokens=3072,
        z_dim=256,
        mem_slots=384,
        mem_dim=256,
        rel_edges=384,
        rel_dim=256,
        fractal_steps=16,
        latent_steps=8,
        batch=1,
        max_bytes=32768,
        lr=1e-4,
        grad_clip=0.5,
        lambda_anchor=0.05,
        lambda_coh=0.08,
        lambda_omega=0.35,
        lambda_structure=0.15,
        lambda_rev=0.15,
    ),
}


def get_profile(name: str) -> CTNetProfile:
    try:
        p = PROFILES[name]
    except KeyError as exc:
        raise SystemExit(f"unknown profile {name!r}; choose one of: {', '.join(PROFILES)}") from exc
    p.validate()
    return p


def iter_profiles() -> Iterable[CTNetProfile]:
    for p in PROFILES.values():
        p.validate()
        yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet 3.0 MAX profile registry")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [get_profile(args.profile).to_dict()] if args.profile else [p.to_dict() for p in iter_profiles()]
    if args.json:
        print(json.dumps(rows if args.profile is None else rows[0], indent=2, ensure_ascii=False))
        return
    for row in rows:
        print(
            f"{row['name']:>5} | Xi=[B,{row['N']},{row['d']}] capacity={row['capacity']} "
            f"semantic={row['semantic_size']} pad={row['pad_size']} "
            f"Z=[B,{row['z_tokens']},{row['z_dim']}] M=[B,{row['mem_slots']},{row['mem_dim']}] "
            f"R=[B,{row['rel_edges']},{row['rel_dim']}] fractal={row['fractal_steps']} latent={row['latent_steps']}"
        )


if __name__ == "__main__":
    main()
