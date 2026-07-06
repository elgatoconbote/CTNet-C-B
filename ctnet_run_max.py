#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entrypoint for CTNet 3.0 MAX."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from ctnet_max_profiles import get_profile, iter_profiles

ROOT = Path(__file__).resolve().parent


def train_command(args: argparse.Namespace) -> List[str]:
    p = get_profile(args.profile)
    cmd = [sys.executable, str(ROOT / "train_vram_up_coherence_ctnet.py")]
    cmd += p.as_train_argv()
    cmd += ["--steps", str(args.steps), "--log-every", str(args.log_every)]
    cmd += ["--save-final", str(args.save_final or ROOT / "checkpoints" / f"ctnet_{args.profile}_final.pt")]
    if args.cuda:
        cmd.append("--cuda")
    if args.fp64:
        cmd.append("--fp64")
    if args.synthetic:
        cmd.append("--synthetic")
    for path in args.local_file or []:
        cmd.extend(["--local-file", path])
    for url in args.url or []:
        cmd.extend(["--url", url])
    if args.coherence_grad_scale:
        cmd.append("--coherence-grad-scale")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet 3.0 MAX runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("profiles", help="List scaling profiles")

    audit = sub.add_parser("audit", help="Run reversibility/layout audit")
    audit.add_argument("--profile", choices=[p.name for p in iter_profiles()], default="smoke")
    audit.add_argument("--steps", type=int, default=1)
    audit.add_argument("--mthd", action="store_true")
    audit.add_argument("--cuda", action="store_true")
    audit.add_argument("--fp64", action="store_true")

    bench = sub.add_parser("bench", help="Run functional reactivity benchmark")
    bench.add_argument("--profile", choices=[p.name for p in iter_profiles()], default="smoke")
    bench.add_argument("--steps", type=int, default=3)
    bench.add_argument("--batch", type=int, default=2)
    bench.add_argument("--cuda", action="store_true")

    train = sub.add_parser("train", help="Train a CTNet profile")
    train.add_argument("--profile", choices=[p.name for p in iter_profiles()], default="smoke")
    train.add_argument("--steps", type=int, default=10)
    train.add_argument("--log-every", type=int, default=1)
    train.add_argument("--local-file", action="append", default=[])
    train.add_argument("--url", action="append", default=[])
    train.add_argument("--synthetic", action="store_true")
    train.add_argument("--cuda", action="store_true")
    train.add_argument("--fp64", action="store_true")
    train.add_argument("--coherence-grad-scale", action="store_true")
    train.add_argument("--save-final", default=None)
    train.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "profiles":
        for p in iter_profiles():
            print(json.dumps(p.to_dict(), ensure_ascii=False))
        return
    if args.cmd == "audit":
        script = "ctnet_max_audit.py"
        cmd = [sys.executable, str(ROOT / script), "--profile", args.profile, "--steps", str(args.steps)]
        if args.mthd:
            cmd.append("--mthd")
        if args.cuda:
            cmd.append("--cuda")
        if args.fp64:
            cmd.append("--fp64")
        raise SystemExit(subprocess.call(cmd))
    if args.cmd == "bench":
        cmd = [sys.executable, str(ROOT / "ctnet_functional_benchmark.py"), "--profile", args.profile, "--steps", str(args.steps), "--batch", str(args.batch)]
        if args.cuda:
            cmd.append("--cuda")
        raise SystemExit(subprocess.call(cmd))
    if args.cmd == "train":
        cmd = train_command(args)
        if args.dry_run:
            print(" ".join(cmd))
            return
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
