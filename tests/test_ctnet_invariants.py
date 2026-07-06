#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ctnet_max_audit import audit_mthd, audit_profile
from ctnet_max_profiles import get_profile, iter_profiles
from ctnet_verifiable_operators import OperatorAtlas, compile_affine_rule


def assert_true(name: str, value: bool) -> None:
    if not value:
        raise AssertionError(name)


def test_profiles() -> dict:
    rows = []
    for p in iter_profiles():
        p.validate()
        assert_true(f"{p.name}.pad_non_negative", p.pad_size >= 0)
        assert_true(f"{p.name}.capacity_identity", p.semantic_size + p.pad_size == p.capacity)
        rows.append(p.to_dict())
    return {"profiles": rows}


def test_reversibility_smoke() -> dict:
    audit = audit_profile("smoke", batch=2, steps=2, fp64=False, cuda=False)
    assert_true("layout_ok", bool(audit["layout_ok"]))
    assert_true("reversible_ok", bool(audit["reversible_ok"]))
    assert_true("packed_mae_threshold", audit["packed_mae"] < 1e-5)
    return audit


def test_mthd_smoke() -> dict:
    audit = audit_mthd("smoke", batch=2, steps=1, fp64=False, cuda=False)
    assert_true("mthd_direct_read_ok", bool(audit["mthd_direct_read_ok"]))
    assert_true("mthd_fold_reversible_ok", bool(audit["mthd_fold_reversible_ok"]))
    assert_true("state_memory_shape_ok", bool(audit["state_memory_shape_ok"]))
    assert_true("state_relations_shape_ok", bool(audit["state_relations_shape_ok"]))
    return audit


def test_operator_atlas() -> dict:
    atlas = OperatorAtlas()
    coverage = atlas.coverage([("2+3*4", "14"), ("reverse: ctnet", "tentc")])
    assert_true("operator_coverage", coverage["coverage"] == 1.0)
    rule = compile_affine_rule([(0, 3), (1, 5), (2, 7), (5, 13)])
    assert_true("affine_rule", rule == {"type": "affine_int", "a": 2, "b": 3})
    return {"coverage": coverage, "compiled_rule": rule}


def main() -> None:
    results = {
        "profiles": test_profiles(),
        "reversibility": test_reversibility_smoke(),
        "mthd": test_mthd_smoke(),
        "operators": test_operator_atlas(),
    }
    print(json.dumps({"ok": True, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
