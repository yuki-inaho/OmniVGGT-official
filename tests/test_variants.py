"""Variant files of OmniVGGTOmega must be complete and consistent with their weight maps."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "configs" / "omnivggt_omega" / "variants"
EXPECTED = {"V0", "V1", "V2", "V3", "V4", "V5", "V6"}
KEYS = {
    "name",
    "description",
    "num_register_tokens",
    "register_attention_layers",
    "global_rope",
    "cached_layers",
    "weight_map",
}


def _variants():
    return {path.stem: json.loads(path.read_text()) for path in sorted(VARIANTS.glob("*.json"))}


def test_all_variants_present():
    assert set(_variants()) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_all_variant_files_build_and_match_maps(name):
    variant = _variants()[name]
    assert set(variant) == KEYS
    assert variant["name"] == name
    weight_map = json.loads((ROOT / variant["weight_map"]).read_text())
    sliced = any(rule.get("slice") for rule in weight_map["rules"])
    assert sliced == (variant["num_register_tokens"] != 4)
    assert variant["num_register_tokens"] in (4, 16)
    assert all(0 <= index < 24 for index in variant["register_attention_layers"])
    assert {4, 11, 17, 23} <= set(variant["cached_layers"])
    uses_omega = "omega" in weight_map["sources"]
    if uses_omega:
        assert variant["global_rope"] is False  # VGGT-Omega's inter-frame blocks were trained without RoPE
