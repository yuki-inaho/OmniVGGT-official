"""Map-driven, audited non-strict weight transfer between checkpoints.

A weight map (JSON) declares where every parameter of the target model comes from:

    {
      "name": "...",
      "sources": {"omni": {"env": "OMNIVGGT_INIT_OMNI", "format": "safetensors"}, ...},
      "rules": [
        {"target_prefix": "aggregator.", "source": "omni", "source_prefix": "aggregator."},
        {"target_prefix": "aggregator.register_token", "source": "omni",
         "source_prefix": "aggregator.register_token", "slice": {"dim": 2, "start": 0, "stop": 4}},
        {"target_prefix": "aggregator.global_blocks.", "source": "omega",
         "source_prefix": "aggregator.inter_frame_blocks.",
         "multiply_by_suffix": {".attn.qkv.bias": ".attn.qkv.bias_mask"}}
      ],
      "new": ["aggregator.register_token"],          # target prefixes allowed to keep their init
      "drop": {"omni": ["point_head."], ...}         # source prefixes deliberately not used
    }

For each target key the rule with the longest matching ``target_prefix`` applies. A ``slice``
rule copies the source into that slice and keeps the model's initial values elsewhere (the key
must be declared ``new``). ``multiply_by_suffix`` multiplies a source tensor by a companion source
tensor (e.g. an effective bias = bias * bias_mask). Loading uses ``load_state_dict(strict=False)``,
but any key that is not accounted for by the map (unmapped/missing target, unused source, shape
mismatch) raises ``WeightTransferError``; nothing is dropped or initialised silently.

usage: uv run python -m omnivggt.utils.weight_transfer --variant <variant.json> [--seed 42] --report <json> --save <safetensors>
       (source paths come from the environment variables named in the map)
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42  # same as the training config's seed, so new parameters start from the same values


def repo_path(path) -> Path:
    """Paths inside variant JSONs are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def variant_map_path(variant_path) -> Path:
    return repo_path(json.loads(Path(variant_path).read_text())["weight_map"])


class WeightTransferError(ValueError):
    """The weight map does not account for a key or a shape."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_map(path) -> dict:
    path = Path(path)
    weight_map = json.loads(path.read_text())
    weight_map["_file"] = path.name
    weight_map["_sha256"] = sha256_of(path)
    return weight_map


def load_sources(weight_map: dict, env: Mapping[str, str] = os.environ):
    """Open every source of the map from the path in its environment variable."""
    sources, files = {}, {}
    for name, spec in weight_map["sources"].items():
        variable = spec["env"]
        if variable not in env or not env[variable]:
            raise KeyError(f"source '{name}': environment variable {variable} is not set")
        path = Path(env[variable])
        if not path.is_file():
            raise FileNotFoundError(f"source '{name}': {variable}={path} does not exist")
        if spec["format"] == "safetensors":
            from safetensors.torch import load_file

            sources[name] = load_file(str(path))
        elif spec["format"] == "torch":
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(state, Mapping):
                raise WeightTransferError(f"source '{name}': {path.name} is not a flat state dict")
            sources[name] = state
        else:
            raise WeightTransferError(f"source '{name}': unknown format {spec['format']!r}")
        files[name] = {"file": path.name, "sha256": sha256_of(path)}
    return sources, files


def _covered(key: str, prefixes) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def _rule_for(key: str, rules):
    matching = [rule for rule in rules if key.startswith(rule["target_prefix"])]
    return max(matching, key=lambda rule: len(rule["target_prefix"])) if matching else None


def _source_value(rule, source_key, sources, used):
    source = sources[rule["source"]]
    value = source[source_key]
    used[rule["source"]].add(source_key)
    for suffix, mask_suffix in (rule.get("multiply_by_suffix") or {}).items():
        if source_key.endswith(suffix):
            mask_key = source_key[: -len(suffix)] + mask_suffix
            if mask_key not in source:
                raise WeightTransferError(
                    f"{source_key}: companion key {mask_key} missing in source '{rule['source']}'"
                )
            used[rule["source"]].add(mask_key)
            return value * source[mask_key], True
    return value, False


def transfer_weights(model: torch.nn.Module, weight_map: dict, sources: Mapping, files: Mapping | None = None) -> dict:
    """Load ``model`` from ``sources`` as declared by ``weight_map``; return an audit report."""
    target = model.state_dict()
    rules, new_prefixes = weight_map["rules"], weight_map.get("new", [])
    for rule in rules:
        if rule["source"] not in sources:
            raise WeightTransferError(f"rule for '{rule['target_prefix']}' names unknown source '{rule['source']}'")
    used = {name: set() for name in sources}
    selected, classes, casts, sources_numel = {}, {}, [], {}
    multiplied, sliced = [], []
    errors = []
    for key, init in target.items():
        rule = _rule_for(key, rules)
        source_key = None if rule is None else rule["source_prefix"] + key[len(rule["target_prefix"]) :]
        if rule is None or source_key not in sources[rule["source"]]:
            if _covered(key, new_prefixes):
                classes[key] = "new"
            else:
                errors.append(
                    f"target {key}: no source ({'unmapped' if rule is None else rule['source'] + ':' + source_key})"
                )
            continue
        value, was_multiplied = _source_value(rule, source_key, sources, used)
        if value.dtype != init.dtype:
            casts.append({"key": key, "from": str(value.dtype), "to": str(init.dtype)})
            value = value.to(init.dtype)
        spec = rule.get("slice")
        if spec is not None:
            expected = list(init.shape)
            expected[spec["dim"]] = spec["stop"] - spec["start"]
            if list(value.shape) != expected:
                errors.append(f"target {key}: slice expects {expected}, source {source_key} is {list(value.shape)}")
                continue
            if not _covered(key, new_prefixes):
                errors.append(f"target {key}: sliced but not declared new (the rest keeps its init)")
                continue
            sources_numel[key] = int(value.numel())
            merged = init.detach().clone()
            merged.narrow(spec["dim"], spec["start"], spec["stop"] - spec["start"]).copy_(value)
            selected[key], classes[key] = merged, "sliced"
            sliced.append(key)
            continue
        if value.shape != init.shape:
            errors.append(f"target {key}: shape {list(init.shape)} != source {source_key} {list(value.shape)}")
            continue
        selected[key] = value
        classes[key] = "loaded" if source_key == key and rule["source_prefix"] == rule["target_prefix"] else "renamed"
        if was_multiplied:
            multiplied.append(key)
    drops = weight_map.get("drop", {})
    for name, source in sources.items():
        for key in source:
            if key not in used[name] and not _covered(key, drops.get(name, [])):
                errors.append(f"source {name}:{key} is neither used nor declared in drop")
    if errors:
        raise WeightTransferError(f"{len(errors)} undeclared difference(s):\n" + "\n".join(errors))

    result = model.load_state_dict(selected, strict=False)
    new_keys = sorted(key for key, kind in classes.items() if kind == "new")
    if sorted(result.missing_keys) != new_keys or result.unexpected_keys:
        raise WeightTransferError(
            f"load_state_dict disagrees with the map: missing={sorted(result.missing_keys)} new={new_keys} "
            f"unexpected={result.unexpected_keys}"
        )

    def summary(kind):
        keys = [key for key, value in classes.items() if value == kind]
        return {"keys": len(keys), "params": int(sum(target[key].numel() for key in keys))}

    source_report = {}
    for name, source in sources.items():
        dropped = [key for key in source if key not in used[name]]
        source_report[name] = {
            **(files or {}).get(name, {}),
            "used_keys": len(used[name]),
            "used_params": int(sum(source[key].numel() for key in used[name])),
            "dropped_keys": len(dropped),
            "dropped_params": int(sum(source[key].numel() for key in dropped)),
        }
    return {
        "map": {"name": weight_map.get("name"), "file": weight_map.get("_file"), "sha256": weight_map.get("_sha256")},
        "sources": source_report,
        "targets": {kind: summary(kind) for kind in ("loaded", "renamed", "sliced", "new")},
        "new_keys": new_keys,
        "sliced_keys": sorted(sliced),
        "sliced_new_elements": {key: int(target[key].numel() - sources_numel[key]) for key in sorted(sliced)},
        "multiplied_keys": sorted(multiplied),
        "casts": casts,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "total_state_entries": len(target),
    }


def main(argv=None, model=None, env: Mapping[str, str] = os.environ) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", type=Path, help="variant JSON; builds OmniVGGTOmega and names the weight map")
    parser.add_argument("--map", type=Path, help="weight map JSON (default: the variant's weight_map)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--save", type=Path, help="write the loaded state dict as safetensors")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="torch seed before building the model (initial values of new parameters)",
    )
    args = parser.parse_args(argv)
    map_path = args.map or (variant_map_path(args.variant) if args.variant is not None else None)
    if map_path is None:
        parser.error("--map or --variant is required")
    weight_map = load_map(map_path)
    sources, files = load_sources(weight_map, env=env)  # fail before building the 1B-parameter model
    if model is None:
        if args.variant is None:
            parser.error("--variant is required to build the model")
        from omnivggt.models.omnivggt_omega import OmniVGGTOmega

        torch.manual_seed(args.seed)  # fixes the initial values of newly created parameters
        model = OmniVGGTOmega.from_variant(args.variant)
    report = transfer_weights(model, weight_map, sources, files=files)
    if args.variant is not None:
        report["variant"] = {"file": args.variant.name, "sha256": sha256_of(args.variant), "seed": args.seed}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    if args.save is not None:
        from safetensors.torch import save_file

        save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(args.save))
    print(json.dumps({"targets": report["targets"], "new_keys": report["new_keys"][:8]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
