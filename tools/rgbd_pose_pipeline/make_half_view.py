"""Build a temporally subsampled view of a standardized RGB-D bundle.

Input  (standard bundle): ``rgb/<stem>_rgb.jpg``, ``depth/<stem>_depth.png``,
``mapped_depth/<stem>_depth.png`` and ``camera_parameters/*.yaml``.
Output (view): every ``step``-th frame starting at ``offset`` with

* ``rgb/<stem>_rgb.png``          decoded RGB saved losslessly as PNG,
* ``mapped_depth/<stem>_depth.png`` hard link to the verified mapped depth,
* ``depth/<stem>_depth.png``        hard link to the raw depth,
* ``camera_parameters/``            copied YAML files,
* ``README.md``                     mapped-depth provenance (required by the staging exporter),
* ``subsample_manifest.json``.

The mapped depth must come with a verification report whose ``all_passed`` is
true and whose ``dilation_kernel_size`` is 3; anything else is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image

RGB_SUFFIX = "_rgb.jpg"
DEPTH_SUFFIX = "_depth.png"
REQUIRED_KERNEL = 3


def _load_check(path: Path) -> dict:
    check = json.loads(Path(path).read_text())
    kernel = check.get("args", {}).get("dilation_kernel_size")
    if check.get("all_passed") is not True or kernel != REQUIRED_KERNEL:
        raise ValueError(
            f"mapped depth check must pass with kernel {REQUIRED_KERNEL}: all_passed={check.get('all_passed')}, kernel={kernel}"
        )
    return check


def _readme(check: dict, step: int, offset: int) -> str:
    max_depth = check["args"].get("max_depth")
    return (
        "# Subsampled RGB-D view\n\n"
        f"- frames: every {step}-th frame of the standardized bundle starting at offset {offset}\n"
        "- rgb: decoded JPEG pixels stored losslessly as PNG (`<stem>_rgb.png`)\n"
        "- mapped_depth: depth projected to the RGB field of view with nearest-Z splatting, "
        f"followed by 3x3 nearest-depth dilation (kernel {REQUIRED_KERNEL}), uint16 millimetres, "
        f"max depth {max_depth} mm, invalid = 0\n"
        "- depth: raw sensor depth (not aligned to RGB)\n"
        "- provenance of the 3x3 nearest-depth dilation was verified by recomputation before building this view\n"
    )


def make_half_view(standard: Path, output: Path, *, step: int, offset: int, mapped_depth_check: Path) -> dict:
    standard, output = Path(standard), Path(output)
    if step < 1 or not 0 <= offset < step:
        raise ValueError("require step >= 1 and 0 <= offset < step")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    check = _load_check(mapped_depth_check)

    stems = sorted(p.name.removesuffix(RGB_SUFFIX) for p in (standard / "rgb").glob(f"*{RGB_SUFFIX}"))
    if not stems:
        raise FileNotFoundError(f"no *{RGB_SUFFIX} files in {standard / 'rgb'}")
    selected = stems[offset::step]
    for stem in selected:
        for sub in ("mapped_depth", "depth"):
            path = standard / sub / f"{stem}{DEPTH_SUFFIX}"
            if not path.is_file():
                raise FileNotFoundError(f"missing {sub} for selected frame {stem}: {path}")
    camera_files = sorted((standard / "camera_parameters").glob("*.yaml"))
    if not camera_files:
        raise FileNotFoundError("no camera YAML files in the standardized bundle")

    for sub in ("rgb", "mapped_depth", "depth", "camera_parameters"):
        (output / sub).mkdir(parents=True)
    for stem in selected:
        with Image.open(standard / "rgb" / f"{stem}{RGB_SUFFIX}") as image:
            image.convert("RGB").save(output / "rgb" / f"{stem}_rgb.png")
        for sub in ("mapped_depth", "depth"):
            os.link(standard / sub / f"{stem}{DEPTH_SUFFIX}", output / sub / f"{stem}{DEPTH_SUFFIX}")
    for camera_file in camera_files:
        shutil.copy2(camera_file, output / "camera_parameters" / camera_file.name)
    (output / "README.md").write_text(_readme(check, step, offset))

    manifest = {
        "source_frame_count": len(stems),
        "step": step,
        "offset": offset,
        "frame_count": len(selected),
        "selected_stems": selected,
        "rgb_conversion": "JPEG decoded with PIL and saved as PNG (lossless w.r.t. decoded pixels)",
        "mapped_depth_check": {k: check[k] for k in check if k != "rows"},
    }
    (output / "subsample_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--mapped-depth-check", type=Path, required=True)
    args = parser.parse_args()
    manifest = make_half_view(
        args.standard, args.output, step=args.step, offset=args.offset, mapped_depth_check=args.mapped_depth_check
    )
    print(json.dumps({k: manifest[k] for k in ("source_frame_count", "step", "offset", "frame_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
