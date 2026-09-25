"""Regenerate ``zero_aggregator_golden.pt``: a characterization fixture of ``ZeroAggregator.inference``.

The fixture freezes the behaviour of the unmodified aggregator (tiny config, conv patch embedding,
random non-zero weights so that the GeoAdapter paths are exercised).  Regenerate it only from a
revision whose aggregator behaviour is known to be correct.

usage: PYTHONPATH=. uv run python tests/fixtures/make_zero_aggregator_golden.py
"""

from pathlib import Path

import torch

from omnivggt.models.omnivggt_aggregator import ZeroAggregator

CONFIG = dict(img_size=28, patch_size=14, embed_dim=32, depth=4, num_heads=2, patch_embed="conv")
FRAMES, HEIGHT, WIDTH = 3, 28, 42
CONDITIONS = {
    "rgb": ([], []),
    "depth": ([0, 1, 2], []),
    "camera": ([], [0, 1, 2]),
    "depth+camera": ([0, 1, 2], [0, 1, 2]),
    "partial": ([1], [0, 1]),
}
FIXTURE = Path(__file__).with_name("zero_aggregator_golden.pt")


def make_inputs(generator: torch.Generator) -> dict:
    angles = torch.tensor([0.0, 0.1, -0.2])
    rotations = []
    for angle in angles:
        c, s = torch.cos(angle), torch.sin(angle)
        rotations.append(torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]))
    translations = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.1], [0.5, -0.1, 0.2]])
    extrinsics = torch.cat([torch.stack(rotations), translations[:, :, None]], dim=-1)[None]
    intrinsics = torch.tensor([[30.0, 0.0, WIDTH / 2], [0.0, 30.0, HEIGHT / 2], [0.0, 0.0, 1.0]]).expand(
        1, FRAMES, 3, 3
    )
    mask = (torch.rand(1, FRAMES, HEIGHT, WIDTH, generator=generator) > 0.2).float()
    return {
        "images": torch.rand(1, FRAMES, 3, HEIGHT, WIDTH, generator=generator),
        "extrinsics": extrinsics.contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "depth": 0.5 + torch.rand(1, FRAMES, HEIGHT, WIDTH, 1, generator=generator),
        "mask": mask,
    }


def build_model(seed: int = 0) -> ZeroAggregator:
    torch.manual_seed(seed)
    model = ZeroAggregator(**CONFIG)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def run(model: ZeroAggregator, inputs: dict) -> dict:
    outputs = {}
    with torch.no_grad():
        for name, (depth_index, camera_index) in CONDITIONS.items():
            layers, patch_start_idx = model.inference(
                **inputs, depth_gt_index=depth_index, camera_gt_index=camera_index
            )
            outputs[name] = [layer.clone() for layer in layers]
    outputs["patch_start_idx"] = patch_start_idx
    return outputs


def main() -> None:
    model = build_model()
    inputs = make_inputs(torch.Generator().manual_seed(1))
    torch.save(
        {"config": CONFIG, "state_dict": model.state_dict(), "inputs": inputs, "outputs": run(model, inputs)},
        FIXTURE,
    )
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
