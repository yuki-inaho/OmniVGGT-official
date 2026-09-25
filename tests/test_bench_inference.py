import bench_inference
import pytest


def test_summary_statistics():
    summary = bench_inference.summarize([4.0, 1.0, 3.0, 2.0, 10.0])
    assert summary == {"median": 3.0, "p90": pytest.approx(7.6), "min": 1.0, "max": 10.0, "n": 5}


def test_section_totals():
    totals = bench_inference.section_totals([("frame_blocks", 1.5), ("global_blocks", 2.0), ("frame_blocks", 0.5)])
    assert totals == {"frame_blocks": 2.0, "global_blocks": 2.0}


def test_no_points_is_rejected_for_omnivggt():
    with pytest.raises(ValueError, match="point head"):
        bench_inference.check_options(model_config=None, no_points=True)
    bench_inference.check_options(model_config="V4.json", no_points=True)


def test_depth_camera_inputs_are_fixed_and_well_formed():
    first = bench_inference.make_inputs(frames=3, height=28, width=42, condition="depth+camera", device="cpu")
    second = bench_inference.make_inputs(frames=3, height=28, width=42, condition="depth+camera", device="cpu")
    assert first["depth_gt_index"] == [0, 1, 2] and first["camera_gt_index"] == [0, 1, 2]
    assert first["depth"].shape == (1, 3, 28, 42, 1) and first["extrinsics"].shape == (1, 3, 3, 4)
    assert (first["images"] == second["images"]).all()
    rgb = bench_inference.make_inputs(frames=3, height=28, width=42, condition="rgb", device="cpu")
    assert rgb["depth_gt_index"] == [] and rgb["camera_gt_index"] == [] and rgb["mask"] is None


def test_sections_include_unprojection_for_omega():
    import torch

    from omnivggt.models.omnivggt_omega import OmniVGGTOmega

    torch.manual_seed(0)
    model = OmniVGGTOmega(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        num_register_tokens=4,
        register_attention_layers=(1,),
        global_rope=False,
        cached_layers=(0, 1, 2, 3),
        aggregator_kwargs=dict(depth=4, num_heads=2, patch_embed="conv"),
        camera_head_kwargs=dict(trunk_depth=1, num_heads=2),
        depth_head_kwargs=dict(features=16, out_channels=[8, 16, 32, 32], intermediate_layer_idx=[0, 1, 2, 3]),
    )
    groups = bench_inference._section_modules(model)
    assert groups["unprojection"] == [model.point_unprojector]
    assert "point_head" not in groups
