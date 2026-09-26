import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _gt_c2w(frames, step_m=0.014, seed=0):
    """Camera-to-world poses along a rail: constant steps along x with small rotations."""
    rng = np.random.default_rng(seed)
    out = np.tile(np.eye(4), (frames, 1, 1))
    out[:, :3, :3] = Rotation.from_rotvec(np.deg2rad(0.5) * rng.normal(size=(frames, 3))).as_matrix()
    out[:, 0, 3] = step_m * np.arange(frames)
    out[:, 1, 3] = 0.002 * np.sin(np.arange(frames) / 5.0)
    return out


def _similarity(c2w, scale, rotation, translation):
    out = c2w.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", rotation, c2w[:, :3, :3])
    out[:, :3, 3] = scale * c2w[:, :3, 3] @ rotation.T + translation
    return out


def _drifted(c2w, per_frame):
    """Poses whose i-th step length is multiplied by (1 + per_frame * i)."""
    relative = np.einsum("nji,njk->nik", c2w[:-1, :3, :3], c2w[1:, :3, :3])
    steps = np.einsum("nji,nj->ni", c2w[:-1, :3, :3], c2w[1:, :3, 3] - c2w[:-1, :3, 3])
    out = [c2w[0]]
    for i in range(len(steps)):
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = relative[i], steps[i] * (1 + per_frame * i)
        out.append(out[-1] @ step)
    return np.array(out)


def _w2c(c2w):
    return np.linalg.inv(c2w)[:, :3]


ROTATION = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()


def test_exact_similarity_gives_zero_trajectory_errors():
    import stream_metrics

    gt = _gt_c2w(40)
    pred = _similarity(gt, 0.37, ROTATION, np.array([1.0, 2.0, 3.0]))
    assert stream_metrics.ate_first_frame(pred, gt, 1 / 0.37) == pytest.approx(np.zeros(40), abs=1e-9)
    assert stream_metrics.ate_sim3(pred, gt) == pytest.approx(np.zeros(40), abs=1e-9)
    for delta in (1, 8, 16):
        assert stream_metrics.rpe_translation(pred, gt, 1 / 0.37, delta) == pytest.approx(0.0, abs=1e-9)
    assert stream_metrics.pose_scale_ratios(pred, gt, 1 / 0.37) == pytest.approx(np.ones(32))


def test_step_length_drift_shows_in_first_frame_gauge_and_pose_scale_ratio():
    import stream_metrics

    gt = _gt_c2w(64)
    pred = _similarity(_drifted(gt, 0.002), 0.37, ROTATION, np.zeros(3))
    first_frame = stream_metrics.ate_first_frame(pred, gt, 1 / 0.37)
    sim3 = stream_metrics.ate_sim3(pred, gt)
    rmse = lambda errors: np.sqrt(np.mean(errors**2))  # noqa: E731
    assert rmse(first_frame) > 2 * rmse(sim3) > 0
    assert first_frame[0] == pytest.approx(0.0, abs=1e-12)  # frame 1 is the gauge
    ratios = stream_metrics.pose_scale_ratios(pred, gt, 1 / 0.37)
    assert len(ratios) == 64 - stream_metrics.DRIFT_DELTA
    assert np.all(np.diff(ratios) > 0) and ratios[0] > 1.0


def test_ate_sim3_uses_every_frame(monkeypatch):
    import stream_metrics

    gt = _gt_c2w(30)
    pred = _similarity(gt, 0.5, ROTATION, np.zeros(3))
    pred[7, :3, 3] += 0.05  # one outlier camera centre
    seen = {}
    original = stream_metrics.align_pose_sim3

    def spy(source, target, trim_fraction=0.1, **kwargs):
        seen["trim_fraction"] = trim_fraction
        return original(source, target, trim_fraction=trim_fraction, **kwargs)

    monkeypatch.setattr(stream_metrics, "align_pose_sim3", spy)
    errors = stream_metrics.ate_sim3(pred, gt)
    assert seen["trim_fraction"] == 0.0
    _, _, _, aligned = original(pred, gt, trim_fraction=0.0)
    assert errors == pytest.approx(np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1))


@pytest.mark.parametrize(
    "frames,deltas", [(8, (1,)), (9, (1, 8)), (32, (1, 8, 16)), (63, (1, 8, 16)), (64, (1, 8, 16, 32))]
)
def test_rpe_deltas_the_window_can_hold(frames, deltas):
    import stream_metrics

    assert stream_metrics.rpe_deltas(frames) == deltas


def test_undefined_rpe_and_pose_drift_are_rejected_not_nan():
    import stream_metrics

    gt = _gt_c2w(8)
    with pytest.raises(ValueError, match="delta"):
        stream_metrics.rpe_translation(gt, gt, 1.0, 8)
    with pytest.raises(ValueError, match="delta"):
        stream_metrics.pose_scale_ratios(gt, gt, 1.0)
    still = np.tile(np.eye(4), (12, 1, 1))
    with pytest.raises(ValueError, match="displacement"):
        stream_metrics.pose_scale_ratios(still, still, 1.0)


def _depth_window(frames, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.5, 1.5, size=(frames, 4, 6))
    mask = rng.uniform(size=gt.shape) > 0.3
    mask[:, 0, 0] = True  # every frame has a valid pixel
    return gt, mask


def test_depth_scales_follow_the_first_frame():
    import stream_metrics

    gt, mask = _depth_window(12)
    drift = 1 + 0.01 * np.arange(12)
    pred = gt / (0.37 * drift)[:, None, None]
    assert stream_metrics.depth_scale(pred[0], gt[0], mask[0]) == pytest.approx(0.37)
    ratios = stream_metrics.depth_scales(pred, gt, mask) / stream_metrics.depth_scale(pred[0], gt[0], mask[0])
    assert ratios == pytest.approx(drift)
    mask[3] = False
    with pytest.raises(ValueError, match="valid"):
        stream_metrics.depth_scales(pred, gt, mask)


def test_time_bins_cover_the_window_once():
    import stream_metrics

    assert stream_metrics.time_bins(8) == [("t1-8", slice(0, 8))]
    assert stream_metrics.time_bins(40) == [("t1-8", slice(0, 8)), ("t9-32", slice(8, 32)), ("t33-64", slice(32, 40))]
    assert [label for label, _ in stream_metrics.time_bins(146)] == ["t1-8", "t9-32", "t33-64", "t65+"]
    assert stream_metrics.time_bins(146)[-1][1] == slice(64, 146)


def test_depth_report_by_time_bin_and_scaling():
    import eval_colmap_rgbd
    import stream_metrics

    gt, mask = _depth_window(70)
    error = np.where(np.arange(70) < 8, 0.0, np.where(np.arange(70) < 32, 0.1, 0.2))  # relative error per frame
    pred = gt * (1 + error)[:, None, None] / 2.0
    s1 = stream_metrics.depth_scale(pred[0], gt[0], mask[0])
    assert s1 == pytest.approx(2.0)
    report = stream_metrics.depth_report(pred, gt, mask, s1)
    assert report["abs_rel_s1@t1-8"] == pytest.approx(0.0, abs=1e-12)
    assert report["abs_rel_s1@t9-32"] == pytest.approx(0.1)
    assert report["abs_rel_s1@t33-64"] == pytest.approx(0.2)
    assert report["abs_rel_s1@t65+"] == pytest.approx(0.2)
    assert report["delta<1.25_s1@t33-64"] == 1.0
    pixels = mask.sum(axis=(1, 2))
    pooled = (pixels * error).sum() / pixels.sum()
    assert report["abs_rel_s1"] == pytest.approx(pooled)
    # the window-median scaling is eval_colmap_rgbd.depth_metrics, bit for bit
    legacy = eval_colmap_rgbd.depth_metrics(pred, gt, mask)
    assert report["abs_rel_median"] == legacy["abs_rel"]
    assert report["delta<1.25_median"] == legacy["delta<1.25"]
    assert {"abs_rel_median@t1-8", "abs_rel_median@t65+", "delta<1.25_median@t9-32"} <= set(report)


def test_window_metrics_are_finite_and_omit_undefined_values():
    import stream_metrics

    gt_c2w = _gt_c2w(32)
    pred_c2w = _similarity(gt_c2w, 0.5, ROTATION, np.zeros(3))
    gt_depth, mask = _depth_window(32)
    result = stream_metrics.window_metrics(_w2c(pred_c2w), _w2c(gt_c2w), gt_depth * 0.5, gt_depth, mask)
    metrics = result["metrics"]
    assert metrics["s1"] == pytest.approx(2.0)
    assert metrics["ate_g1_rmse_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["ate_sim3_rmse_mm"] == pytest.approx(0.0, abs=1e-6)
    assert {"rpe_t_mm@1", "rpe_t_mm@8", "rpe_t_mm@16"} <= set(metrics) and "rpe_t_mm@32" not in metrics
    assert metrics["pose_scale_ratio_max_dev"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["depth_scale_ratio_last"] == pytest.approx(1.0)
    assert metrics["AUC@30"] > 0.99 and "abs_rel_median@t33-64" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert len(result["series"]["ate_g1_err_mm"]) == 32 and len(result["series"]["depth_scale_ratio"]) == 32
    short = stream_metrics.window_metrics(_w2c(pred_c2w[:8]), _w2c(gt_c2w[:8]), gt_depth[:8], gt_depth[:8], mask[:8])
    assert "rpe_t_mm@8" not in short["metrics"] and "pose_scale_ratio_first" not in short["metrics"]
    assert "pose_scale_ratio" not in short["series"]


# ---------------------------------------------------------------- windows and loading (eval_stream)
W, H = 64, 48
RESOLUTION = (56, 42)
SPLITS = {"train": range(0, 8), "val": range(10, 42), "smoke": range(44, 50)}


def _write_root(root, seed, frames=50):
    """A colmap_rgbd_v1 root laid out like tests/test_colmap_rgbd_dataset.py, with longer contiguous splits."""
    import json

    from PIL import Image

    scene = root / "scenes" / "scene_000000"
    (scene / "rgb").mkdir(parents=True)
    (scene / "depth").mkdir()
    rng = np.random.default_rng(seed)
    k = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]], dtype=np.float32)
    w2c = np.zeros((frames, 3, 4), dtype=np.float32)
    for index in range(frames):
        w2c[index, :3, :3] = np.eye(3)
        w2c[index, :3, 3] = [0.0, -0.01 * index, 0.001 * seed]
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(scene / "rgb" / f"frame_{index:06d}.png")
        depth = rng.integers(400, 1300, (H, W), dtype=np.uint16)
        depth[:4] = 0
        Image.fromarray(depth).save(scene / "depth" / f"frame_{index:06d}.png")
    np.savez_compressed(
        scene / "cameras.npz",
        frame_ids=np.arange(frames),
        intrinsics=np.repeat(k[None], frames, 0),
        extrinsics_w2c=w2c,
        quality_flags=np.ones(frames, bool),
        chunk_ids=np.zeros(frames, np.int64),
    )
    sequences, split_ids = [], []
    for split_id, frame_range in enumerate(SPLITS.values()):
        chunk = list(frame_range)
        for first in range(0, len(chunk), 4):
            rows = chunk[first : first + 4]
            sequences.append(rows + [-1] * (4 - len(rows)))
            split_ids.append(split_id)
    np.savez_compressed(
        scene / "sequences.npz",
        sequences=np.array(sequences),
        lengths=np.array([sum(f >= 0 for f in row) for row in sequences]),
        split_ids=np.array(split_ids),
        chunk_ids=np.zeros(len(sequences), np.int64),
    )
    meta = {
        "format": "colmap_rgbd_v1",
        "schema_version": 1,
        "frame_count": frames,
        "depth": {"unit": "millimeters", "invalid_value": 0, "max_depth_mm": 1300},
        "camera": {"extrinsics": "opencv_world_to_camera", "intrinsics": "pixel_units"},
    }
    (root / "dataset.json").write_text(json.dumps(meta))
    return str(root)


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("staging")
    return [_write_root(base / "root_a", seed=0), _write_root(base / "root_b", seed=1)]


@pytest.mark.parametrize(
    "spec,expected",
    [("s0:950-981", (0, 950, 981, 1)), ("s1:792-930/2", (1, 792, 930, 2)), (" s1:10-13/3 ", (1, 10, 13, 3))],
)
def test_window_grammar(spec, expected):
    import eval_stream

    window = eval_stream.parse_window(spec)
    assert (window.session, window.start, window.end, window.stride) == expected
    assert window.frames == list(range(expected[1], expected[2] + 1, expected[3]))
    assert eval_stream.parse_window(window.spec) == window


@pytest.mark.parametrize("spec", ["x0:1-5", "s0:5-1", "s0:5-5", "s0:1-6/2", "s0:1-5/0", "s0:1-", "s0:-3-5", "s0:1-5/"])
def test_malformed_windows_are_rejected(spec):
    import eval_stream

    with pytest.raises(ValueError, match="window"):
        eval_stream.parse_window(spec)


def test_window_list():
    import eval_stream

    windows = eval_stream.parse_windows("s0:950-981, s1:968-999,s0:950-997")
    assert [w.spec for w in windows] == ["s0:950-981", "s1:968-999", "s0:950-997"]
    with pytest.raises(ValueError, match="window"):
        eval_stream.parse_windows("s0:950-981,,s1:968-999")


@pytest.mark.parametrize("spec", ["s0:30-45", "s0:5-12", "s1:40-44", "s2:12-20"])
def test_windows_outside_the_split_or_roots_are_rejected(roots, spec):
    import eval_stream

    loader = eval_stream.WindowLoader(roots, "val", RESOLUTION)
    with pytest.raises(ValueError, match="window"):
        loader.load(eval_stream.parse_window(spec))


def test_window_frames_are_colmap_rgbd_sequential_views(roots):
    import eval_stream
    import torch

    from omnivggt.datasets.colmap_rgbd import ColmapRgbd
    from omnivggt.datasets.utils.transforms import ImgNorm

    loader = eval_stream.WindowLoader(roots, "val", RESOLUTION)
    item = loader.load(eval_stream.parse_window("s1:12-20/2"))
    reference = ColmapRgbd(
        roots=[roots[1]],
        split="val",
        resolution=[RESOLUTION],
        transform=ImgNorm,
        aug_crop=0,
        seed=1,
        view_selection="sequential",
        sequential_stride=2,
    )[(12 - SPLITS["val"][0], 0, 5)]
    assert item["instance"] == [f"frame_{f:06d}.png" for f in (12, 14, 16, 18, 20)] == reference["instance"]
    assert torch.equal(item["images"], reference["images"])
    for key in ("depth", "extrinsic", "intrinsic", "valid_mask"):
        assert np.array_equal(item[key], reference[key])
    smoke = eval_stream.WindowLoader(roots, "smoke", RESOLUTION).load(eval_stream.parse_window("s0:44-49"))
    assert smoke["instance"][-1] == "frame_000049.png"


def test_l8_preset_reproduces_eval_colmap_rgbd_anchors(roots):
    import eval_colmap_rgbd
    import eval_stream
    import torch

    dataset = eval_colmap_rgbd.sequential_dataset(roots, "val", RESOLUTION, stride=3)
    # the anchor rule of eval_colmap_rgbd (frames 8, stride 3, 16 samples), written out
    span = 7 * 3
    valid = [
        i
        for i in range(len(dataset))
        if i + span < len(dataset) and dataset.scene_labels[i + span] == dataset.scene_labels[i]
    ]
    expected = [valid[k] for k in np.linspace(0, len(valid) - 1, 16).round().astype(int)]
    assert eval_colmap_rgbd.sequential_anchors(dataset, frames=8, stride=3, num_samples=16) == expected

    windows, anchors = eval_stream.l8_windows(roots, "val", RESOLUTION)
    assert anchors == expected and len(windows) == 16
    assert {w.session for w in windows} == {0, 1} and all(w.stride == 3 and len(w.frames) == 8 for w in windows)
    for window, anchor in zip(windows, anchors, strict=True):
        reference = dataset[(anchor, 0, 8)]
        assert reference["instance"] == [f"frame_{f:06d}.png" for f in window.frames]
        assert reference["label"][0].split("/")[0] == ("root_a", "root_b")[window.session]
    loader = eval_stream.WindowLoader(roots, "val", RESOLUTION)
    for session in (0, 1):
        window, anchor = next((w, a) for w, a in zip(windows, anchors, strict=True) if w.session == session)
        item, reference = loader.load(window), dataset[(anchor, 0, 8)]
        assert torch.equal(item["images"], reference["images"]) and np.array_equal(item["depth"], reference["depth"])
    with pytest.raises(ValueError, match="val"):
        eval_stream.l8_windows(roots, "smoke", RESOLUTION)


def test_cli_requires_split_and_exactly_one_window_source():
    import eval_stream

    parser = eval_stream.build_parser()
    base = ["--model-config", "V5.json", "--checkpoint", "ckpt", "--roots", "a", "b", "--mode", "bidir"]
    base += ["--output", "o.json"]
    args = parser.parse_args([*base, "--split", "smoke", "--windows", "s0:950-981"])
    assert args.precision == "fp32" and args.conditions == ["depth"] and args.preset is None
    assert parser.parse_args([*base, "--split", "val", "--preset", "L8"]).preset == "L8"
    for extra in (
        ["--windows", "s0:950-981"],  # no split
        ["--split", "train", "--windows", "s0:1-9"],  # not an evaluation split
        ["--split", "val"],  # no windows
        ["--split", "val", "--preset", "L8", "--windows", "s0:792-823"],  # both
        ["--split", "val", "--preset", "L8", "--precision", "fp16"],
        ["--split", "val", "--preset", "L8", "--mode", "causal"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, *extra])


# ---------------------------------------------------------------- modes, provenance, efficiency (stub models)
def _pose_enc(w2c, height, width):
    from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding

    k = w2c.new_tensor([[50.0, 0, width / 2], [0, 50.0, height / 2], [0, 0, 1]]).expand(*w2c.shape[:2], 3, 3)
    return extri_intri_to_pose_encoding(w2c, k, (height, width))


class OracleModel:
    """``model.inference`` stand-in that returns the GT cameras and GT depth of the frames it is given, with the
    world points unprojected from them and unit confidences."""

    def __init__(self):
        self.calls, self.options = [], []

    def inference(self, images, extrinsics, intrinsics, depth, mask, depth_gt_index, camera_gt_index, **options):
        import torch

        from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.calls.append((images.shape[1], list(depth_gt_index), list(camera_gt_index)))
        self.options.append(options)
        pose_enc = _pose_enc(extrinsics, *images.shape[-2:])
        cameras = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        return {
            "pose_enc": pose_enc,
            "depth": depth.clone(),
            "depth_conf": torch.ones(depth.shape[:-1]),
            "world_points": unproject_depth_to_world_points_torch(depth, *cameras),
        }


class FakeStreaming:
    """``StreamingOmega`` stand-in: camera centres 0.01 m apart along +y (the fake staging's motion)."""

    def __init__(self, model, policy, dtype):
        self.model, self.policy, self.dtype = model, policy, dtype
        self.t, self.resets, self.inputs = 0, 0, []

    def reset(self, max_frames=None):
        self.t, self.resets = 0, self.resets + 1

    def step(self, image, depth, mask):
        import torch

        self.inputs.append(tuple(None if x is None else tuple(x.shape) for x in (image, depth, mask)))
        w2c = torch.eye(3, 4).expand(1, 1, 3, 4).clone()
        w2c[..., 1, 3] = -0.01 * self.t
        self.t += 1
        out_depth = depth[:, None] if depth is not None else torch.ones(1, 1, *image.shape[-2:], 1)
        return {"pose_enc": _pose_enc(w2c, *image.shape[-2:]), "depth": out_depth, "depth_conf": None}

    def kv_bytes(self):
        return 100 * self.t


@pytest.fixture
def fake_stream_api(monkeypatch):
    import sys
    import types

    class CachePolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def full(cls):
            return cls(full=True)

    package = types.ModuleType("omnivggt.stream")
    package.__path__ = []
    streaming, kv_cache = types.ModuleType("omnivggt.stream.streaming"), types.ModuleType("omnivggt.stream.kv_cache")
    streaming.StreamingOmega, kv_cache.CachePolicy = FakeStreaming, CachePolicy
    for module in (package, streaming, kv_cache):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return CachePolicy


POLICY = {"recent": 4, "long_special": 16, "long_patch": 1210, "selector": "query", "quant": "int8"}


def _window(roots, spec="s0:12-19"):
    import eval_stream

    item = eval_stream.WindowLoader(roots, "val", RESOLUTION).load(eval_stream.parse_window(spec))
    return item, eval_stream.window_inputs(item, "cpu")


def test_modes_build_the_pre_registered_models(monkeypatch, tmp_path):
    import eval_colmap_rgbd
    import eval_stream

    from omnivggt.models.omnivggt_omega import OmniVGGTOmega

    assert eval_stream.MODES == {
        "bidir": {"causal": False, "depth_norm": "joint"},
        "bidir_f0": {"causal": False, "depth_norm": "first_frame"},
        "bidir_prefix": {"causal": False, "depth_norm": "first_frame"},
        "stream": {"causal": True, "depth_norm": "first_frame"},
        "causal_batch": {"causal": True, "depth_norm": "first_frame"},
        "bidir_band": {"causal": False, "depth_norm": "first_frame"},
        "g2f": {"causal": False, "depth_norm": "first_frame"},  # --g2f-causal: causal True
        "chunk": {"causal": False, "depth_norm": "first_frame"},  # bidir_f0 per chunk
    }
    built = []
    monkeypatch.setattr(OmniVGGTOmega, "from_variant", staticmethod(lambda path, **options: built.append(options)))
    eval_colmap_rgbd._build_model(tmp_path / "V5.json", **eval_stream.MODES["stream"])
    assert built == [{"causal": True, "depth_norm": "first_frame"}]
    with pytest.raises(ValueError, match="variant"):
        eval_colmap_rgbd._build_model(None, causal=True)


def test_policy_is_required_by_stream_only():
    import json

    import eval_stream

    assert eval_stream.check_options("stream", "full") == "full"
    assert eval_stream.check_options("stream", json.dumps(POLICY)) == POLICY
    assert eval_stream.check_options("bidir", None) is None
    for mode, policy in (("stream", None), ("bidir", "full"), ("bidir_prefix", json.dumps(POLICY)), ("stream", "[4]")):
        with pytest.raises(ValueError, match="policy"):
            eval_stream.check_options(mode, policy)
    with pytest.raises(ValueError):
        eval_stream.check_options("stream", "{recent: 4")


def test_streamer_comes_from_the_stream_api(fake_stream_api):
    import eval_stream
    import torch

    full = eval_stream.make_streamer("model", "full", "fp32")
    assert isinstance(full, FakeStreaming) and full.model == "model" and full.dtype is torch.float32
    assert full.policy.kwargs == {"full": True}
    bounded = eval_stream.make_streamer("model", POLICY, "bf16")
    assert bounded.policy.kwargs == POLICY and bounded.dtype is torch.bfloat16


def test_batch_mode_runs_one_forward_with_all_views_and_no_camera(roots):
    import eval_stream

    _, inputs = _window(roots)
    model, meter = OracleModel(), eval_stream.Meter("cpu")
    depth = eval_stream.predict_batch(model, inputs, True, meter)
    rgb = eval_stream.predict_batch(model, inputs, False, meter)
    assert model.calls == [(8, list(range(8)), []), (8, [], [])]
    assert depth["pose_enc"].shape == (1, 8, 9) and depth["depth"].shape == (1, 8, 42, 56, 1)
    assert len(depth["step_ms"]) == 1 and "kv_bytes" not in depth and rgb["pose_enc"].shape == (1, 8, 9)


def test_prefix_mode_keeps_the_last_frame_of_growing_prefixes(roots):
    import eval_stream
    import torch

    _, inputs = _window(roots)
    model, meter = OracleModel(), eval_stream.Meter("cpu")
    prefix = eval_stream.predict_prefix(model, inputs, True, meter)
    assert model.calls == [(t, list(range(t)), []) for t in range(1, 9)]
    full = eval_stream.predict_batch(OracleModel(), inputs, True, meter)
    assert torch.allclose(prefix["pose_enc"], full["pose_enc"]) and torch.equal(prefix["depth"], full["depth"])
    assert len(prefix["step_ms"]) == 8


def test_stream_mode_steps_one_frame_at_a_time(roots):
    import eval_stream

    _, inputs = _window(roots)
    streamer, meter = FakeStreaming("model", "full", None), eval_stream.Meter("cpu")
    depth = eval_stream.predict_stream(streamer, inputs, True, meter)
    assert streamer.resets == 1 and streamer.inputs == [((1, 3, 42, 56), (1, 42, 56, 1), (1, 42, 56))] * 8
    assert depth["pose_enc"].shape == (1, 8, 9) and depth["kv_bytes"] == [100 * t for t in range(1, 9)]
    streamer.inputs.clear()
    rgb = eval_stream.predict_stream(streamer, inputs, False, meter)
    assert streamer.resets == 2 and streamer.inputs == [((1, 3, 42, 56), None, None)] * 8
    assert rgb["kv_bytes"][0] == 100 and len(rgb["step_ms"]) == 8


def test_stream_step_output_must_hold_one_frame(roots):
    import eval_stream

    class TwoFrames(FakeStreaming):
        def step(self, image, depth, mask):
            out = super().step(image, depth, mask)
            out["pose_enc"] = out["pose_enc"].expand(1, 2, 9)
            return out

    _, inputs = _window(roots)
    with pytest.raises(ValueError, match="one frame"):
        eval_stream.predict_stream(TwoFrames("model", "full", None), inputs, True, eval_stream.Meter("cpu"))


def test_oracle_predictions_score_zero_and_efficiency_is_recorded(roots):
    import contextlib
    import functools

    import eval_stream

    item, _ = _window(roots)
    meter = eval_stream.Meter("cpu")
    batch = functools.partial(eval_stream.predict_batch, OracleModel(), meter=meter)
    record = eval_stream.evaluate_window(batch, item, ["depth", "rgb"], "cpu", contextlib.nullcontext())
    metrics = record["depth"]["metrics"]
    assert metrics["ate_g1_rmse_mm"] < 1e-3 and metrics["abs_rel_s1"] == pytest.approx(0.0, abs=1e-6)
    assert record["depth"]["efficiency"]["step_ms"]["n"] == 1 and "kv_bytes_max" not in record["depth"]["efficiency"]
    assert set(record) == {"depth", "rgb"} and len(record["rgb"]["series"]["step_ms"]) == 1
    stream = functools.partial(eval_stream.predict_stream, FakeStreaming("model", "full", None), meter=meter)
    streamed = eval_stream.evaluate_window(stream, item, ["depth"], "cpu", contextlib.nullcontext())["depth"]
    assert streamed["metrics"]["ate_g1_rmse_mm"] < 1e-3
    assert streamed["efficiency"]["kv_bytes_max"] == 800 and streamed["series"]["kv_bytes"][-1] == 800
    assert streamed["efficiency"]["step_ms"]["n"] == 8 and streamed["efficiency"]["total_ms"] >= 0


def test_precision_turns_tf32_off_and_bf16_autocasts(monkeypatch):
    import eval_stream
    import torch

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    context, record = eval_stream.precision_setup("fp32", "cpu")
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert record == {"name": "fp32", "dtype": "float32", "autocast": None, "tf32_matmul": False, "tf32_cudnn": False}
    with context:
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32
    context, record = eval_stream.precision_setup("bf16", "cpu")
    assert record["autocast"] == "bfloat16" and record["dtype"] == "bfloat16" and not record["tf32_cudnn"]
    for _ in range(2):  # one context serves every window
        with context:
            assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32
    with pytest.raises(ValueError, match="precision"):
        eval_stream.precision_setup("fp16", "cpu")


def test_meter_uses_perf_counter_and_no_memory_on_cpu():
    import eval_stream

    meter = eval_stream.Meter("cpu")
    meter.start()
    value, milliseconds = meter.timed(lambda: 3)
    assert value == 3 and milliseconds >= 0 and meter.memory() == {}


def test_git_state_reports_commit_and_dirty_flag(tmp_path):
    import subprocess

    import eval_stream

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "a.txt").write_text("a")
    subprocess.run([*git, "add", "a.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "a"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (tmp_path / "untracked.txt").write_text("u")
    assert eval_stream.git_state(tmp_path) == {"commit": head, "dirty": False}
    (tmp_path / "a.txt").write_text("b")
    assert eval_stream.git_state(tmp_path) == {"commit": head, "dirty": True}


def _run_main(monkeypatch, tmp_path, roots, extra):
    import hashlib
    import json

    import eval_stream

    variant = tmp_path / "V9.json"
    variant.write_text('{"name": "V9"}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    built = []

    def load(checkpoint, device, model_config, **options):
        built.append((checkpoint, device, model_config, options))
        return OracleModel(), weights

    monkeypatch.setattr(eval_stream, "_load_model", load)
    output = tmp_path / "result.json"
    argv = ["--model-config", str(variant), "--checkpoint", str(weights), "--roots", *roots, "--split", "val"]
    argv += ["--device", "cpu", "--output", str(output), *extra]
    assert eval_stream.main(argv) == 0
    with pytest.raises(FileExistsError):
        eval_stream.main(argv)
    return json.loads(output.read_text()), built, hashlib.sha256(b"weights").hexdigest()


@pytest.mark.parametrize("mode", ["bidir", "bidir_f0", "bidir_prefix", "stream"])
def test_main_records_provenance_metrics_and_efficiency(monkeypatch, tmp_path, roots, fake_stream_api, mode):
    import eval_stream

    extra = ["--windows", "s0:12-19,s1:20-27", "--mode", mode, "--conditions", "depth", "rgb"]
    extra += ["--policy", "full"] if mode == "stream" else []
    result, built, sha256 = _run_main(monkeypatch, tmp_path, roots, extra)
    assert [options for *_, options in built] == [eval_stream.MODES[mode]]
    provenance = result["provenance"]
    assert provenance["checkpoint"]["sha256"] == sha256 and provenance["model_config"]["variant"] == {"name": "V9"}
    assert provenance["mode"] == mode and provenance["model_options"] == eval_stream.MODES[mode]
    assert provenance["mode_arguments"] == {}
    assert provenance["policy"] == ("full" if mode == "stream" else None)
    assert provenance["precision"]["name"] == "fp32" and provenance["precision"]["tf32_matmul"] is False
    assert len(provenance["git"]["commit"]) == 40 and isinstance(provenance["git"]["dirty"], bool)
    assert result["split"] == "val" and result["roots"] == ["root_a", "root_b"] and result["anchors"] is None
    assert [(w["spec"], w["session"]) for w in result["windows"]] == [("s0:12-19", "s0"), ("s1:20-27", "s1")]
    depth = result["windows"][0]["conditions"]["depth"]
    assert depth["metrics"]["ate_g1_rmse_mm"] < 1e-3 and depth["efficiency"]["step_ms"]["n"] >= 1
    assert ("kv_bytes_max" in depth["efficiency"]) == (mode == "stream")
    assert set(result["aggregate"]["depth"]) == {"s0", "s1", "all"} and "rgb" in result["aggregate"]


def test_main_l8_preset_records_the_eval_colmap_rgbd_anchors(monkeypatch, tmp_path, roots):
    import eval_stream

    result, _, _ = _run_main(monkeypatch, tmp_path, roots, ["--preset", "L8", "--mode", "bidir"])
    windows, anchors = eval_stream.l8_windows(roots, "val", RESOLUTION)
    assert result["anchors"] == anchors and result["preset"] == "L8"
    assert [w["spec"] for w in result["windows"]] == [w.spec for w in windows]
    assert [w["anchor"] for w in result["windows"]] == anchors
    assert "AUC@30" in result["aggregate"]["depth"]["all"] and "abs_rel_median" in result["aggregate"]["depth"]["all"]


def test_mean_over_windows_by_session():
    import stream_metrics

    def record(session, metrics):
        return {"session": session, "conditions": {"depth": {"metrics": metrics}}}

    records = [record("s0", {"a": 1.0, "b": 2.0}), record("s0", {"a": 3.0}), record("s1", {"a": 5.0, "b": 1.0})]
    assert stream_metrics.mean_over_windows(records, "depth") == {
        "s0": {"a": 2.0},
        "s1": {"a": 5.0, "b": 1.0},
        "all": {"a": 3.0},
    }


# ---------------------------------------------------------------- phase-5 modes: causal_batch, bidir_band, g2f
G2F_ARGUMENTS = {"g2f_k": 6, "frame_only_layers": [0, 1, 3, 4, 5], "g2f_causal": False}


def test_g2f_k_names_the_pre_registered_layer_sets():
    import eval_stream

    assert eval_stream.G2F_LAYERS == {3: (0, 1, 3), 6: (0, 1, 3, 4, 5), 9: (0, 1, 3, 4, 5, 7, 8)}
    for k, layers in eval_stream.G2F_LAYERS.items():  # global layers <= k, without the register layers 2, 6, 9
        assert layers == tuple(i for i in range(k + 1) if i not in (2, 6, 9))


def test_mode_arguments_belong_to_their_mode():
    import eval_stream

    check = eval_stream.check_mode_arguments
    for mode in ("bidir", "bidir_f0", "bidir_prefix", "stream", "causal_batch"):
        assert check(mode) == {}
    assert check("bidir_band", band_width=4) == {"band_width": 4}
    assert check("g2f", g2f_k=6) == G2F_ARGUMENTS
    assert check("g2f", g2f_k=3, g2f_causal=True) == {"g2f_k": 3, "frame_only_layers": [0, 1, 3], "g2f_causal": True}
    for mode, arguments, match in (
        ("bidir_band", {}, "band-width"),
        ("bidir_band", {"band_width": 0}, "band-width"),
        ("bidir_f0", {"band_width": 4}, "band-width"),
        ("g2f", {"g2f_k": 3, "band_width": 4}, "band-width"),
        ("g2f", {}, "g2f-k"),
        ("g2f", {"g2f_k": 4}, "g2f-k"),
        ("causal_batch", {"g2f_k": 3}, "g2f-k"),
        ("bidir", {"g2f_causal": True}, "g2f-causal"),
        ("bidir_band", {"band_width": 4, "g2f_causal": True}, "g2f-causal"),
    ):
        with pytest.raises(ValueError, match=match):
            check(mode, **arguments)


def test_g2f_causal_makes_the_g2f_model_frame_causal():
    import eval_stream

    assert eval_stream.model_options("g2f", G2F_ARGUMENTS) == {"causal": False, "depth_norm": "first_frame"}
    causal = {**G2F_ARGUMENTS, "g2f_causal": True}
    assert eval_stream.model_options("g2f", causal) == {"causal": True, "depth_norm": "first_frame"}
    assert eval_stream.model_options("bidir_band", {"band_width": 4}) == eval_stream.MODES["bidir_band"]
    for mode in ("bidir", "stream", "causal_batch"):
        assert eval_stream.model_options(mode, {}) == eval_stream.MODES[mode]


def test_cli_takes_the_mode_arguments():
    import eval_stream

    parser = eval_stream.build_parser()
    base = ["--model-config", "V5.json", "--checkpoint", "ckpt", "--roots", "a", "--split", "val", "--preset", "L8"]
    base += ["--output", "o.json"]
    args = parser.parse_args([*base, "--mode", "g2f", "--g2f-k", "9", "--g2f-causal"])
    assert (args.g2f_k, args.g2f_causal, args.band_width) == (9, True, None)
    args = parser.parse_args([*base, "--mode", "bidir_band", "--band-width", "8"])
    assert (args.band_width, args.g2f_k, args.g2f_causal) == (8, None, False)
    for extra in (["--mode", "g2f", "--g2f-k", "4"], ["--mode", "bidir_band", "--band-width", "wide"]):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, *extra])


def test_batch_modes_pass_their_inference_options(roots):
    import eval_stream
    import torch

    from omnivggt.stream.visibility import band_visibility

    _, inputs = _window(roots)
    model, meter = OracleModel(), eval_stream.Meter("cpu")
    eval_stream.make_predictor("bidir_band", model, None, "fp32", meter, {"band_width": 2})(inputs, True)
    eval_stream.make_predictor("g2f", model, None, "fp32", meter, G2F_ARGUMENTS)(inputs, False)
    eval_stream.make_predictor("causal_batch", model, None, "fp32", meter)(inputs, True)
    band, g2f, causal = model.options
    assert list(band) == ["frame_visibility"] and torch.equal(band["frame_visibility"], band_visibility(8, 2))
    assert g2f == {"frame_only_layers": [0, 1, 3, 4, 5]} and causal == {}
    assert model.calls == [(8, list(range(8)), []), (8, [], []), (8, list(range(8)), [])]
    for mode in ("bidir_band", "g2f"):
        with pytest.raises(ValueError, match="argument"):
            eval_stream.make_predictor(mode, model, None, "fp32", meter)


@pytest.mark.parametrize("use_depth", [True, False])
def test_causal_batch_mode_equals_the_full_cache_stream(use_depth):
    import eval_stream
    import torch
    from test_stream_causal import _inputs, _model

    model, inputs = _model(causal=True, depth_norm="first_frame"), _inputs()
    meter = eval_stream.Meter("cpu")
    batch = eval_stream.make_predictor("causal_batch", model, None, "fp32", meter)(inputs, use_depth)
    stream = eval_stream.make_predictor("stream", model, "full", "fp32", meter)(inputs, use_depth)
    assert len(batch["step_ms"]) == 1 and "kv_bytes" not in batch and len(stream["step_ms"]) == 4
    # a fp32 stream cache under the fp64 tiny model, and fp32 predictions: the tolerance is fp32 rounding
    torch.testing.assert_close(batch["pose_enc"], stream["pose_enc"], atol=1e-6, rtol=0)
    torch.testing.assert_close(batch["depth"], stream["depth"], atol=1e-6, rtol=1e-6)


def test_band_and_g2f_modes_run_the_model_with_their_options():
    import eval_stream
    import torch
    from test_stream_causal import _inputs, _model

    from omnivggt.stream.visibility import band_visibility

    model, inputs = _model(depth_norm="first_frame"), _inputs()
    meter = eval_stream.Meter("cpu")
    tiny_g2f = {"g2f_k": 3, "frame_only_layers": [0, 2], "g2f_causal": False}  # TINY's layer 1 is a register layer
    band = eval_stream.make_predictor("bidir_band", model, None, "fp32", meter, {"band_width": 1})(inputs, True)
    g2f = eval_stream.make_predictor("g2f", model, None, "fp32", meter, tiny_g2f)(inputs, True)
    views = {"depth_gt_index": [0, 1, 2, 3], "camera_gt_index": []}
    with torch.no_grad():
        want_band = model.inference(**inputs, **views, frame_visibility=band_visibility(4, 1))
        want_g2f = model.inference(**inputs, **views, frame_only_layers=[0, 2])
        default = model.inference(**inputs, **views)
    for got, want in ((band, want_band), (g2f, want_g2f)):
        assert torch.equal(got["pose_enc"], want["pose_enc"].float())
        assert torch.equal(got["depth"], want["depth"].float())
        assert not torch.equal(got["pose_enc"], default["pose_enc"].float())


@pytest.mark.parametrize(
    "mode, extra, arguments, options",
    [
        ("causal_batch", [], {}, {"causal": True, "depth_norm": "first_frame"}),
        ("bidir_band", ["--band-width", "4"], {"band_width": 4}, {"causal": False, "depth_norm": "first_frame"}),
        ("g2f", ["--g2f-k", "6"], G2F_ARGUMENTS, {"causal": False, "depth_norm": "first_frame"}),
        (
            "g2f",
            ["--g2f-k", "3", "--g2f-causal"],
            {"g2f_k": 3, "frame_only_layers": [0, 1, 3], "g2f_causal": True},
            {"causal": True, "depth_norm": "first_frame"},
        ),
    ],
)
def test_main_records_the_mode_arguments(monkeypatch, tmp_path, roots, mode, extra, arguments, options):
    result, built, _ = _run_main(monkeypatch, tmp_path, roots, ["--windows", "s0:12-19", "--mode", mode, *extra])
    assert [model_options for *_, model_options in built] == [options]
    provenance = result["provenance"]
    assert provenance["mode"] == mode and provenance["mode_arguments"] == arguments
    assert provenance["model_options"] == options and provenance["policy"] is None
    assert result["windows"][0]["conditions"]["depth"]["metrics"]["ate_g1_rmse_mm"] < 1e-3


def test_main_rejects_the_arguments_of_other_modes(tmp_path):
    import eval_stream

    output = tmp_path / "result.json"
    argv = ["--model-config", "V5.json", "--checkpoint", "ckpt", "--roots", "a", "--split", "val", "--preset", "L8"]
    argv += ["--device", "cpu", "--output", str(output)]
    for extra, match in (
        (["--mode", "bidir_f0", "--band-width", "4"], "band-width"),
        (["--mode", "g2f"], "g2f-k"),
        (["--mode", "stream", "--policy", "full", "--g2f-causal"], "g2f-causal"),
    ):
        with pytest.raises(ValueError, match=match):
            eval_stream.main([*argv, *extra])
    assert not output.exists()


def test_main_records_the_vram_limit_and_the_peaks(monkeypatch, tmp_path, roots):
    import eval_stream

    limits = []
    monkeypatch.setattr(eval_stream, "apply_vram_limit", lambda device: limits.append(device) or {"limit_gb": 8.0})
    result, _, _ = _run_main(monkeypatch, tmp_path, roots, ["--windows", "s0:12-19", "--mode", "bidir"])
    assert limits[0] == "cpu" and result["provenance"]["vram"] == {"limit": {"limit_gb": 8.0}, "peaks": None}


def test_main_refuses_a_vram_limit_on_the_cpu_before_reading_anything(monkeypatch, tmp_path):
    import eval_stream

    monkeypatch.setenv("OMNIVGGT_VRAM_LIMIT_GB", "8")
    monkeypatch.setattr(eval_stream, "_load_model", lambda *args, **kwargs: pytest.fail("the model was loaded"))
    argv = ["--model-config", "V5.json", "--checkpoint", "ckpt", "--roots", "missing", "--split", "val", "--preset",
            "L8", "--mode", "bidir", "--device", "cpu", "--output", str(tmp_path / "result.json")]
    with pytest.raises(ValueError, match="OMNIVGGT_VRAM_LIMIT_GB"):
        eval_stream.main(argv)


# ---------------------------------------------------------------- phase-5 chunk mode (VGGT-Long style, offline)
CHUNK_ARGUMENTS = {"chunk": 4, "overlap": 2}
ALIGNMENT = {"pixel_step": 4, "min_weight_ratio": 0.1, "irls_iterations": 5, "huber_delta": "median_initial_residual"}
GAUGE_AXIS = np.array([0.3, 1.0, -0.4]) / np.linalg.norm([0.3, 1.0, -0.4])


def _gauge(chunk):
    """The similarity (scale, rotation, translation) from the GT world to the world of the ``chunk``-th call."""
    rotation = Rotation.from_rotvec(np.deg2rad(4.0 * (chunk + 1)) * GAUGE_AXIS).as_matrix()
    return 0.5 + 0.25 * chunk, rotation, np.array([0.1 * chunk, -0.05, 0.2])


def _in_gauge(c2w, scale, rotation, translation):
    out = c2w.copy()
    out[..., :3, :3] = rotation @ c2w[..., :3, :3]
    out[..., :3, 3] = scale * c2w[..., :3, 3] @ rotation.T + translation
    return out


def _gauge_cameras(extrinsics, chunk, size_hw):
    """(w2c, intrinsics) of GT ``extrinsics`` (1, S, 3, 4) in the gauge of the ``chunk``-th call, focal 50 + chunk."""
    import torch
    from rgbd_pose_pipeline.se3 import invert

    c2w = _in_gauge(invert(extrinsics[0].double().numpy()), *_gauge(chunk))
    w2c = torch.from_numpy(invert(c2w)[None, :, :3]).float()
    focal, (height, width) = 50.0 + chunk, size_hw
    k = torch.tensor([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    return w2c, k.expand(*w2c.shape[:2], 3, 3)


class ChunkModel:
    """``model.inference`` stand-in whose j-th call predicts the GT cameras, depth and world points of its frames in
    its own gauge (``_gauge(j)``) with confidences 2 + j. Its pose encoding carries focal 50 + j, so that the output
    shows which chunk's intrinsics it keeps; the world points are the GT ones (GT intrinsics) in the gauge."""

    def __init__(self):
        self.calls, self.options, self.outputs = [], [], []

    def inference(self, images, extrinsics, intrinsics, depth, mask, depth_gt_index, camera_gt_index, **options):
        import torch

        from omnivggt.utils.geometry import unproject_depth_to_world_points_torch
        from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding

        chunk, size_hw = len(self.calls), images.shape[-2:]
        self.calls.append((images.shape[1], list(depth_gt_index), list(camera_gt_index)))
        self.options.append(options)
        w2c, k = _gauge_cameras(extrinsics, chunk, size_hw)
        predicted_depth = depth * _gauge(chunk)[0]
        out = {
            "pose_enc": extri_intri_to_pose_encoding(w2c, k, size_hw),
            "depth": predicted_depth,
            "depth_conf": torch.full(depth.shape[:-1], 2.0 + chunk),
            "world_points": unproject_depth_to_world_points_torch(predicted_depth, w2c, intrinsics),
        }
        self.outputs.append(out)
        return out


def test_chunk_arguments_belong_to_the_chunk_mode():
    import eval_stream

    check = eval_stream.check_mode_arguments
    assert check("chunk", chunk=12, overlap=6) == {"chunk": 12, "overlap": 6, "alignment": ALIGNMENT}
    assert check("chunk", chunk=24, overlap=12)["overlap"] == 12
    assert check("chunk", chunk=5, overlap=3)["chunk"] == 5  # any valid pair is accepted
    assert eval_stream.model_options("chunk", CHUNK_ARGUMENTS) == {"causal": False, "depth_norm": "first_frame"}
    for mode, arguments, match in (
        ("chunk", {}, "chunk"),
        ("chunk", {"chunk": 12}, "overlap"),
        ("chunk", {"overlap": 6}, "chunk"),
        ("chunk", {"chunk": 12, "overlap": 12}, "overlap"),
        ("chunk", {"chunk": 12, "overlap": 0}, "overlap"),
        ("chunk", {"chunk": 1, "overlap": 0}, "chunk"),
        ("chunk", {"chunk": 12, "overlap": 6, "band_width": 4}, "band-width"),
        ("chunk", {"chunk": 12, "overlap": 6, "g2f_k": 3}, "g2f-k"),
        ("bidir_f0", {"chunk": 12, "overlap": 6}, "chunk"),
        ("bidir_f0", {"overlap": 6}, "overlap"),
        ("g2f", {"g2f_k": 3, "chunk": 12, "overlap": 6}, "chunk"),
    ):
        with pytest.raises(ValueError, match=match):
            check(mode, **arguments)


def test_cli_takes_the_chunk_arguments():
    import eval_stream

    parser = eval_stream.build_parser()
    base = ["--model-config", "V5.json", "--checkpoint", "ckpt", "--roots", "a", "--split", "val", "--windows"]
    base += ["s0:792-855", "--output", "o.json", "--mode", "chunk"]
    args = parser.parse_args([*base, "--chunk", "12", "--overlap", "6"])
    assert (args.chunk, args.overlap, args.band_width, args.g2f_k) == (12, 6, None, None)
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--chunk", "twelve", "--overlap", "6"])


def test_chunk_mode_chains_the_chunks_into_the_first_chunk_frame(roots):
    import eval_stream
    import torch
    from rgbd_pose_pipeline.se3 import invert

    from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

    _, inputs = _window(roots)  # 8 frames: chunks 0-3, 2-5, 4-7
    model, meter = ChunkModel(), eval_stream.Meter("cpu")
    predict = eval_stream.make_predictor("chunk", model, None, "fp32", meter, CHUNK_ARGUMENTS)
    prediction = predict(inputs, True)
    assert model.calls == [(4, [0, 1, 2, 3], [])] * 3 and model.options == [{}] * 3
    for index, start in enumerate([0, 2, 4]):  # each chunk is the window's frames start..start+3
        want = inputs["depth"][:, start : start + 4] * _gauge(index)[0]
        torch.testing.assert_close(model.outputs[index]["depth"], want)
    owners = [0, 0, 0, 1, 1, 2, 2, 2]
    assert prediction["chunks"]["starts"] == [0, 2, 4] and prediction["chunks"]["owners"] == owners
    assert prediction["pose_enc"].shape == (1, 8, 9) and prediction["depth"].shape == (1, 8, 42, 56, 1)
    assert prediction["pose_enc"].dtype == prediction["depth"].dtype == torch.float32

    # frames 1-3 are the first chunk's prediction itself
    first = model.outputs[0]
    assert torch.equal(prediction["pose_enc"][:, :3], first["pose_enc"][:, :3].float())
    assert torch.equal(prediction["depth"][:, :3], first["depth"][:, :3].float())

    # every frame is the GT in the gauge of the first chunk: later chunks are mapped by their Sim(3)
    size_hw = inputs["images"].shape[-2:]
    w2c, _ = _gauge_cameras(inputs["extrinsics"], 0, size_hw)
    for frame, owner in enumerate(owners):
        _, k = _gauge_cameras(inputs["extrinsics"][:, frame : frame + 1], owner, size_hw)  # the owner's intrinsics
        want = extri_intri_to_pose_encoding(w2c[:, frame : frame + 1], k, size_hw)
        torch.testing.assert_close(prediction["pose_enc"][:, frame : frame + 1], want, atol=2e-5, rtol=0)
    torch.testing.assert_close(prediction["depth"], inputs["depth"] * _gauge(0)[0], atol=0, rtol=1e-5)
    pred_w2c, _ = pose_encoding_to_extri_intri(prediction["pose_enc"], size_hw)
    np.testing.assert_allclose(
        invert(pred_w2c[0].double().numpy())[:, :3, 3],
        _in_gauge(invert(inputs["extrinsics"][0].double().numpy()), *_gauge(0))[:, :3, 3],
        atol=2e-5,
    )
    for alignment, chunk in zip(prediction["chunks"]["alignments"], (1, 2), strict=True):
        assert alignment["scale"] == pytest.approx(_gauge(0)[0] / _gauge(chunk)[0], rel=1e-5)
        assert alignment["start"] == 2 * chunk and alignment["correspondences"] > 0
        assert alignment["rotation_deg"] == pytest.approx(4.0 * chunk, abs=1e-3)  # the gauges differ by 4 deg a chunk
    assert prediction["lookahead_frames"] == 2 and prediction["frame_lookahead"] == [3, 2, 1, 2, 1, 2, 1, 0]
    assert len(prediction["step_ms"]) == 3 and prediction["memory"] == {}


def test_chunk_mode_needs_valid_overlap_points(roots):
    import eval_stream

    _, inputs = _window(roots)
    inputs["mask"][:, 2:4] = 0  # the overlap of chunks 0-3 and 2-5
    predict = eval_stream.make_predictor("chunk", ChunkModel(), None, "fp32", eval_stream.Meter("cpu"), CHUNK_ARGUMENTS)
    with pytest.raises(ValueError, match="overlap"):
        predict(inputs, True)
    with pytest.raises(ValueError, match="argument"):
        eval_stream.make_predictor("chunk", ChunkModel(), None, "fp32", eval_stream.Meter("cpu"))


def test_chunk_mode_refuses_windows_shorter_than_a_chunk(roots):
    import eval_stream

    _, inputs = _window(roots)
    predict = eval_stream.make_predictor(
        "chunk", ChunkModel(), None, "fp32", eval_stream.Meter("cpu"), {"chunk": 12, "overlap": 6}
    )
    with pytest.raises(ValueError, match="frames"):
        predict(inputs, True)


def test_chunk_mode_on_the_tiny_model_is_bidir_f0_within_the_first_chunk():
    import functools

    import eval_stream
    import torch
    from test_stream_causal import _inputs, _model

    model, inputs = _model(depth_norm="first_frame"), _inputs(frames=6)
    meter = eval_stream.Meter("cpu")
    predictor = functools.partial(eval_stream.make_predictor, model=model, policy=None, precision="fp32", meter=meter)
    with torch.no_grad():
        whole = predictor("chunk", arguments={"chunk": 6, "overlap": 3})(inputs, True)  # one chunk: the window
        batch = predictor("bidir_f0")(inputs, True)
        chunked = predictor("chunk", arguments=CHUNK_ARGUMENTS)(inputs, True)  # chunks 0-3 and 2-5
        first = eval_stream.predict_batch(model, {key: value[:, :4] for key, value in inputs.items()}, True, meter)
    assert torch.equal(whole["pose_enc"], batch["pose_enc"]) and torch.equal(whole["depth"], batch["depth"])
    assert whole["chunks"]["alignments"] == []
    assert chunked["chunks"]["starts"] == [0, 2] and chunked["chunks"]["owners"] == [0, 0, 0, 1, 1, 1]
    assert torch.equal(chunked["pose_enc"][:, :3], first["pose_enc"][:, :3])
    assert torch.equal(chunked["depth"][:, :3], first["depth"][:, :3])
    (alignment,) = chunked["chunks"]["alignments"]
    assert alignment["correspondences"] > 0 and np.isfinite(alignment["scale"]) and alignment["scale"] > 0
    assert torch.isfinite(chunked["pose_enc"]).all() and torch.isfinite(chunked["depth"]).all()


def test_main_records_the_chunk_mode(monkeypatch, tmp_path, roots):
    extra = ["--windows", "s0:12-19,s1:20-27", "--mode", "chunk", "--chunk", "4", "--overlap", "2"]
    result, built, _ = _run_main(monkeypatch, tmp_path, roots, [*extra, "--conditions", "depth", "rgb"])
    assert [options for *_, options in built] == [{"causal": False, "depth_norm": "first_frame"}]
    provenance = result["provenance"]
    assert provenance["mode"] == "chunk" and provenance["policy"] is None
    assert provenance["mode_arguments"] == {**CHUNK_ARGUMENTS, "alignment": ALIGNMENT}
    for window in result["windows"]:
        for condition in ("depth", "rgb"):
            record = window["conditions"][condition]
            assert record["metrics"]["ate_g1_rmse_mm"] < 1e-3  # the oracle's chunks share one world
            efficiency = record["efficiency"]
            assert efficiency["lookahead_frames"] == 2 and efficiency["lookahead_frames_max"] == 3
            assert efficiency["step_ms"]["n"] == 3 and efficiency["per_frame_ms"] >= 0
            assert record["series"]["lookahead_frames"] == [3, 2, 1, 2, 1, 2, 1, 0]
            assert record["chunks"]["starts"] == [0, 2, 4] and len(record["chunks"]["alignments"]) == 2
            assert record["chunks"]["alignments"][0]["scale"] == pytest.approx(1.0, abs=1e-5)


def test_main_checks_the_window_length_before_loading_the_model(monkeypatch, tmp_path, roots):
    import eval_stream

    def load(*args, **kwargs):
        raise AssertionError("the model must not be loaded")

    monkeypatch.setattr(eval_stream, "_load_model", load)
    output = tmp_path / "result.json"
    argv = ["--model-config", "V9.json", "--checkpoint", "ckpt", "--roots", *roots, "--split", "val"]
    argv += ["--windows", "s0:12-19", "--device", "cpu", "--output", str(output)]
    with pytest.raises(ValueError, match="frames"):
        eval_stream.main([*argv, "--mode", "chunk", "--chunk", "12", "--overlap", "6"])
    with pytest.raises(ValueError, match="overlap"):
        eval_stream.main([*argv, "--mode", "bidir_f0", "--overlap", "6"])
    assert not output.exists()
