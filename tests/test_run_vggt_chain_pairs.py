import pytest
from rgbd_pose_pipeline.run_vggt_chain import collect_pairs


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_pairs_use_the_explicit_depth_directory(tmp_path):
    for stem in ("00000003", "00000001"):
        _touch(tmp_path / "rgb" / f"{stem}_rgb.png")
        _touch(tmp_path / "mapped_depth" / f"{stem}_depth.png")
        _touch(tmp_path / "mapped_depth_dense" / f"{stem}_depth.png")
    pairs = collect_pairs(tmp_path, "rgb", "_rgb.png", "mapped_depth", "_depth.png")
    assert [(r.name, d.parent.name, d.name) for r, d in pairs] == [
        ("00000001_rgb.png", "mapped_depth", "00000001_depth.png"),
        ("00000003_rgb.png", "mapped_depth", "00000003_depth.png"),
    ]


def test_missing_depth_is_an_error_not_a_skip(tmp_path):
    _touch(tmp_path / "rgb" / "00000001_rgb.png")
    _touch(tmp_path / "rgb" / "00000002_rgb.png")
    _touch(tmp_path / "mapped_depth" / "00000001_depth.png")
    with pytest.raises(FileNotFoundError, match="00000002"):
        collect_pairs(tmp_path, "rgb", "_rgb.png", "mapped_depth", "_depth.png")


def test_no_rgb_with_suffix_is_an_error(tmp_path):
    _touch(tmp_path / "rgb" / "00000001_rgb.jpg")
    _touch(tmp_path / "mapped_depth" / "00000001_depth.png")
    with pytest.raises(FileNotFoundError):
        collect_pairs(tmp_path, "rgb", "_rgb.png", "mapped_depth", "_depth.png")
