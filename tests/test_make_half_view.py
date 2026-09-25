import json

import numpy as np
import pytest
from PIL import Image
from rgbd_pose_pipeline.make_half_view import make_half_view


def _write_standard(root, count=5):
    for sub in ("rgb", "depth", "mapped_depth", "camera_parameters"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(1, count + 1):
        stem = f"{index:08d}"
        Image.fromarray(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)).save(root / "rgb" / f"{stem}_rgb.jpg")
        depth = rng.integers(0, 1300, (6, 8), dtype=np.uint16)
        Image.fromarray(depth).save(root / "depth" / f"{stem}_depth.png")
        Image.fromarray(depth).save(root / "mapped_depth" / f"{stem}_depth.png")
    (root / "camera_parameters" / "rgb_camera_param.yaml").write_text("width: 8\nheight: 6\n")
    (root / "camera_parameters" / "depth_camera_param.yaml").write_text("width: 8\nheight: 6\n")


def _write_check(path, passed=True, kernel=3):
    path.write_text(json.dumps({"all_passed": passed, "args": {"max_depth": 1300, "dilation_kernel_size": kernel}}))
    return path


def test_selects_every_second_frame_and_writes_contract(tmp_path):
    standard = tmp_path / "standard"
    _write_standard(standard)
    out = tmp_path / "half"
    manifest = make_half_view(standard, out, step=2, offset=0, mapped_depth_check=_write_check(tmp_path / "c.json"))

    assert manifest["selected_stems"] == ["00000001", "00000003", "00000005"]
    assert sorted(p.name for p in (out / "rgb").iterdir()) == [f"{s}_rgb.png" for s in manifest["selected_stems"]]
    assert sorted(p.name for p in (out / "mapped_depth").iterdir()) == [
        f"{s}_depth.png" for s in manifest["selected_stems"]
    ]
    decoded = np.asarray(Image.open(standard / "rgb" / "00000003_rgb.jpg").convert("RGB"))
    assert np.array_equal(np.asarray(Image.open(out / "rgb" / "00000003_rgb.png")), decoded)
    assert (out / "mapped_depth" / "00000003_depth.png").samefile(standard / "mapped_depth" / "00000003_depth.png")
    readme = (out / "README.md").read_text().lower()
    assert "3x3" in readme and "nearest-depth dilation" in readme
    assert (out / "camera_parameters" / "rgb_camera_param.yaml").is_file()
    assert json.loads((out / "subsample_manifest.json").read_text())["frame_count"] == 3


def test_offset_one(tmp_path):
    standard = tmp_path / "standard"
    _write_standard(standard)
    manifest = make_half_view(
        standard, tmp_path / "half", step=2, offset=1, mapped_depth_check=_write_check(tmp_path / "c.json")
    )
    assert manifest["selected_stems"] == ["00000002", "00000004"]


def test_refuses_existing_output(tmp_path):
    standard = tmp_path / "standard"
    _write_standard(standard)
    (tmp_path / "half").mkdir()
    with pytest.raises(FileExistsError):
        make_half_view(
            standard, tmp_path / "half", step=2, offset=0, mapped_depth_check=_write_check(tmp_path / "c.json")
        )


def test_refuses_missing_mapped_depth(tmp_path):
    standard = tmp_path / "standard"
    _write_standard(standard)
    (standard / "mapped_depth" / "00000003_depth.png").unlink()
    with pytest.raises(FileNotFoundError):
        make_half_view(
            standard, tmp_path / "half", step=2, offset=0, mapped_depth_check=_write_check(tmp_path / "c.json")
        )


@pytest.mark.parametrize("passed,kernel", [(False, 3), (True, 5)])
def test_refuses_unverified_mapped_depth(tmp_path, passed, kernel):
    standard = tmp_path / "standard"
    _write_standard(standard)
    with pytest.raises(ValueError):
        make_half_view(
            standard,
            tmp_path / "half",
            step=2,
            offset=0,
            mapped_depth_check=_write_check(tmp_path / "c.json", passed=passed, kernel=kernel),
        )
