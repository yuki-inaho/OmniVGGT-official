"""configs/train_colmap_rgbd_omega.py: environment overrides for the high-resolution RGB-D run."""

from pathlib import Path

import pytest

from omnivggt.utils.configs import read_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_colmap_rgbd_omega.py"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OMNIVGGT_COLMAP_RGBD_ROOTS", "/data/a,/data/b")
    monkeypatch.setenv("OMNIVGGT_OMEGA_VARIANT", "configs/omnivggt_omega/variants/V5.json")
    for name in ("OMNIVGGT_RESOLUTION", "OMNIVGGT_DEPTH_DROP_PROB", "OMNIVGGT_GRAD_ACCUM", "OMNIVGGT_INIT_CHECKPOINT",
                 "OMNIVGGT_DEPTH_ALL_VIEWS", "OMNIVGGT_STEPS_PER_EPOCH"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_are_the_existing_recipe(env):
    cfg = read_config(str(CONFIG))
    assert [tuple(r) for r in cfg.resolution] == [(392, 294)]
    assert "resolution=[(392, 294)]" in cfg.train_dataset
    assert cfg.depth_drop_prob == 0.3 and cfg.cam_drop_prob == 0.1 and cfg.depth_all_views is False
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.get("init_checkpoint") is None


def test_high_resolution_rgbd_overrides(env):
    env.setenv("OMNIVGGT_RESOLUTION", "644x476")
    env.setenv("OMNIVGGT_DEPTH_DROP_PROB", "0")
    env.setenv("OMNIVGGT_DEPTH_ALL_VIEWS", "1")
    env.setenv("OMNIVGGT_GRAD_ACCUM", "2")
    env.setenv("OMNIVGGT_INIT_CHECKPOINT", "/runs/prev/exp/final_checkpoint")
    cfg = read_config(str(CONFIG))
    assert [tuple(r) for r in cfg.resolution] == [(644, 476)]
    assert "resolution=[(644, 476)]" in cfg.train_dataset
    assert cfg.depth_drop_prob == 0.0 and cfg.cam_drop_prob == 0.1 and cfg.depth_all_views is True
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.init_checkpoint == "/runs/prev/exp/final_checkpoint"


@pytest.mark.parametrize("value", ["640x480", "644", "644x476x2"])
def test_resolution_must_be_multiples_of_the_patch_size(env, value):
    env.setenv("OMNIVGGT_RESOLUTION", value)
    with pytest.raises(Exception, match="OMNIVGGT_RESOLUTION"):
        read_config(str(CONFIG))


def test_steps_per_epoch_must_be_a_multiple_of_accumulation(env):
    env.setenv("OMNIVGGT_GRAD_ACCUM", "3")  # default steps_per_epoch 1000 leaves a remainder
    with pytest.raises(Exception, match="OMNIVGGT_STEPS_PER_EPOCH"):
        read_config(str(CONFIG))
    env.setenv("OMNIVGGT_STEPS_PER_EPOCH", "4680")
    assert read_config(str(CONFIG)).steps_per_epoch == 4680
