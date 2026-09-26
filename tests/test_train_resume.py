"""Resuming training: data order of each epoch and where to restart inside an epoch."""

from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from accelerate.data_loader import prepare_data_loader

import train_utils
from omnivggt.datasets import get_data_loader
from omnivggt.datasets.base.easy_dataset import EasyDataset

ROOT = Path(__file__).resolve().parents[1]


class _Frames(EasyDataset):
    _resolutions = ((28, 28),)

    def __len__(self):
        return 50

    def __getitem__(self, idx):
        index, _, views = idx
        return torch.tensor([index, views])


def _loader():
    PartialState()
    loader = get_data_loader(24 @ _Frames(), batch_size=4, num_workers=0, shuffle=True, drop_last=True)
    return prepare_data_loader(loader, put_on_device=False)  # what accelerator.prepare builds (a DataLoaderShard)


def _epoch(loader, epoch):
    train_utils.start_epoch(loader, epoch)
    return [tuple(torch.cat([t.flatten() for t in batch]).tolist()) for batch in loader]


def test_resumed_epoch_sees_the_data_of_an_uninterrupted_run():
    uninterrupted = _loader()
    first, second = _epoch(uninterrupted, 0), _epoch(uninterrupted, 1)
    assert first != second
    assert _epoch(_loader(), 1) == second  # a fresh process resuming at epoch 1


@pytest.mark.parametrize(
    "global_step, per_epoch, accum, expected",
    [(0, 5, 3, (0, 0)), (2, 5, 3, (2, 6)), (5, 5, 3, (0, 0)), (7, 5, 3, (2, 6)), (3120, 1560, 2, (0, 0)), (1561, 1560, 1, (1, 1))],
)
def test_resume_skip_counts_micro_batches(global_step, per_epoch, accum, expected):
    assert train_utils.resume_skip(global_step, per_epoch, accum) == expected


def test_training_loop_uses_the_helpers():
    source = (ROOT / "train_omnivggt.py").read_text()
    assert "start_epoch(train_dataloader, epoch)" in source
    assert "resume_skip(global_step, local_steps_per_epoch, accumulation_steps)" in source
    assert "itertools.islice(train_dataloader, skip_micro_batches, None)" in source
    assert "enumerate(train_iter, start=skip_micro_batches)" in source


@pytest.mark.parametrize("batches, accum, steps", [(1560, 1, 1560), (3120, 2, 1560), (4680, 3, 1560)])
def test_optimizer_steps_per_epoch(batches, accum, steps):
    assert train_utils.optimizer_steps_per_epoch(batches, accum) == steps


@pytest.mark.parametrize("batches, accum", [(5, 3), (1000, 3), (2, 3), (1, 2), (10, 0)])
def test_optimizer_steps_per_epoch_rejects_remainders_and_short_epochs(batches, accum):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_utils.optimizer_steps_per_epoch(batches, accum)


def test_training_loop_uses_one_accumulation_setting():
    source = (ROOT / "train_omnivggt.py").read_text()
    assert "optimizer_steps_per_epoch(actual_local_batches, gradient_accumulation_steps)" in source
    assert 'cfg.get("gradient_accumulation_steps", 2)' not in source
