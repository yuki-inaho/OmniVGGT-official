"""AnchorFrameSampler: images per step -> (anchors, views per anchor)."""

import pytest

from omnivggt.datasets.base.batched_sampler import AnchorFrameSampler


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _anchor_counts(images_per_step, n=400):
    sampler = AnchorFrameSampler(_Sized(n), images_per_step, pool_size=1)
    sampler.set_epoch(0)
    items = list(sampler)
    assert len(items) == n
    assert all(item[-1] == images_per_step and item[-2] == 0 for item in items)
    return {len(item) - 2 for item in items}


@pytest.mark.parametrize(
    "images, anchors",
    [(12, {1, 2, 4, 6}), (6, {1, 2, 3}), (8, {1, 2, 4}), (4, {1, 2})],
)
def test_anchor_counts_keep_at_least_two_views(images, anchors):
    counts = _anchor_counts(images)
    assert counts == anchors
    assert all(images % a == 0 and images // a >= 2 for a in counts)


def test_unsupported_image_count_raises_value_error():
    with pytest.raises(ValueError, match="5"):
        _anchor_counts(5)
