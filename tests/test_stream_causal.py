"""Frame-causal (batch) OmniVGGTOmega: the inter-frame mask, the aggregator, the camera head and the wiring."""

import pytest
import torch

from omnivggt.stream.masks import frame_causal_mask


def test_frame_causal_mask():
    got = frame_causal_mask(3, 2, torch.device("cpu"))
    want = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert got.dtype == torch.bool and got.device.type == "cpu"
    assert torch.equal(got, want)


def test_frame_causal_mask_of_one_token_per_frame_is_lower_triangular():
    assert torch.equal(frame_causal_mask(4, 1, "cpu"), torch.ones(4, 4, dtype=torch.bool).tril())


@pytest.mark.parametrize("frames, tokens", [(0, 2), (3, 0)])
def test_frame_causal_mask_rejects_empty_sizes(frames, tokens):
    with pytest.raises(ValueError):
        frame_causal_mask(frames, tokens, "cpu")
