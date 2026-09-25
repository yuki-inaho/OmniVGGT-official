import pytest
import torch
from safetensors.torch import save_file

from train_utils import load_initial_weights


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)


def test_init_checkpoint_is_loaded_strictly(tmp_path):
    source = Tiny()
    save_file(source.state_dict(), tmp_path / "w.safetensors")
    target = Tiny()
    assert (
        load_initial_weights(target, {"init_checkpoint": str(tmp_path / "w.safetensors")})
        == "init_checkpoint:w.safetensors"
    )
    assert torch.equal(target.layer.weight, source.layer.weight)


def test_missing_init_checkpoint_raises_instead_of_training_from_scratch(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_initial_weights(Tiny(), {"init_checkpoint": str(tmp_path / "missing.safetensors")})


def test_mismatched_keys_raise(tmp_path):
    save_file({"other.weight": torch.zeros(2, 3)}, tmp_path / "bad.safetensors")
    with pytest.raises(RuntimeError):
        load_initial_weights(Tiny(), {"init_checkpoint": str(tmp_path / "bad.safetensors")})
