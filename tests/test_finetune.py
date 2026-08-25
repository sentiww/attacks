import pytest
from torch import nn

from training.finetune import freeze_before, validate_checkpoint_compatibility


class LayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(2, 2)
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)


def test_freeze_before_freezes_only_preceding_top_level_modules() -> None:
    model = LayeredModel()
    freeze_before(model, "layer2")
    assert all(not parameter.requires_grad for parameter in model.stem.parameters())
    assert all(not parameter.requires_grad for parameter in model.layer1.parameters())
    assert all(parameter.requires_grad for parameter in model.layer2.parameters())


def test_freeze_before_none_is_noop_and_unknown_module_errors() -> None:
    model = LayeredModel()
    freeze_before(model, None)
    assert all(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError, match="not found"):
        freeze_before(model, "missing")


def test_checkpoint_compatibility_accepts_matching_or_absent_config() -> None:
    current = {"model": {"name": "resnet18"}, "dataset": {"name": "cifar10"}}
    validate_checkpoint_compatibility(None, current)
    validate_checkpoint_compatibility({}, current)
    validate_checkpoint_compatibility(current, current)


@pytest.mark.parametrize("section", ["model", "dataset"])
def test_checkpoint_compatibility_rejects_mismatch(section: str) -> None:
    previous = {"model": {"name": "resnet18"}, "dataset": {"name": "cifar10"}}
    current = {"model": {"name": "resnet18"}, "dataset": {"name": "cifar10"}}
    current[section]["name"] = "different"
    with pytest.raises(ValueError, match=section):
        validate_checkpoint_compatibility(previous, current)
