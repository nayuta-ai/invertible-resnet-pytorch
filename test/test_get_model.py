from typing import Tuple

import pytest
import torch.nn as nn

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader
from models import get_model


@pytest.mark.parametrize("dataset, num_class", [("cifar10", 10), ("cifar100", 100)])
def test_get_dataloader(dataset: str, num_class: int) -> None:
    """test get_model function
    Args:
        dataset (str): the kind of dataset
        num_class (int): the number of correct classes to test the number of classes in the dataset
    """
    args = parse_args(["--dataset", dataset])
    get_dataloader(args)
    model = get_model(args)
    assert isinstance(
        model, nn.Module
    ), "The type of the output from get_model should be nn.Module."
    assert (
        model.nClasses == num_class
    ), f"The output class should be {num_class}, but the output class is {model.nClasses}."
