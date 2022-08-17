from typing import Tuple

import pytest
import torch.nn as nn
import torch.optim as optim

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader
from models import get_model
from optimizers.get_optimizer import get_optimizer


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
    optimizer = get_optimizer(args, model)
    assert isinstance(
        optimizer, optim
    ), f"The type of the output should be optim but, the type of the optput is {type(optimizer)}."