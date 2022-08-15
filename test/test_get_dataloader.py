import pytest
from torch.utils.data import DataLoader

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader

"""
def test_transform():
    args = parse_args(['--dataset', "cifar10"])
    transform_train, transform_test = get_dataloader(args)
    assert type(transform_train) == transforms.transforms.Compose
    assert type(transform_test) == transforms.transforms.Compose
"""


@pytest.mark.parametrize(
    "dataset, num_class", [
        ("cifar10", 10),
        ("cifar100", 100)
    ]
)
def test_get_dataloader(dataset: str, num_class: int) -> None:
    """ test get_dataloader function
    Args:
        dataset (str): the kind of dataset
        num_class (int): the number of correct classes to test the number of classes in the dataset
    """
    args = parse_args(["--dataset", dataset])
    trainloader, testloader = get_dataloader(args)
    assert type(trainloader) == DataLoader, f"The type of trainloader should be DataLoader, but the type of trainloader is {type(trainloader)}"
    assert type(testloader) == DataLoader, f"The type of testloader should be DataLoader, but the type of testloader is {type(testloader)}"
    assert args.nClasses == num_class, f"The number of class should be {num_class}, but the number of class is {args.nClasses}."
