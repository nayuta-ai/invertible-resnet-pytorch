from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.parse_args import MyArgs
from utils.train_cifar import mean, std


def get_dataloader(args: MyArgs) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        args (MyArgs): the object converted by the argument in the command line

    Returns:
        Tuple[DataLoader, DataLoader]: creating the dataloader from the dataset
    """
    transform_train, transform_test = transform(args)
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        args.nClasses = 10
    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        args.nClasses = 100
    args.lenData = len(trainset)
    args.in_shape = (3, 32, 32)

    if args.deterministic:
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=2,
            worker_init_fn=np.random.seed(1234),
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=2,
            worker_init_fn=np.random.seed(1234),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch, shuffle=True, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch, shuffle=False, num_workers=2
        )
    return trainset, testset, trainloader, testloader


def transform(args: MyArgs) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Args:
        args (MyArgs): the object converted by the argument in the command line

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: transforming and augmenting the train and test dataset

    Explanations:
        if densityEstimation
            train
                - padding (the width = 4, padding mode = symmetric)
                - random crop (the size = 32)
                - random horizontal flip
                - transform to Tensor
                - dens_est_chain
            test
                - transform to Tensor
                - dens_est_chain
        else
            train
                - padding (the width = 4, padding mode = symmetric)
                - random crop (the size = 32)
                - random horizontal flip
                - transform to Tensor
                - normalization
            test
                - transform to Tensor
                - normalization
    """
    train_chain = [
        transforms.Pad(4, padding_mode="symmetric"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    test_chain = [transforms.ToTensor()]
    dens_est_chain = [
        lambda x: (255.0 * x) + torch.zeros_like(x).uniform_(0.0, 1.0),
        lambda x: x / 256.0,
        lambda x: x - 0.5,
    ]
    if args.densityEstimation:
        transform_train = transforms.Compose(train_chain + dens_est_chain)
        transform_test = transforms.Compose(test_chain + dens_est_chain)
    else:
        clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
        transform_train = transforms.Compose(train_chain + clf_chain)
        transform_test = transforms.Compose(test_chain + clf_chain)
    return transform_train, transform_test
