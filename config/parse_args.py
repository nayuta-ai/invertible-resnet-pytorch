import argparse

from typed_argparse import TypedArgs


class MyArgs(TypedArgs):
    lr: float
    batch: int
    epochs: int
    dataset: str


def parse_args(args) -> MyArgs:
    parser = argparse.ArgumentParser(description="Configuration for experiments")
    parser.add_argument(
        "-densityEstimation",
        "--densityEstimation",
        dest="densityEstimation",
        action="store_true",
        help="perform density estimation",
    )
    parser.add_argument(
        "-deterministic",
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="fix random seeds and set cuda deterministic",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch", default=128, type=int, help="batch size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset")
    return parser.parse_args(args)
