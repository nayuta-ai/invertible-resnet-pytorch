import argparse
import sys

from typed_argparse import TypedArgs


class MyArgs(TypedArgs):
    """
    class of providing a way to annotate python3 function
    with type information on args
    """

    lr: float
    batch: int
    epochs: int
    dataset: str
    densityEstimation: bool
    deterministic: bool


def parse_args(args: sys.argv) -> MyArgs:
    """Create parser
    Args:
        args (sys.argv): the argument in the command line

    Returns:
        MyArgs: the object converted by the argument in the command line

    Explanation:
        --densityEstimation (bool): true if it performs density estimation
        --deterministic (bool): true if it fixes random seeds and set cuda deterministic
        --lr (float): learning rate
        --batch (int): batch size
        --epoch (int): the number of epoch
        --dataset (str): the kind of dataset
    """
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
