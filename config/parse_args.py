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
    ## dataset
    parser.add_argument(
        "-densityEstimation",
        "--densityEstimation",
        dest="densityEstimation",
        action="store_false",
        help="perform density estimation",
    )
    parser.add_argument(
        "-deterministic",
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="fix random seeds and set cuda deterministic",
    )
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset")
    ## model
    parser.add_argument(
        "-multiScale",
        "--multiScale",
        dest="multiScale",
        action="store_true",
        help="use multiscale",
    )
    parser.add_argument("--nBlocks", nargs="+", type=int, default=[4, 4, 4])
    parser.add_argument("--nStrides", nargs="+", type=int, default=[1, 2, 2])
    parser.add_argument("--nChannels", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--init_ds", default=2, type=int, help="initial downsampling")
    parser.add_argument("--inj_pad", default=0, type=int, help="initial inj padding")
    parser.add_argument(
        "--coeff",
        default=0.9,
        type=float,
        help="contraction coefficient for linear layers",
    )
    parser.add_argument(
        "--numTraceSamples",
        default=1,
        type=int,
        help="number of samples used for trace estimation",
    )
    parser.add_argument(
        "--numSeriesTerms",
        default=1,
        type=int,
        help="number of terms used in power series for matrix log",
    )
    parser.add_argument(
        "--powerIterSpectralNorm",
        default=5,
        type=int,
        help="number of power iterations used for spectral norm",
    )
    parser.add_argument(
        "-fixedPrior",
        "--fixedPrior",
        dest="fixedPrior",
        action="store_true",
        help="use fixed prior, default is learned prior",
    )
    parser.add_argument(
        "-noActnorm",
        "--noActnorm",
        dest="noActnorm",
        action="store_true",
        help="disable actnorm, default uses actnorm",
    )
    parser.add_argument(
        "--nonlin",
        default="elu",
        type=str,
        choices=["relu", "elu", "sorting", "softplus"],
    )
    # optimizer
    parser.add_argument(
        "--optimizer",
        default="adamax",
        type=str,
        help="optimizer",
        choices=["adam", "adamax", "sgd"],
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="coefficient for weight decay"
    )
    parser.add_argument(
        "-nesterov",
        "--nesterov",
        dest="nesterov",
        action="store_true",
        help="nesterov momentum",
    )
    # train
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "-interpolate",
        "--interpolate",
        dest="interpolate",
        action="store_true",
        help="train iresnet",
    )
    parser.add_argument("--batch", default=128, type=int, help="batch size")
    parser.add_argument("--drop_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument(
        "--warmup_epochs", default=10, type=int, help="epochs for warmup"
    )
    parser.add_argument("--log_every", default=10, type=int, help="logs every x iters")
    parser.add_argument(
        "-log_verbose",
        "--log_verbose",
        dest="log_verbose",
        action="store_true",
        help="verbose logging: sigmas, max gradient",
    )
    parser.add_argument(
        "--save_dir", default="./results", type=str, help="directory to save results"
    )
    return parser.parse_args(args)
