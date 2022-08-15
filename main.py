import sys

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader
from models import get_model
from optimizers.get_optimizer import get_optimizer
from utils.comet_utils import exp
from utils.train_cifar import train


def main():
    args = parse_args(sys.argv[1:])
    trainloader, testloader = get_dataloader(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    experiment = exp(args)
    train(args, model, optimizer, 0, trainloader, experiment, False)


if __name__ == "__main__":
    main()
