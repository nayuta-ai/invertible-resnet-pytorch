import sys

from config import parse_args
from data.get_dataloader import get_dataloader
from models import get_model
from optimizers.get_optimizer import get_optimizer


def main():
    args = parse_args(sys.argv[1:])
    trainloader, testloader = get_dataloader(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)


if __name__ == "__main__":
    main()
