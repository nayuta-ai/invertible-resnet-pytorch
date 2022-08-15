import sys

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader
from models import get_model


def main():
    args = parse_args(sys.argv[1:])
    trainloader, testloader = get_dataloader(args)
    model = get_model(args)
    print(model)


if __name__ == "__main__":
    main()
