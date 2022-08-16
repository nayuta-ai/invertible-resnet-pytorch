import sys

from config import parse_args
from data.get_dataloader import get_dataloader


def main():
    args = parse_args(sys.argv[1:])
    trainloader, testloader = get_dataloader(args)
    print(args.nClasses)


if __name__ == "__main__":
    main()
