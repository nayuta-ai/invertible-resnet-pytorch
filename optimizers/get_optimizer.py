import torch.optim as optim

from config.parse_args import TypedArgs


def get_optimizer(args: TypedArgs, model) -> optim:
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    return optimizer
