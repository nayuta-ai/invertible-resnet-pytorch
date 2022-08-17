import sys

import torch
import torch.backends.cudnn as cudnn

from config.parse_args import parse_args
from data.get_dataloader import get_dataloader
from models import get_model
from optimizers.get_optimizer import get_optimizer
from utils.comet_utils import exp
from utils.train import train


def main():
    args = parse_args(sys.argv[1:])
    trainset, testset, trainloader, testloader = get_dataloader(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    experiment = exp(args)
    use_cuda = torch.cuda.is_available()
    print(args.densityEstimation)
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        cudnn.benchmark = True
        in_shapes = model.module.get_in_shapes()
    else:
        in_shapes = model.get_in_shapes()
    train(
        in_shapes,
        args,
        model,
        optimizer,
        trainloader,
        testloader,
        trainset,
        testset,
        experiment,
        use_cuda,
    )


if __name__ == "__main__":
    main()
