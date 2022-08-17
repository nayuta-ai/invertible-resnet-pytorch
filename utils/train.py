import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config.parse_args import MyArgs
from utils.train_cifar import cifar_test, cifar_train, get_hms


def train(
    in_shapes,
    args: MyArgs,
    model: nn.Module,
    optimizer: optim,
    trainloader: DataLoader,
    testloader: DataLoader,
    trainset,
    testset,
    experiment: Experiment,
    use_cuda: bool,
):
    try_make_dir(args.save_dir)
    """
    if args.analysisTraceEst:
        anaylse_trace_estimation(model, testset, use_cuda, args.extension)
        return

    if args.norm:
        test_spec_norm(model, in_shapes, args.extension)
        return
    """
    best_result = 0
    if args.evaluate:
        if use_cuda:
            model.module.set_num_terms(args.numSeriesTerms)
        else:
            model.set_num_terms(args.numSeriesTerms)
        model = torch.nn.DataParallel(model.module)
        cifar_test(
            best_result,
            args,
            model,
            10,
            testloader,
            experiment,
            use_cuda,
        )
        return

    print("|  Train Epochs: " + str(args.epochs))
    print("|  Initial Learning Rate: " + str(args.lr))

    elapsed_time = 0
    test_objective = -np.inf

    for epoch in range(1, 1 + args.epochs):
        start_time = time.time()
        cifar_train(
            args,
            model,
            optimizer,
            epoch,
            trainloader,
            experiment,
            use_cuda,
        )
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print("| Elapsed time : %d:%02d:%02d" % (get_hms(elapsed_time)))

    print("Testing model")
    test_objective = cifar_test(
        test_objective, args, model, 10, testloader, experiment, use_cuda
    )
    print("* Test results : objective = %.2f%%" % (test_objective))


def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


def anaylse_trace_estimation(model, testset, use_cuda, extension):
    # setup range for analysis
    numSamples = np.arange(10) * 10 + 1
    numIter = np.arange(10)
    # setup number of datapoints
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    # TODO change

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        # compute trace
        out_bij, p_z_g_y, trace, gt_trace = model(
            inputs[:, :, :8, :8], exact_trace=True
        )
        trace = [t.cpu().numpy() for t in trace]
        np.save("gtTrace" + extension, gt_trace)
        np.save("estTrace" + extension, trace)
        return


def test_spec_norm(model, in_shapes, extension):
    i = 0
    j = 0
    params = [
        v
        for v in model.module.state_dict().keys()
        if "bottleneck"
        and "weight" in v
        and not "weight_u" in v
        and not "weight_orig" in v
        and not "bn1" in v
        and not "linear" in v
    ]
    print(len(params))
    print(len(in_shapes))
    svs = []
    for param in params:
        if i == 0:
            input_shape = in_shapes[j]
        else:
            input_shape = in_shapes[j]
            input_shape[1] = int(input_shape[1] // 4)

        convKernel = model.module.state_dict()[param].cpu().numpy()
        input_shape = input_shape[2:]
        fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
        t_fft_coeff = np.transpose(fft_coeff)
        U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
        Dflat = np.sort(D.flatten())[::-1]
        print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
        svs.append(Dflat[0])
        if i == 2:
            i = 0
            j += 1
        else:
            i += 1
    np.save("singular_values" + extension, svs)
    return
