import torch.nn as nn

from config.parse_args import MyArgs
from models.invertible_resnet import conv_iResNet as iResNet
from models.invertible_resnet import multiscale_conv_iResNet as multiscale_iResNet


def get_model(args: MyArgs) -> nn.Module:
    if args.multiScale:
        model = multiscale_iResNet(
            in_shape=args.in_shape,
            nBlocks=args.nBlocks,
            nStrides=args.nStrides,
            nChannels=args.nChannels,
            init_ds=args.init_ds,
            inj_pad=args.inj_pad,
            coeff=args.coeff,
            densityEstimation=args.densityEstimation,
            nClasses=args.nClasses,
            numTraceSamples=args.numTraceSamples,
            numSeriesTerms=args.numSeriesTerms,
            n_power_iter=args.powerIterSpectralNorm,
            actnorm=(not args.noActnorm),
            learn_prior=(not args.fixedPrior),
            nonlin=args.nonlin,
        )
    else:
        model = iResNet(
            nBlocks=args.nBlocks,
            nStrides=args.nStrides,
            nChannels=args.nChannels,
            nClasses=args.nClasses,
            init_ds=args.init_ds,
            inj_pad=args.inj_pad,
            in_shape=args.in_shape,
            coeff=args.coeff,
            numTraceSamples=args.numTraceSamples,
            numSeriesTerms=args.numSeriesTerms,
            n_power_iter=args.powerIterSpectralNorm,
            density_estimation=args.densityEstimation,
            actnorm=(not args.noActnorm),
            learn_prior=(not args.fixedPrior),
            nonlin=args.nonlin,
        )
    return model
