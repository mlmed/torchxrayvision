import torch
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable

from .utils import *

pspnet_specs = {
    "pascal": {
        "n_classes": 21,
        "input_size": (473, 473),
        "block_config": [3, 4, 23, 3],
    },
    "cityscapes": {
        "n_classes": 19,
        "input_size": (713, 713),
        "block_config": [3, 4, 23, 3],
    },
    "ade20k": {
        "n_classes": 150,
        "input_size": (473, 473),
        "block_config": [3, 4, 6, 3],
    },
}


class pspnet(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(
        self,
        n_classes=18,
        block_config=[3, 4, 23, 3],
        input_size=(473, 473),
        version=None,
    ):

        super(pspnet, self).__init__()

        self.block_config = (
            pspnet_specs[version]["block_config"]
            if version is not None
            else block_config
        )
        self.n_classes = (
            pspnet_specs[version]["n_classes"] if version is not None else n_classes
        )
        self.input_size = (
            pspnet_specs[version]["input_size"] if version is not None else input_size
        )

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(
            in_channels=3, k_size=3, n_filters=64, padding=1, stride=2, bias=False
        )
        self.convbnrelu1_2 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=64, padding=1, stride=1, bias=False
        )
        self.convbnrelu1_3 = conv2DBatchNormRelu(
            in_channels=64, k_size=3, n_filters=128, padding=1, stride=1, bias=False
        )

        # Vanilla Residual Blocks
        self.res_block2 = residualBlockPSP(self.block_config[0], 128, 64, 256, 1, 1)
        self.res_block3 = residualBlockPSP(self.block_config[1], 256, 128, 512, 2, 1)

        # Dilated Residual Blocks
        self.res_block4 = residualBlockPSP(self.block_config[2], 512, 256, 1024, 1, 2)
        self.res_block5 = residualBlockPSP(self.block_config[3], 1024, 512, 2048, 1, 4)

        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])

        # Final conv layers
        self.cbr_final = conv2DBatchNormRelu(4096, 512, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.classification = nn.Conv2d(512, self.n_classes, 1, 1, 0)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv2DBatchNormRelu(
            in_channels=1024, k_size=3, n_filters=256, padding=1, stride=1, bias=False
        )
        self.aux_cls = nn.Conv2d(256, self.n_classes, 1, 1, 0)

    def forward(self, x):
        inp_shape = x.shape[2:]

        # H, W -> H/2, W/2
        x = self.convbnrelu1_1(x)
        x = self.convbnrelu1_2(x)
        x = self.convbnrelu1_3(x)

        # H/2, W/2 -> H/4, W/4
        x = F.max_pool2d(x, 3, 2, 1)

        # H/4, W/4 -> H/8, W/8
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Auxiliary layers for training
        x_aux = self.convbnrelu4_aux(x)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)

        x = self.res_block5(x)

        x = self.pyramid_pooling(x)

        x = self.cbr_final(x)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode="bilinear")#, align_corners=True)

        if self.training:
            return x_aux, x
        else:  # eval mode
            return x