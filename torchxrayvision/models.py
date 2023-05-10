from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pathlib
import os
import sys
import requests
import numpy as np
from collections import OrderedDict
from . import datasets
import warnings
warnings.filterwarnings("ignore")


model_urls = {}

model_urls['all'] = {
    "description": 'This model was trained on the datasets: nih-pc-chex-mimic_ch-google-openi-rsna and is described here: https://arxiv.org/abs/2002.02497',
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
    "op_threshs": [0.07422872, 0.038290843, 0.09814756, 0.0098118475, 0.023601074, 0.0022490358, 0.010060724, 0.103246614, 0.056810737, 0.026791653, 0.050318155, 0.023985857, 0.01939503, 0.042889766, 0.053369623, 0.035975814, 0.20204692, 0.05015312],
    "ppv80_thres": [0.72715247, 0.8885005, 0.92493945, 0.6527224, 0.68707734, 0.46127197, 0.7272054, 0.6127343, 0.9878492, 0.61979693, 0.66309816, 0.7853459, 0.930661, 0.93645346, 0.6788558, 0.6547198, 0.61614525, 0.8489876]
}
model_urls['densenet121-res224-all'] = model_urls['all']


model_urls['nih'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', '', '', '', ''],
    "op_threshs": [0.039117552, 0.0034529066, 0.11396341, 0.0057298196, 0.00045666535, 0.0018880932, 0.012037827, 0.038744126, 0.0037213727, 0.014730946, 0.016149804, 0.054241467, 0.037198864, 0.0004403434, np.nan, np.nan, np.nan, np.nan],
}
model_urls['densenet121-res224-nih'] = model_urls['nih']


model_urls['pc'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', '', 'Fracture', '', ''],
    "op_threshs": [0.031012505, 0.013347598, 0.081435576, 0.001262615, 0.002587246, 0.0035944257, 0.0023071, 0.055412333, 0.044385884, 0.042766232, 0.043258056, 0.037629247, 0.005658899, 0.0091741895, np.nan, 0.026507627, np.nan, np.nan]
}
model_urls['densenet121-res224-pc'] = model_urls['pc']

model_urls['chex'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
    "op_threshs": [0.1988969, 0.05710573, np.nan, 0.0531293, 0.1435217, np.nan, np.nan, 0.27212676, 0.07749717, np.nan, 0.19712369, np.nan, np.nan, np.nan, 0.09932402, 0.09273402, 0.3270967, 0.10888247],
}
model_urls['densenet121-res224-chex'] = model_urls['chex']


model_urls['rsna'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['', '', '', '', '', '', '', '', 'Pneumonia', '', '', '', '', '', '', '', 'Lung Opacity', ''],
    "op_threshs": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.13486601, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.13511065, np.nan]
}
model_urls['densenet121-res224-rsna'] = model_urls['rsna']

model_urls['mimic_nb'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
    "op_threshs": [0.08558747, 0.011884617, np.nan, 0.0040595434, 0.010733786, np.nan, np.nan, 0.118761964, 0.022924708, np.nan, 0.06358637, np.nan, np.nan, np.nan, 0.022143636, 0.017476924, 0.1258702, 0.014020768],
}
model_urls['densenet121-res224-mimic_nb'] = model_urls['mimic_nb']

model_urls['mimic_ch'] = {
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels": ['Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
    "op_threshs": [0.09121389, 0.010573786, np.nan, 0.005023008, 0.003698257, np.nan, np.nan, 0.08001232, 0.037242252, np.nan, 0.05006329, np.nan, np.nan, np.nan, 0.019866971, 0.03823637, 0.11303808, 0.0069147074],
}
model_urls['densenet121-res224-mimic_ch'] = model_urls['mimic_ch']

model_urls['resnet50-res512-all'] = {
    "description": 'This model was trained on the datasets pc-nih-rsna-siim-vin at a 512x512 resolution.',
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt',
    "labels": ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
    "op_threshs": [0.51570356, 0.50444704, 0.53787947, 0.50723547, 0.5025118, 0.5035252, 0.5038076, 0.51862943, 0.5078151, 0.50724894, 0.5056339, 0.510706, 0.5053923, 0.5020846, np.nan, 0.5080557, 0.5138526, np.nan],
    "ppv80_thres": [0.690908, 0.720028, 0.7303882, 0.7235838, 0.6787441, 0.7304924, 0.73105824, 0.6839408, 0.7241559, 0.7219969, 0.6346738, 0.72764945, 0.7285066, 0.5735704, np.nan, 0.69684714, 0.7135549, np.nan]
}

# Just created for documentation
class Model:
    """The library is composed of core and baseline classifiers. Core
    classifiers are trained specifically for this library and baseline
    classifiers come from other papers that have been adapted to provide the
    same interface and work with the same input pixel scaling as our core
    models. All models will automatically resize input images (higher or
    lower using bilinear interpolation) to match the specified size they were
    trained on. This allows them to be easily swapped out for experiments.
    Pre-trained models are hosted on GitHub and automatically downloaded to
    the user’s local `~/.torchxrayvision` directory.

    Core pre-trained classifiers are provided as PyTorch Modules which are
    fully differentiable in order to work seamlessly with other PyTorch code.

    """

    targets: List[str]
    """Each classifier provides a field `model.targets` which aligns to
    the list of predictions that the model makes. Depending on the
    weights loaded this list will change. The predictions can be aligned
    to pathology names as follows:
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The pre-trained models can also be used as features extractors for
        semi-supervised training or transfer learning tasks. A feature vector
        can be obtained for each image using the model.features function. The
        resulting size will vary depending on the architecture and the input
        image size. For some models there is a model.features2 method that
        will extract features at a different point of the computation graph.

        .. code-block:: python

            feats = model.features(img)
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The model will output a tensor with the shape [batch, pathologies]
        which is aligned to the order of the list `model.pathologies`.

        .. code-block:: python

            preds = model(img)
            print(dict(zip(model.targets, preds.tolist()[0])))
            # {'Atelectasis': 0.5583771,
            #  'Consolidation': 0.5279943,
            #  'Infiltration': 0.60061914,
            #  ...
        """
        pass

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Based on 
    `"Densely Connected Convolutional Networks" <https://arxiv.org/abs/1608.06993>`_

    Possible weights for this class include:

    .. code-block:: python

        ## 224x224 models
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
        model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
        model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
        model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
        model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)

    :param weights: Specify a weight name to load pre-trained weights
    :param op_threshs: Specify a weight name to load pre-trained weights 
    :param apply_sigmoid: Apply a sigmoid 
        
    """

    targets: List[str] = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia',
        'Lung Lesion',
        'Fracture',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
    ]
    """"""

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=len(datasets.default_pathologies),
                 in_channels=1,
                 weights=None,
                 op_threshs=None,
                 apply_sigmoid=False
                 ):

        super(DenseNet, self).__init__()

        self.apply_sigmoid = apply_sigmoid
        self.weights = weights

        if self.weights is not None:
            if not self.weights in model_urls.keys():
                possible_weights = [k for k in model_urls.keys() if k.startswith("densenet")]
                raise Exception("Weights value must be in {}".format(possible_weights))

            # set to be what this model is trained to predict
            self.targets = model_urls[weights]["labels"]
            self.pathologies = self.targets  # keep to be backward compatible

            # if different from default number of classes
            if num_classes != len(datasets.default_pathologies):
                raise ValueError("num_classes and weights cannot both be specified. The weights loaded will define the own number of output classes.")

            num_classes = len(self.pathologies)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)

        if self.weights != None:
            self.weights_filename_local = get_weights(weights)

            try:
                savedmodel = torch.load(self.weights_filename_local, map_location='cpu')
                # patch to load old models https://github.com/pytorch/pytorch/issues/42242
                for mod in savedmodel.modules():
                    if not hasattr(mod, "_non_persistent_buffers_set"):
                        mod._non_persistent_buffers_set = set()

                self.load_state_dict(savedmodel.state_dict())
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                raise e

            self.eval()

            if "op_threshs" in model_urls[weights]:
                self.op_threshs = torch.tensor(model_urls[weights]["op_threshs"])

            self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def __repr__(self):
        if self.weights is not None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"

    def features2(self, x):
        x = fix_resolution(x, 224, self)
        warn_normalization(x)

        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

    def forward(self, x):
        x = fix_resolution(x, 224, self)

        features = self.features2(x)
        out = self.classifier(features)

        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)

        if hasattr(self, "op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out


##########################
class ResNet(nn.Module):
    """
    Based on `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Possible weights for this class include:

    .. code-block:: python

        # 512x512 models
        model = xrv.models.ResNet(weights="resnet50-res512-all")

    :param weights: Specify a weight name to load pre-trained weights
    :param op_threshs: Specify a weight name to load pre-trained weights 
    :param apply_sigmoid: Apply a sigmoid 

    """

    targets: List[str] = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia',
        'Lung Lesion',
        'Fracture',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
    ]
    """"""


    def __init__(self, weights: str = None, apply_sigmoid: bool = False):
        super(ResNet, self).__init__()

        self.weights = weights
        self.apply_sigmoid = apply_sigmoid

        if not self.weights in model_urls.keys():
            possible_weights = [k for k in model_urls.keys() if k.startswith("resnet")]
            raise Exception("Weights value must be in {}".format(possible_weights))

        self.weights_filename_local = get_weights(weights)
        self.weights_dict = model_urls[weights]
        self.targets = model_urls[weights]["labels"]
        self.pathologies = self.targets  # keep to be backward compatible

        if self.weights.startswith("resnet101"):
            self.model = torchvision.models.resnet101(num_classes=len(self.weights_dict["labels"]), pretrained=False)
            # patch for single channel
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif self.weights.startswith("resnet50"):
            self.model = torchvision.models.resnet50(num_classes=len(self.weights_dict["labels"]), pretrained=False)
            # patch for single channel
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        try:
            self.model.load_state_dict(torch.load(self.weights_filename_local))
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise e

        if "op_threshs" in model_urls[weights]:
            self.register_buffer('op_threshs', torch.tensor(model_urls[weights]["op_threshs"]))

        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

        self.eval()

    def __repr__(self):
        if self.weights is not None:
            return "XRV-ResNet-{}".format(self.weights)
        else:
            return "XRV-ResNet"

    def features(self, x):
        x = fix_resolution(x, 512, self)
        warn_normalization(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = fix_resolution(x, 512, self)
        warn_normalization(x)

        out = self.model(x)

        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)

        if hasattr(self, "op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out


warning_log = {}


def fix_resolution(x, resolution: int, model: nn.Module):
    """Check resolution of input and resize to match requested."""

    # just skip it if upsample was removed somehow
    if not hasattr(model, 'upsample') or (model.upsample == None):
        return x

    if (x.shape[2] != resolution) | (x.shape[3] != resolution):
        if not hash(model) in warning_log:
            print("Warning: Input size ({}x{}) is not the native resolution ({}x{}) for this model. A resize will be performed but this could impact performance.".format(x.shape[2], x.shape[3], resolution, resolution))
            warning_log[hash(model)] = True
        return model.upsample(x)
    return x


def warn_normalization(x):
    """Check normalization of input and warn if possibly wrong. When 
    processing an image that may likely not have the correct 
    normalization we can issue a warning. But running min and max on 
    every image/batch is costly so we only do it on the first image/batch.
    """

    # Only run this check on the first image so we don't hurt performance.
    if not "norm_check" in warning_log:
        x_min = x.min()
        x_max = x.max()
        if torch.logical_or(-255 < x_min, x_max < 255) or torch.logical_or(x_min < -1024, 1024 < x_max):
            print(f'Warning: Input image does not appear to be normalized correctly. The input image has the range [{x_min:.2f},{x_max:.2f}] which doesn\'t seem to be in the [-1024,1024] range. This warning may be wrong though. Only the first image is tested and we are only using a heuristic in an attempt to save a user from using the wrong normalization.')
            warning_log["norm_correct"] = False
        else:
            warning_log["norm_correct"] = True

        warning_log["norm_check"] = True


def op_norm(outputs, op_threshs):
    """Normalize outputs according to operating points for a given model.
    Args: 
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks) 
        op_threshs_arr: torch.Size(batch_size, num_tasks) with self.op_threshs expanded.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    # expand to batch size so we can do parallel comp
    op_threshs = op_threshs.expand(outputs.shape[0], -1)

    # initial values will be 0.5
    outputs_new = torch.zeros(outputs.shape, device=outputs.device) + 0.5

    # only select non-nan elements otherwise the gradient breaks
    mask_leq = (outputs < op_threshs) & ~torch.isnan(op_threshs)
    mask_gt = ~(outputs < op_threshs) & ~torch.isnan(op_threshs)

    # scale outputs less than thresh
    outputs_new[mask_leq] = outputs[mask_leq] / (op_threshs[mask_leq] * 2)
    # scale outputs greater than thresh
    outputs_new[mask_gt] = 1.0 - ((1.0 - outputs[mask_gt]) / ((1 - op_threshs[mask_gt]) * 2))

    return outputs_new


def get_densenet_params(arch: str):
    assert 'dense' in arch
    if arch == 'densenet161':
        ret = dict(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96)
    elif arch == 'densenet169':
        ret = dict(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64)
    elif arch == 'densenet201':
        ret = dict(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64)
    else:
        # default configuration: densenet121
        ret = dict(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)
    return ret


def get_model(weights: str, **kwargs):
    if weights.startswith("densenet"):
        return DenseNet(weights=weights, **kwargs)
    elif weights.startswith("resnet"):
        return ResNet(weights=weights, **kwargs)
    else:
        raise Exception("Unknown model")


def get_weights(weights: str):
    if not weights in model_urls:
        raise Exception("Weights not found. Valid options: {}".format(list(model_urls.keys())))

    url = model_urls[weights]["weights_url"]
    weights_filename = os.path.basename(url)
    weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
    weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

    if not os.path.isfile(weights_filename_local):
        print("Downloading weights...")
        print("If this fails you can run `wget {} -O {}`".format(url, weights_filename_local))
        pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
        download(url, weights_filename_local)

    return weights_filename_local


# from here https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
