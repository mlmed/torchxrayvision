import os
import pathlib
from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

import torchxrayvision as xrv
from ... import utils


MODEL_NAME = "UNet_ResNet50_default"
WEIGHTS_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "baseline_models_cxas_UNet_ResNet50_default_repack.pth"
)

TARGETS: List[str] = [
    "spine",
    "cervical spine",
    "thoracic spine",
    "lumbar spine",
    "vertebrae C1",
    "vertebrae C2",
    "vertebrae C3",
    "vertebrae C4",
    "vertebrae C5",
    "vertebrae C6",
    "vertebrae C7",
    "vertebrae T1",
    "vertebrae T2",
    "vertebrae T3",
    "vertebrae T4",
    "vertebrae T5",
    "vertebrae T6",
    "vertebrae T7",
    "vertebrae T8",
    "vertebrae T9",
    "vertebrae T10",
    "vertebrae T11",
    "vertebrae T12",
    "vertebrae L1",
    "vertebrae L2",
    "vertebrae L3",
    "vertebrae L4",
    "vertebrae L5",
    "rib_cartilage",
    "sternum",
    "clavicles",
    "clavicle left",
    "clavicle right",
    "scapulas",
    "scapula left",
    "scapula right",
    "posterior 12th rib right",
    "posterior 12th rib left",
    "anterior 11th rib right",
    "posterior 11th rib right",
    "anterior 11th rib left",
    "posterior 11th rib left",
    "anterior 10th rib right",
    "posterior 10th rib right",
    "anterior 10th rib left",
    "posterior 10th rib left",
    "anterior 9th rib right",
    "posterior 9th rib right",
    "anterior 9th rib left",
    "posterior 9th rib left",
    "anterior 8th rib right",
    "posterior 8th rib right",
    "anterior 8th rib left",
    "posterior 8th rib left",
    "anterior 7th rib right",
    "posterior 7th rib right",
    "anterior 7th rib left",
    "posterior 7th rib left",
    "anterior 6th rib right",
    "posterior 6th rib right",
    "anterior 6th rib left",
    "posterior 6th rib left",
    "anterior 5th rib right",
    "posterior 5th rib right",
    "anterior 5th rib left",
    "posterior 5th rib left",
    "anterior 4th rib right",
    "posterior 4th rib right",
    "anterior 4th rib left",
    "posterior 4th rib left",
    "anterior 3rd rib right",
    "posterior 3rd rib right",
    "anterior 3rd rib left",
    "posterior 3rd rib left",
    "anterior 2nd rib right",
    "posterior 2nd rib right",
    "anterior 2nd rib left",
    "posterior 2nd rib left",
    "anterior 1st rib right",
    "posterior 1st rib right",
    "anterior 1st rib left",
    "posterior 1st rib left",
    "12th rib",
    "posterior 11th rib",
    "anterior 11th rib",
    "posterior 10th rib",
    "anterior 10th rib",
    "posterior 9th rib",
    "anterior 9th rib",
    "posterior 8th rib",
    "anterior 8th rib",
    "posterior 7th rib",
    "anterior 7th rib",
    "posterior 6th rib",
    "anterior 6th rib",
    "posterior 5th rib",
    "anterior 5th rib",
    "posterior 4th rib",
    "anterior 4th rib",
    "posterior 3rd rib",
    "anterior 3rd rib",
    "posterior 2nd rib",
    "anterior 2nd rib",
    "posterior 1st rib",
    "anterior 1st rib",
    "diaphragm",
    "left hemidiaphragm",
    "right hemidiaphragm",
    "stomach",
    "small bowel",
    "duodenum",
    "liver",
    "pancreas",
    "kidney left",
    "kidney right",
    "cardiomediastinum",
    "upper mediastinum",
    "lower mediastinum",
    "anterior mediastinum",
    "middle mediastinum",
    "posterior mediastinum",
    "heart",
    "heart atrium left",
    "heart atrium right",
    "heart myocardium",
    "heart ventricle left",
    "heart ventricle right",
    "aorta",
    "ascending aorta",
    "descending aorta",
    "aortic arch",
    "pulmonary artery",
    "inferior vena cava",
    "esophagus",
    "lung",
    "right lung",
    "left lung",
    "lung base",
    "mid zone lung",
    "upper zone lung",
    "apical zone lung",
    "right upper zone lung",
    "right mid zone lung",
    "right lung base",
    "right apical zone lung",
    "left upper zone lung",
    "left mid zone lung",
    "left lung base",
    "left apical zone lung",
    "lung lower lobe left",
    "lung upper lobe left",
    "lung lower lobe right",
    "lung middle lobe right",
    "lung upper lobe right",
    "trachea",
    "tracheal bifurcation",
    "breast",
    "breast left",
    "breast right",
]


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "net."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "network", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a state dict or a checkpoint dict."
        )

    return {_strip_prefix(key): value for key, value in checkpoint.items()}


def _remap_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Align checkpoint keys with the xrv wrapper layout.

    The upstream CXAS checkpoints sometimes store backbone parameters under
    ``backbone.*`` while this wrapper nests the same module one level deeper
    as ``backbone.backbone.*``.
    """

    if any(key.startswith("backbone.backbone.") for key in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            key = key.replace("backbone.", "backbone.backbone.", 1)
        remapped[key] = value
    return remapped


class Backbone(nn.Module):
    """Backbone module for extracting features from pre-trained networks."""

    def __init__(self, network_name):
        super(Backbone, self).__init__()
        network_name = network_name.split("_")[1].lower()
        self.network_name = network_name
        self.backbone = self._get_backbone(network_name)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _get_backbone(self, network_name):
        if network_name == "vgg16":
            full_net = getattr(models, network_name)()
            features = list(full_net.features)[:30]
            net = nn.Sequential(*features)
            self.classifier = nn.Sequential(*list(full_net.classifier)[:5])
        elif "resnet" in network_name:
            full_net = getattr(models, network_name)()
            features = [
                (
                    "layer0",
                    torch.nn.Sequential(
                        *[full_net.conv1, full_net.bn1, full_net.relu, full_net.maxpool]
                    ),
                ),
                ("layer1", full_net.layer1),
                ("layer2", full_net.layer2),
                ("layer3", full_net.layer3),
                ("layer4", full_net.layer4),
            ]
            self.inplanes = full_net.inplanes
            net = nn.Sequential(OrderedDict(features))
        else:
            raise NotImplementedError(
                "{} not implemented as BACKBONE Network".format(network_name)
            )

        return net

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        stride=1,
        dilation=1,
        new_level=True,
        residual=True,
    ):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample,
                dilation=((1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation)),
                residual=residual,
            )
        )
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, residual=residual, dilation=(dilation, dilation))
            )

        return nn.Sequential(*layers)

    def insert_dilations(self, features, dilation_size):
        feat = [f for f in features.children()]
        for f in feat:
            if not hasattr(f, "downsample"):
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1, 1):
                            tmp.stride = 1
                            tmp.dilation = (int(dilation_size), int(dilation_size))
                            tmp.padding = (int(dilation_size), int(dilation_size))
            elif f.downsample is None:
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1, 1):
                            tmp.stride = 1
                            tmp.dilation = (int(dilation_size), int(dilation_size))
                            tmp.padding = (int(dilation_size), int(dilation_size))
            else:
                if type(f) == models.resnet.BasicBlock:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1, 1):
                                tmp.stride = 1
                                tmp.dilation = (int(dilation_size), int(dilation_size))
                                tmp.padding = (int(dilation_size), int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride = 1
                else:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1, 1):
                                tmp.stride = 1
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride = 1
        return features

    def forward(self, x):
        if "resnet" in self.network_name:
            return self.preset_forward(x)
        else:
            return self._forward(x)

    def _forward(self, x):
        return self.backbone(x)

    def preset_forward(self, x, insert_layer=None, return_layer=[1, 2, 3, 4, 5]):
        assert (
            insert_layer is None
            or return_layer is None
            or type(return_layer) is list
            or insert_layer < return_layer
        )

        if type(return_layer) is int:
            return_layer = [return_layer]
        result = OrderedDict()
        if insert_layer is None or insert_layer == 0:
            x = self.backbone[0](x)
        if 1 in return_layer:
            result["feats_{}_map".format(1)] = x
            if 1 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 1:
            x = self.backbone[1](x)
        if 2 in return_layer:
            result["feats_{}_map".format(2)] = x
            if 2 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 2:
            x = self.backbone[2](x)
        if 3 in return_layer:
            result["feats_{}_map".format(3)] = x
            if 3 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 3:
            x = self.backbone[3](x)
        if 4 in return_layer:
            result["feats_{}_map".format(4)] = x
            if 4 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 4:
            x = self.backbone[4](x)

        result["feats_last_map"] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        result["feats_pooled"] = x
        return result


class DoubleConv(nn.Module):
    """Double Convolution Block: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Conv(nn.Module):
    """Convolution Block: (convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = F.interpolate
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1, x2.shape[2:], mode="bilinear")
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpInit(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, temp_channels, in_channels, out_channels, bilinear=True, hax=False):
        super().__init__()
        self.up = F.interpolate
        self.conv1 = Conv(in_channels, temp_channels)
        if hax:
            self.conv2 = DoubleConv(
                in_channels + temp_channels, out_channels, in_channels // 2
            )
        else:
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.up(x1, x2.shape[2:], mode="bilinear")
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BackboneUNet(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, classes: int = len(TARGETS)):
        super().__init__()
        self.backbone = Backbone(model_name)
        self.head = get_unet_head(model_name, classes)
        self.dropout = nn.Dropout(p=0)
        self.threshold = 0.5

    def get_results(self, forward_dict, orig_dict):
        out_dict = {**forward_dict, **orig_dict}
        out_dict["segmentation_preds"] = (
            forward_dict["logits"].sigmoid() > self.threshold
        ).bool()
        return out_dict

    def forward(self, x):
        img = x["data"]
        forward_dict = self._forward(img)
        return self.get_results(forward_dict, x)

    def _forward(self, x):
        backbone_dict = self.backbone(x)
        down1 = backbone_dict["feats_1_map"]
        down2 = backbone_dict["feats_2_map"]
        down3 = backbone_dict["feats_3_map"]
        down4 = backbone_dict["feats_4_map"]
        down5 = backbone_dict["feats_last_map"]
        up1 = self.dropout(self.head.up1(down5, down4))
        up2 = self.dropout(self.head.up2(up1, down3))
        up3 = self.dropout(self.head.up3(up2, down2))
        up4 = self.dropout(self.head.up4(up3, down1))
        up = F.interpolate(up4, x.shape[2:], mode="bilinear")
        logits = self.head.out(up)
        return {"feats": up4, "logits": logits}


def get_unet_head(network_name, classes, batch_size=1):
    if "vit" in network_name:
        return UNetHead(
            256,
            128,
            classes,
            norm="batch" if batch_size > 1 else "instance",
            constant=True,
        )
    elif "resnet34" in network_name:
        return UNetHead(
            512, 128, classes, norm="batch" if batch_size > 1 else "instance"
        )
    else:
        return UNetHead(
            2048, 128, classes, norm="batch" if batch_size > 1 else "instance"
        )


class UNetHead(nn.Module):
    def __init__(self, in_channels, ngf, num_classes, norm="batch", constant=False):
        super().__init__()
        if constant:
            self.up1 = UpInit(in_channels, in_channels, in_channels, True, True)
            self.up2 = Up(2 * in_channels, in_channels, True)
            self.up3 = Up(2 * in_channels, in_channels, True)
            self.up4 = Up(2 * in_channels, in_channels, True)
            self.out = OutConv(in_channels, num_classes)
        else:
            self.up1 = UpInit(in_channels // 2, in_channels, in_channels // 4, True)
            self.up2 = Up(in_channels // 4 * 2, in_channels // 8, True)
            self.up3 = Up(in_channels // 8 * 2, in_channels // 32, True)
            self.up4 = Up(in_channels // 32 * 2, ngf, True)
            self.out = OutConv(ngf, num_classes)


class UNetResNet50(BackboneUNet):
    """ChestXRayAnatomySegmentation model (UNet-ResNet50)

    A U-Net with a ResNet-50 backbone pre-trained for pixel-level segmentation
    of 159 anatomical structures in chest X-rays. Output shape is
    ``[batch, 159, 512, 512]``.

    **Targets (159):** 
        "spine",
        "cervical spine",
        "thoracic spine",
        "lumbar spine",
        "vertebrae C1",
        "vertebrae C2",
        "vertebrae C3",
        "vertebrae C4",
        "vertebrae C5",
        "vertebrae C6",
        "vertebrae C7",
        "vertebrae T1",
        "vertebrae T2",
        "vertebrae T3",
        "vertebrae T4",
        "vertebrae T5",
        "vertebrae T6",
        "vertebrae T7",
        "vertebrae T8",
        "vertebrae T9",
        "vertebrae T10",
        "vertebrae T11",
        "vertebrae T12",
        "vertebrae L1",
        "vertebrae L2",
        "vertebrae L3",
        "vertebrae L4",
        "vertebrae L5",
        "rib_cartilage",
        "sternum",
        "clavicles",
        "clavicle left",
        "clavicle right",
        "scapulas",
        "scapula left",
        "scapula right",
        "posterior 12th rib right",
        "posterior 12th rib left",
        "anterior 11th rib right",
        "posterior 11th rib right",
        "anterior 11th rib left",
        "posterior 11th rib left",
        "anterior 10th rib right",
        "posterior 10th rib right",
        "anterior 10th rib left",
        "posterior 10th rib left",
        "anterior 9th rib right",
        "posterior 9th rib right",
        "anterior 9th rib left",
        "posterior 9th rib left",
        "anterior 8th rib right",
        "posterior 8th rib right",
        "anterior 8th rib left",
        "posterior 8th rib left",
        "anterior 7th rib right",
        "posterior 7th rib right",
        "anterior 7th rib left",
        "posterior 7th rib left",
        "anterior 6th rib right",
        "posterior 6th rib right",
        "anterior 6th rib left",
        "posterior 6th rib left",
        "anterior 5th rib right",
        "posterior 5th rib right",
        "anterior 5th rib left",
        "posterior 5th rib left",
        "anterior 4th rib right",
        "posterior 4th rib right",
        "anterior 4th rib left",
        "posterior 4th rib left",
        "anterior 3rd rib right",
        "posterior 3rd rib right",
        "anterior 3rd rib left",
        "posterior 3rd rib left",
        "anterior 2nd rib right",
        "posterior 2nd rib right",
        "anterior 2nd rib left",
        "posterior 2nd rib left",
        "anterior 1st rib right",
        "posterior 1st rib right",
        "anterior 1st rib left",
        "posterior 1st rib left",
        "12th rib",
        "posterior 11th rib",
        "anterior 11th rib",
        "posterior 10th rib",
        "anterior 10th rib",
        "posterior 9th rib",
        "anterior 9th rib",
        "posterior 8th rib",
        "anterior 8th rib",
        "posterior 7th rib",
        "anterior 7th rib",
        "posterior 6th rib",
        "anterior 6th rib",
        "posterior 5th rib",
        "anterior 5th rib",
        "posterior 4th rib",
        "anterior 4th rib",
        "posterior 3rd rib",
        "anterior 3rd rib",
        "posterior 2nd rib",
        "anterior 2nd rib",
        "posterior 1st rib",
        "anterior 1st rib",
        "diaphragm",
        "left hemidiaphragm",
        "right hemidiaphragm",
        "stomach",
        "small bowel",
        "duodenum",
        "liver",
        "pancreas",
        "kidney left",
        "kidney right",
        "cardiomediastinum",
        "upper mediastinum",
        "lower mediastinum",
        "anterior mediastinum",
        "middle mediastinum",
        "posterior mediastinum",
        "heart",
        "heart atrium left",
        "heart atrium right",
        "heart myocardium",
        "heart ventricle left",
        "heart ventricle right",
        "aorta",
        "ascending aorta",
        "descending aorta",
        "aortic arch",
        "pulmonary artery",
        "inferior vena cava",
        "esophagus",
        "lung",
        "right lung",
        "left lung",
        "lung base",
        "mid zone lung",
        "upper zone lung",
        "apical zone lung",
        "right upper zone lung",
        "right mid zone lung",
        "right lung base",
        "right apical zone lung",
        "left upper zone lung",
        "left mid zone lung",
        "left lung base",
        "left apical zone lung",
        "lung lower lobe left",
        "lung upper lobe left",
        "lung lower lobe right",
        "lung middle lobe right",
        "lung upper lobe right",
        "trachea",
        "tracheal bifurcation",
        "breast",
        "breast left",
        "breast right"

    `CXAS Demo notebook <https://github.com/mlmed/torchxrayvision/blob/main/scripts/segmentation-chestx_anatomy.ipynb>`_

    .. code-block:: python

        seg_model = xrv.baseline_models.chestx_anatomy.UNetResNet50()
        output = seg_model(image)
        output.shape  # [1, 159, 512, 512]

    .. image:: _static/segmentation-cxas.png

    License:
        Creative Commons Attribution-NonCommercial-ShareAlike
        "the license applies only to the CXAS model and weights, and does not extend to or restrict the rest of 
        the TorchXRayVision library, which can remain under Apache" - Constantin Seibold (code author)

    Citation:
        Seibold C M, Reiß S, Sarfraz M S, et al.
        Detailed Annotations of Chest X-Rays via CT Projection for Report
        Understanding.
        *33rd British Machine Vision Conference (BMVC)*, 2022.
        url: https://bmvc2022.mpi-inf.mpg.de/0058.pdf

        Seibold C, Jaus A, Fink M A, Kim M, Reiß S, Herrmann K, Kleesiek J,
        Stiefelhagen R.
        Accurate fine-grained segmentation of human anatomy in radiographs via
        volumetric pseudo-labeling.
        arXiv:2306.03934, 2023.
        url: https://arxiv.org/abs/2306.03934

        https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation
    """

    targets: List[str] = TARGETS

    def __init__(self, weights: bool = True, cache_dir: str = None):
        super().__init__(MODEL_NAME, classes=len(self.targets))

        self.model_name = MODEL_NAME
        self.input_resolution = 512
        self.targets = TARGETS
        self.transform = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        if weights:
            url = WEIGHTS_URL
            weights_filename = os.path.basename(url)
            if cache_dir is None:
                weights_storage_folder = utils.get_cache_dir()
            else:
                weights_storage_folder = cache_dir
            self.weights_filename_local = os.path.expanduser(
                os.path.join(weights_storage_folder, weights_filename)
            )

            if not os.path.isfile(self.weights_filename_local):
                print("Downloading weights...")
                print(
                    "If this fails you can run `wget {} -O {}`".format(
                        url, self.weights_filename_local
                    )
                )
                pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
                xrv.utils.download(url, self.weights_filename_local)

            try:
                checkpoint = torch.load(
                    self.weights_filename_local, map_location="cpu", weights_only=False
                )
                state_dict = _remap_state_dict_keys(_extract_state_dict(checkpoint))
                self.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print("Loading failure. Check weights file:", self.weights_filename_local)
                raise e
        else:
            self.weights_filename_local = None

        self.eval()

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if isinstance(x, dict):
            return super().forward(x)

        x = x.repeat(1, 3, 1, 1)
        x = utils.fix_resolution(x, self.input_resolution, self)
        utils.warn_normalization(x)
        x = (x + 1024) / 2048
        x = self.transform(x)
        out = self._forward(x)
        return out["logits"]

    def __repr__(self):
        return "cxas-unet-resnet50"
