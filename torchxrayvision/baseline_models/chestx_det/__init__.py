import os
import pathlib
import sys
from collections import OrderedDict

import requests
import torch
import torch.nn as nn
import torchvision

from .ptsemseg.pspnet import pspnet


def _convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


class PSPNet(nn.Module):
    """ChestX-Det Segmentation Model

    https://github.com/Deepwise-AILab/ChestX-Det-Dataset

    @article{Lian2021,
        title = {{A Structure-Aware Relation Network for Thoracic Diseases Detection and Segmentation}},
        author = {Lian, Jie and Liu, Jingyu and Zhang, Shu and Gao, Kai and Liu, Xiaoqing and Zhang, Dingwen and Yu, Yizhou},
        doi = {10.48550/arxiv.2104.10326},
        journal = {IEEE Transactions on Medical Imaging},
        url = {https://arxiv.org/abs/2104.10326},
        year = {2021}
    }
    """

    def __init__(self):

        super(PSPNet, self).__init__()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self._targets = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
                         'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
                         'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum', 'Weasand', 'Spine']

        model = pspnet(len(self.targets))

        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth"

        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            download(url, self.weights_filename_local)

        try:
            ckpt = torch.load(self.weights_filename_local, map_location="cpu")
            ckpt = _convert_state_dict(ckpt)
            model.load_state_dict(ckpt)
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise (e)

        model.eval()
        self.model = model
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

    @property
    def targets(self):
        """A list of the targets that this model will predict"""
        return self._targets

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)

        # expecting values between [-1024,1024]
        x = (x + 1024) / 2048

        # now between [0,1] for this model preprocessing
        x = self.transform(x)
        y = self.model(x)

        return y

    def __repr__(self):
        return "chestx-det-pspnet"


# from here https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url, filename):
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
