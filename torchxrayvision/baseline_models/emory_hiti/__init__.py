import sys, os
from typing import List

import numpy as np
import pathlib
import torch
import torch.nn as nn
import torchvision
import torchxrayvision as xrv


class RaceModel(nn.Module):
    """This model is from the work below and is trained to predict the
    patient race from a chest X-ray. Public data from the MIMIC dataset is used
    to train this model. The native resolution of the model is 320x320. Images
    are scaled automatically.

    `Demo notebook <https://github.com/mlmed/torchxrayvision/blob/master/scripts/race_prediction.ipynb>`__

    .. code-block:: python

        model = xrv.baseline_models.emory_hiti.RaceModel()

        image = xrv.utils.load_image('00027426_000.png')
        image = torch.from_numpy(image)[None,...]

        pred = model(image)

        model.targets[torch.argmax(pred)]
        # 'White'

    .. code-block:: bibtex

        @article{Gichoya2022,
            title = {AI recognition of patient race in medical imaging: a modelling study},
            author = {Gichoya, Judy Wawira and Banerjee, Imon and Bhimireddy, Ananth Reddy and Burns, John L and Celi, Leo Anthony and Chen, Li-Ching and Correa, Ramon and Dullerud, Natalie and Ghassemi, Marzyeh and Huang, Shih-Cheng and Kuo, Po-Chih and Lungren, Matthew P and Palmer, Lyle J and Price, Brandon J and Purkayastha, Saptarshi and Pyrros, Ayis T and Oakden-Rayner, Lauren and Okechukwu, Chima and Seyyed-Kalantari, Laleh and Trivedi, Hari and Wang, Ryan and Zaiman, Zachary and Zhang, Haoran},
            doi = {10.1016/S2589-7500(22)00063-2},
            journal = {The Lancet Digital Health},
            pmid = {35568690},
            url = {https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext},
            year = {2022}
        }

    """

    targets: List[str] = ["Asian", "Black", "White"]
    """"""

    def __init__(self):

        super(RaceModel, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=False)
        n_classes = 3
        self.model.fc = nn.Sequential(
            nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))

        self.model = nn.DataParallel(self.model)

        url = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/resnet_race_detection_val-loss_0.157_mimic_public.pt'

        weights_filename = os.path.basename(url)
        weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
        self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

        if not os.path.isfile(self.weights_filename_local):
            print("Downloading weights...")
            print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
            pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
            xrv.utils.download(url, self.weights_filename_local)

        try:
            ckpt = torch.load(self.weights_filename_local, map_location="cpu")
            self.model.load_state_dict(ckpt)
            self.model = self.model.module
            self.model = self.model.eval()  # Must be in eval mode to work correctly
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise e

        self.upsample = nn.Upsample(
            size=(320, 320),
            mode='bilinear',
            align_corners=False,
        )

        self.targets = ["Asian", "Black", "White"]

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.norm = torchvision.transforms.Normalize(self.mean, self.std)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)

        # Expecting values between [-1024,1024]
        x = (x + 1024) / 2048
        # Now between [0,1] for this model

        x = self.norm(x)
        y = self.model(x)

        return y

    def __repr__(self):
        return "Emory-HITI-RaceModel-resnet34"
