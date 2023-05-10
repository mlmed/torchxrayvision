import sys, os
from typing import List
import torchxrayvision as xrv

thisfolder = os.path.dirname(__file__)
sys.path.insert(0, thisfolder)
from .model import classifier
import json
import pathlib
import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """A model trained on the CheXpert data

    https://github.com/jfhealthcare/Chexpert
    Apache-2.0 License

    .. code-block:: bibtex

        @misc{ye2020weakly,
            title={Weakly Supervised Lesion Localization With Probabilistic-CAM Pooling},
            author={Wenwu Ye and Jin Yao and Hui Xue and Yi Li},
            year={2020},
            eprint={2005.14480},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

    """

    targets: List[str] = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Effusion',
    ]
    """"""

    def __init__(self, apply_sigmoid=True):

        super(DenseNet, self).__init__()
        self.apply_sigmoid = apply_sigmoid

        with open(os.path.join(thisfolder, 'config/example.json')) as f:
            self.cfg = json.load(f)

        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        self.cfg = Struct(**self.cfg)

        model = classifier.Classifier(self.cfg)
        model = nn.DataParallel(model).eval()

        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/baseline_models_jfhealthcare-DenseNet121_pre_train.pth"

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
            model.module.load_state_dict(ckpt)
        except Exception as e:
            print("Loading failure. Check weights file:", self.weights_filename_local)
            raise (e)

        self.model = model
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

        self.pathologies = self.targets

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)

        # expecting values between [-1024,1024]
        x = x / 512
        # now between [-2,2] for this model

        y, _ = self.model(x)
        y = torch.cat(y, 1)

        if self.apply_sigmoid:
            y = torch.sigmoid(y)

        return y

    def __repr__(self):
        return "jfhealthcare-DenseNet121"
