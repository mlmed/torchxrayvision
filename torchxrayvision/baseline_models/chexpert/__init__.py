import sys, os
from typing import List

thisfolder = os.path.dirname(__file__)
sys.path.insert(0, thisfolder)
import torch
import torch.nn as nn
from .model import Tasks2Models
from ... import utils


class DenseNet(nn.Module):
    """CheXpert ensemble DenseNet classifier

    An ensemble of up to 30 DenseNet models trained on the Stanford CheXpert
    dataset, predicting 5 pathologies. Setting ``num_models`` to a value less
    than 30 loads a subset of the ensemble, which reduces memory use and
    inference time at the cost of accuracy.

    **Targets (5):** Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion.

    Modified for TorchXRayVision to maintain the PyTorch gradient tape and
    to expose a ``features()`` method compatible with the rest of the library.

    .. note::
        This class requires a local copy of the pre-trained weights ZIP file,
        which must be passed as ``weights_zip``. The weights are **not**
        downloaded automatically.

    Citation:
        Irvin J, Rajpurkar P, Ko M, et al.
        CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels
        and Expert Comparison.
        *AAAI Conference on Artificial Intelligence*, 2019.
        https://arxiv.org/abs/1901.07031

    Download weights:
        https://academictorrents.com/details/5c7ee21e6770308f2d2b4bd829e896dbd9d3ee87
    """

    targets: List[str] = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
    ]
    """"""

    def __init__(self, weights_zip="", num_models=30):

        super(DenseNet, self).__init__()

        url = "https://academictorrents.com/details/5c7ee21e6770308f2d2b4bd829e896dbd9d3ee87"
        self.weights_zip = weights_zip
        self.num_models = num_models

        if self.weights_zip == "":
            raise Exception("Need to specify weights_zip file location. You can download them from {}".format(url))

        self.use_gpu = torch.cuda.is_available()
        dirname = os.path.dirname(os.path.realpath(__file__))
        self.model = Tasks2Models(os.path.join(dirname, 'predict_configs.json'),
                                  weights_zip=self.weights_zip,
                                  num_models=self.num_models,
                                  dynamic=False,
                                  use_gpu=self.use_gpu)

        self.pathologies = self.targets

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        
        x = utils.fix_resolution(x, 320, self)
        utils.warn_normalization(x)

        # expecting values between [-1024,1024]
        x = x / 512
        # now between [-2,2] for this model

        outputs = []
        for sample in x:  # sorry hard to make parallel
            all_task2prob = {}
            for tasks in self.model:
                task2prob = self.model.infer(sample.unsqueeze(0), tasks)
                for task, task_prob in task2prob.items():
                    all_task2prob[task] = task_prob

            output = [all_task2prob[patho] for patho in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]]
            output = torch.stack(output)
            outputs.append(output)

        return torch.stack(outputs)

    def features(self, x):
        x = x.repeat(1, 3, 1, 1)
        
        x = utils.fix_resolution(x, 320, self)
        utils.warn_normalization(x)

        # expecting values between [-1024,1024]
        x = x / 512
        # now between [-2,2] for this model

        outputs = []
        for sample in x:  # sorry hard to make parallel
            all_feats = []
            for tasks in self.model:
                task2prob = self.model.features(sample.unsqueeze(0), tasks)
                all_feats.append(task2prob)
            feats = torch.stack(all_feats)
            outputs.append(feats.flatten())

        return torch.stack(outputs)

    def __repr__(self):
        return "CheXpert-DenseNet121-ensemble"


