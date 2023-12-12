import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import zipfile
import io
import tqdm


def uncertain_logits_to_probs(logits):
    """Convert explicit uncertainty modeling logits to probabilities P(is_abnormal).

    Args:
        logits: Input of shape (batch_size, num_tasks * 3).

    Returns:
        probs: Output of shape (batch_size, num_tasks).
            Position (i, j) interpreted as P(example i has pathology j).
    """
    b, n_times_d = logits.size()
    d = 3
    if n_times_d % d:
        raise ValueError('Expected logits dimension to be divisible by {}, got size {}.'.format(d, n_times_d))
    n = n_times_d // d

    logits = logits.view(b, n, d)
    probs = F.softmax(logits[:, :, 1:], dim=-1)
    probs = probs[:, :, 1]

    return probs


class Model(nn.Module):
    """Models from TorchVision's GitHub page of pretrained neural networks:
        https://github.com/pytorch/vision/tree/master/torchvision/models
    """

    def __init__(self, model_fn, task_sequence, model_uncertainty, use_gpu):
        super(Model, self).__init__()

        self.task_sequence = task_sequence
        self.get_probs = uncertain_logits_to_probs if model_uncertainty else torch.sigmoid
        self.use_gpu = use_gpu

        # Set pretrained to False to avoid loading weights which will be overwritten
        self.model = model_fn(pretrained=False)

        self.pool = nn.AdaptiveAvgPool2d(1)

        num_ftrs = self.model.classifier.in_features
        if model_uncertainty:
            num_outputs = 3 * len(task_sequence)
        else:
            num_outputs = len(task_sequence)

        self.model.classifier = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x).view(x.size(0), -1)
        x = self.model.classifier(x)

        return x

    def features2(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

    def infer(self, x, tasks):

        preds = self(x)
        probs = self.get_probs(preds)[0]

        task2results = {}
        for task in tasks:

            idx = self.task_sequence[task]
            # task_prob = probs.detach().cpu().numpy()[idx]
            task_prob = probs[idx]
            task2results[task] = task_prob

        return task2results


class DenseNet121(Model):
    def __init__(self, task_sequence, model_uncertainty, use_gpu):
        super(DenseNet121, self).__init__(models.densenet121, task_sequence, model_uncertainty, use_gpu)


def load_individual(weights_zip, ckpt_path, model_uncertainty, use_gpu=False):

    with weights_zip.open(ckpt_path) as file:

        stream = io.BytesIO(file.read())
        ckpt_dict = torch.load(stream, map_location="cpu")

    device = 'cuda:0' if use_gpu else 'cpu'

    # Build model, load parameters
    task_sequence = ckpt_dict['task_sequence']
    model = DenseNet121(task_sequence, model_uncertainty, use_gpu)
    model = nn.DataParallel(model)
    model.load_state_dict(ckpt_dict['model_state'])

    return model.eval().to(device), ckpt_dict['ckpt_info']


class Tasks2Models(object):
    """
    Main attribute is a (task tuple) -> {iterator, list} dictionary,
    which loads models iteratively depending on the
    specified task.
    """

    def __init__(self, config_path, weights_zip, num_models=1, dynamic=True, use_gpu=False):

        super(Tasks2Models).__init__()

        self.get_config(config_path)
        self.dynamic = dynamic
        self.use_gpu = use_gpu
        self.weights_zip = zipfile.ZipFile(weights_zip)

        if dynamic:
            model_loader = self.model_iterator
        else:
            model_loader = self.model_list

        model_dicts2tasks = {}
        for task, model_dicts in self.task2model_dicts.items():
            hashable_model_dict = self.get_hashable(model_dicts)
            if hashable_model_dict in model_dicts2tasks:
                model_dicts2tasks[hashable_model_dict].append(task)
            else:
                model_dicts2tasks[hashable_model_dict] = [task]

        # Initialize the iterators
        self.tasks2models = {}
        for task, model_dicts in self.task2model_dicts.items():
            hashable_model_dict = self.get_hashable(model_dicts)
            tasks = tuple(model_dicts2tasks[hashable_model_dict])

            if tasks not in self.tasks2models:
                self.tasks2models[tasks] = model_loader(model_dicts,
                                                        num_models=num_models,
                                                        desc="Loading weights {}".format(tasks))

        self.tasks = list(self.task2model_dicts.keys())

    def get_hashable(self, model_dicts):
        return tuple([tuple(model_dict.items()) for model_dict in model_dicts])

    @property
    def module(self):
        return self

    def get_config(self, config_path):
        """Read configuration from a JSON file.

        Args:
            config_path: Path to configuration JSON file.

        Returns:
            task2models: Dictionary mapping task names to list of dicts.
                Each dict has keys 'ckpt_path' and 'model_uncertainty'.
            aggregation_fn: Aggregation function to combine predictions from multiple models.
        """
        with open(config_path, 'r') as json_fh:
            config_dict = json.load(json_fh)
        self.task2model_dicts = config_dict['task2models']
        agg_method = config_dict['aggregation_method']
        if agg_method == 'max':
            self.aggregation_fn = torch.max
        elif agg_method == 'mean':
            self.aggregation_fn = torch.mean
        else:
            raise ValueError('Invalid configuration: {} = {} (expected "max" or "mean")'.format('aggregation_method', agg_method))

    def model_iterator(self, model_dicts, num_models, desc=""):

        def iterator():

            for model_dict in model_dicts[:num_models]:

                ckpt_path = model_dict['ckpt_path']
                model_uncertainty = model_dict['is_3class']
                model, ckpt_info = load_individual(self.weights_zip, ckpt_path, model_uncertainty, self.use_gpu)

                yield model

        return iterator

    def model_list(self, model_dicts, num_models, desc=""):

        loaded_models = []
        toiter = tqdm.tqdm(model_dicts[:num_models])
        toiter.set_description(desc)
        for model_dict in toiter:
            ckpt_path = model_dict['ckpt_path']
            model_uncertainty = model_dict['is_3class']
            model, ckpt_info = load_individual(self.weights_zip, ckpt_path, model_uncertainty, self.use_gpu)

            loaded_models.append(model)

        def iterator():
            return loaded_models

        return iterator

    def infer(self, img, tasks):

        ensemble_probs = []

        model_iterable = self.tasks2models[tasks]
        task2ensemble_results = {}
        for model in model_iterable():
            individual_task2results = model.module.infer(img, tasks)

            for task in tasks:
                if task not in task2ensemble_results:
                    task2ensemble_results[task] = [individual_task2results[task]]
                else:
                    task2ensemble_results[task].append(individual_task2results[task])

        assert all([task in task2ensemble_results for task in tasks]), \
            "Not all tasks in task2ensemble_results"

        task2results = {}
        for task in tasks:
            ensemble_probs = task2ensemble_results[task]
            task2results[task] = self.aggregation_fn(torch.stack(ensemble_probs), dim=0)

        assert all([task in task2results for task in tasks]), "Not all tasks in task2results"

        return task2results

    def features(self, img, tasks):
        """
        Return shape is [3, 30, 1, 1024]
        3 task groups, 30 models each
        """
        ensemble_probs = []

        model_iterable = self.tasks2models[tasks]
        ensemble_results = []
        for model in model_iterable():
            individual_feats = model.module.features2(img)
            ensemble_results.append(individual_feats)

        return torch.stack(ensemble_results)

    def __iter__(self):
        return iter(self.tasks2models)



