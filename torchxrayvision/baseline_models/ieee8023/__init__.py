import os
import pathlib
import pickle
import types

import torch
import torch.nn as nn

from ... import utils
from .vqgan import VQModel

model_urls = {}
model_urls["cxr2-vqgan-768-sd"] = {
    "description": "Trained on  mimic_ch pc vin siim objectcxr chex at 768x768 resolution (306 MB).",
    "weights_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/2023-07-04T06-25-02_cxr2_vqgan_768_sd.ckpt",
    "config_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/2023-07-04T06-25-02_cxr2_vqgan_768_sd.yaml",
    "image_range": [-1024, 1024],
    "resolution": 768,
}
model_urls["cxr2-vqgan-768-msdd"] = {
    "description": "Trained on mimic_ch pc vin siim objectcxr chex at 768x768 resolution (1.23 GB).",
    "weights_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/2023-07-29T17-34-14_cxr2_vqgan_768_msdd.ckpt",
    "config_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/2023-07-29T17-34-14_cxr2_vqgan_768_msdd.yaml",
    "image_range": [-1024, 1024],
    "resolution": 768,
}


class _Stub:
    """Stands in for classes the checkpoint pickled that we don't need (e.g.
    pytorch_lightning callbacks) so they don't have to be installed."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("torch", "collections"):
            return super().find_class(module, name)
        return _Stub


# torch.load takes a module that provides an Unpickler
_pickle_module = types.ModuleType("stub_pickle")
_pickle_module.__dict__.update(
    {k: v for k, v in vars(pickle).items() if not k.startswith("__")}
)
_pickle_module.Unpickler = _Unpickler


def _parse_scalar(text):
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if text.lower() in ("null", "~", ""):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _load_yaml(path):
    """Minimal YAML reader (nested mappings, lists of scalars, scalars) for the
    model config files, so no yaml package is needed."""
    root = {}
    stack = [(-1, root)]  # (indent, container)
    with open(path) as f:
        lines = [
            (len(l) - len(l.lstrip()), l.strip())
            for l in f
            if l.strip() and not l.strip().startswith("#")
        ]
    for i, (indent, line) in enumerate(lines):
        while stack[-1][0] >= indent and not (
            line.startswith("- ")
            and stack[-1][0] == indent
            and isinstance(stack[-1][1], list)
        ):
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            parent.append(_parse_scalar(line[2:]))
            continue
        key, _, value = line.partition(":")
        if value.strip():
            parent[key.strip()] = _parse_scalar(value)
        else:
            # a block is a list if its first child is a "- " item
            is_list = i + 1 < len(lines) and lines[i + 1][1].startswith("- ")
            child = [] if is_list else {}
            parent[key.strip()] = child
            stack.append((indent, child))
    return root


def _fetch(url, cache_dir=None):
    """Download ``url`` into the cache directory if needed and return the local path."""
    folder = utils.get_cache_dir() if cache_dir is None else cache_dir
    local = os.path.expanduser(os.path.join(folder, os.path.basename(url)))
    if not os.path.isfile(local):
        print("Downloading {}...".format(os.path.basename(url)))
        print("If this fails you can run `wget {} -O {}`".format(url, local))
        pathlib.Path(local).parent.mkdir(parents=True, exist_ok=True)
        utils.download(url, local)
    return local


class VQGAN(nn.Module):
    """VQGAN image autoencoder for chest X-rays

    A VQ-GAN (`Esser et al., 2021 <https://arxiv.org/abs/2012.09841>`_) that
    encodes a chest X-ray into a grid of discrete latent codes and decodes it
    back. It was built for generating counterfactual explanations with Latent
    Shift (`Cohen et al., 2021 <https://arxiv.org/abs/2102.09475>`_): the
    gradient of a classifier's output is followed in the latent space, and
    decoding gives an image the classifier scores as less positive. It was 
    used for the 6th place MIDRC XAI Challenge winner.

    .. image:: _static/ieee8023-vqgan.jpg

    The native resolution of the model is 768x768. Inputs are expected in the
    TorchXRayVision pixel range of [-1024, 1024] and are resized by bilinear
    interpolation automatically. The single channel is repeated to three for the
    VQ-GAN, and ``decode`` returns one channel clipped to [-1024, 1024] at
    768x768, regardless of the input size.

    Available pre-trained weights, both at 768x768 resolution:

    - ``"cxr2-vqgan-768-sd"`` — 306 MB checkpoint, 64-dimensional codes on a
      24x24 grid (256-entry codebook).
    - ``"cxr2-vqgan-768-msdd"`` — 1.23 GB checkpoint, 128-dimensional codes on
      a 12x12 grid (512-entry codebook).

    .. code-block:: python

        vq = xrv.baseline_models.ieee8023.VQGAN(weights="cxr2-vqgan-768-sd")
        image = torch.ones(1,1,768,768)
        z = vq.encode(image)    # image in [-1024, 1024], shape (B, 1, H, W)
        image2 = vq.decode(z)

    :param weights: Name of pre-trained weights to download and load. Alternatively
        pass ``ckpt_path`` and ``yaml_path`` to use local files.
    :param ckpt_path: Path to a local checkpoint (used instead of ``weights``).
    :param yaml_path: Path to the local model config yaml (used instead of ``weights``).
    :param cache_dir: Directory used to store downloaded weights
        (default: ``~/.torchxrayvision/models_data/``).
    """

    def __init__(
        self,
        weights="cxr2-vqgan-768-sd",
        ckpt_path=None,
        yaml_path=None,
        cache_dir=None,
    ):
        super().__init__()

        self.weights = None
        if ckpt_path is None or yaml_path is None:
            if weights not in model_urls:
                raise Exception(
                    "weights value must be in {}".format(list(model_urls.keys()))
                )
            ckpt_path = _fetch(model_urls[weights]["weights_url"], cache_dir)
            yaml_path = _fetch(model_urls[weights]["config_url"], cache_dir)
            self.weights = weights
            self.weights_metadata = model_urls[weights]
            self.description = model_urls[weights]["description"]

        self.config = _load_yaml(yaml_path)["model"]["params"]
        self.resolution = self.config["ddconfig"]["resolution"]
        # lossconfig is only used for training
        self.model = VQModel(
            ddconfig=self.config["ddconfig"],
            n_embed=self.config["n_embed"],
            embed_dim=self.config["embed_dim"],
        )

        try:
            # Lightning checkpoint with non-tensor objects, so weights_only=True can't load it
            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
                pickle_module=_pickle_module,
            )
            # drop the training-only modules (loss, discriminator, LPIPS)
            state_dict = {
                k: v for k, v in ckpt["state_dict"].items() if not k.startswith("loss.")
            }
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print("Loading failure. Check weights file:", ckpt_path)
            raise (e)
        self.model.eval()

        self.upsample = torch.nn.Upsample(
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )

    def encode(self, x):
        x = x / 1024
        x = x.repeat(1, 3, 1, 1)
        x = self.upsample(x)
        return self.model.encode(x)[0]

    def decode(self, z):
        xp = self.model.decode(z)
        xp = xp[:, :1, :, :]
        xp = xp * 1024
        return torch.clip(xp, -1024, 1024)

    def forward(self, x):
        return self.decode(self.encode(x))

    def __repr__(self):
        if self.weights is not None:
            return self.weights
        return "VQGAN"
