import torch
import torch.nn as nn
import pathlib
import os
import sys
import requests


model_urls = {}
model_urls['101-elastic'] = {
    "description": 'This model was trained on the datasets: nih pc rsna mimic_ch chex datasets.',
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/nihpcrsnamimic_ch-resnet101-2-ae-test2-elastic-e250.pt',
    "image_range": [-1024, 1024],
    "resolution": 224,
    "class": "ResNetAE101"
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, progress=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = out + shortcut
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = out + shortcut
        out = self.relu(out)

        return out


# source: https://github.com/ycszen/pytorch-segmentation/blob/master/resnet.py
class _ResNetAE(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(_ResNetAE, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.layer4 = self._make_downlayer(downblock, 128, num_layers[3], stride=6)

        self.uplayer1 = self._make_up_block(upblock, 128, num_layers[3], stride=6)
        self.uplayer2 = self._make_up_block(upblock, 64, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 32, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 16, num_layers[0], stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 64, kernel_size=1, stride=2, bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1, bias=False)

    def __repr__(self):
        if self.weights != None:
            return "XRV-ResNetAE-{}".format(self.weights)
        else:
            return "XRV-ResNetAE"

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))

        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def encode(self, x, check_resolution=True):

        if check_resolution and hasattr(self, 'weights_metadata'):
            resolution = self.weights_metadata['resolution']
            if (x.shape[2] != resolution) | (x.shape[3] != resolution):
                raise ValueError("Input size ({}x{}) is not the native resolution ({}x{}) for this model. Set check_resolution=False on the encode function to override this error.".format(x.shape[2], x.shape[3], resolution, resolution))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def features(self, x):
        return self.encode(x)

    def decode(self, x, image_size=[1, 1, 224, 224]):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=image_size)
        return x

    def forward(self, x):
        ret = {}
        ret["z"] = z = self.encode(x)
        ret["out"] = self.decode(z, x.size())

        return ret


def ResNetAE50(**kwargs):
    return _ResNetAE(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 1, **kwargs)


def ResNetAE101(**kwargs):
    return _ResNetAE(Bottleneck, DeconvBottleneck, [3, 4, 23, 2], 1, **kwargs)


def ResNetAE(weights=None):
    """A ResNet based autoencoder.

    Possible weights for this class include:

    .. code-block:: python

        ae = xrv.autoencoders.ResNetAE(weights="101-elastic") # trained on PadChest, NIH, CheXpert, and MIMIC
        z = ae.encode(image)
        image2 = ae.decode(z)

    """

    if weights == None:
        return ResNetAE101()

    if not weights in model_urls.keys():
        raise Exception("weights value must be in {}".format(list(model_urls.keys())))

    method_to_call = globals()[model_urls[weights]["class"]]
    ae = method_to_call()

    # load pretrained models
    url = model_urls[weights]["weights_url"]
    weights_filename = os.path.basename(url)
    weights_storage_folder = os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data"))
    weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder, weights_filename))

    if not os.path.isfile(weights_filename_local):
        print("Downloading weights...")
        print("If this fails you can run `wget {} -O {}`".format(url, weights_filename_local))
        pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
        download(url, weights_filename_local)

    try:
        state_dict = torch.load(weights_filename_local, map_location='cpu')
        ae.load_state_dict(state_dict)
    except Exception as e:
        print("Loading failure. Check weights file:", weights_filename_local)
        raise (e)

    ae = ae.eval()

    ae.weights = weights
    ae.weights_metadata = model_urls[weights]
    ae.description = model_urls[weights]["description"]

    return ae


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
