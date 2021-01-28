from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib
import pathlib
import os
import numpy as np
import warnings; warnings.filterwarnings("ignore")

model_urls = {}

model_urls['all'] = {
    "description": 'This model was trained on the datasets: nih-pc-chex-mimic_ch-google-openi-rsna and is described here: https://arxiv.org/abs/2002.02497',
    "weights_url": 'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum' ],
    "op_threshs":[0.07422872, 0.038290843, 0.09814756, 0.0098118475, 0.023601074, 0.0022490358, 0.010060724, 0.103246614, 0.056810737, 0.026791653, 0.050318155, 0.023985857, 0.01939503, 0.042889766, 0.053369623, 0.035975814, 0.20204692, 0.05015312],
    "ppv80_thres":[0.72715247, 0.8885005, 0.92493945, 0.6527224, 0.68707734, 0.46127197, 0.7272054, 0.6127343, 0.9878492, 0.61979693, 0.66309816, 0.7853459, 0.930661, 0.93645346, 0.6788558, 0.6547198, 0.61614525, 0.8489876]
}

model_urls['nih'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', '', '', '', '' ],
    "op_threshs":[0.039117552, 0.0034529066, 0.11396341, 0.0057298196, 0.00045666535, 0.0018880932, 0.012037827, 0.038744126, 0.0037213727, 0.014730946, 0.016149804, 0.054241467, 0.037198864, 0.0004403434, np.nan, np.nan, np.nan, np.nan],
}

model_urls['pc'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', '', 'Fracture', '', '' ],
    "op_threshs": [0.031012505, 0.013347598, 0.081435576, 0.001262615, 0.002587246, 0.0035944257, 0.0023071, 0.055412333, 0.044385884, 0.042766232, 0.043258056, 0.037629247, 0.005658899, 0.0091741895, np.nan, 0.026507627, np.nan, np.nan]
}

model_urls['chex'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum' ],
    "op_threshs": [0.1988969, 0.05710573, np.nan, 0.0531293, 0.1435217, np.nan, np.nan, 0.27212676, 0.07749717, np.nan, 0.19712369, np.nan, np.nan, np.nan, 0.09932402, 0.09273402, 0.3270967, 0.10888247],
}

model_urls['rsna'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  '', '', '', '', '', '', '', '', 'Pneumonia', '', '', '', '', '', '', '', 'Lung Opacity', '' ],
    "op_threshs": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.13486601, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.13511065, np.nan]
}

model_urls['mimic_nb'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum' ],
    "op_threshs": [0.08558747, 0.011884617, np.nan, 0.0040595434, 0.010733786, np.nan, np.nan, 0.118761964, 0.022924708, np.nan, 0.06358637, np.nan, np.nan, np.nan, 0.022143636, 0.017476924, 0.1258702, 0.014020768],
}

model_urls['mimic_ch'] = {
    "weights_url":'https://github.com/mlmed/torchxrayvision/releases/download/v1/mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
    "labels":[  'Atelectasis', 'Consolidation', '', 'Pneumothorax', 'Edema', '', '', 'Effusion', 'Pneumonia', '', 'Cardiomegaly', '', '', '', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum' ],
    "op_threshs": [0.09121389, 0.010573786, np.nan, 0.005023008, 0.003698257, np.nan, np.nan, 0.08001232, 0.037242252, np.nan, 0.05006329, np.nan, np.nan, np.nan, 0.019866971, 0.03823637, 0.11303808, 0.0069147074],
}

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
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Modified from torchvision to have a variable number of input channels

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 drop_rate=0, num_classes=18, in_channels=1, weights=None, op_threshs=None, progress=True,
                 apply_sigmoid=False):

        super(DenseNet, self).__init__()            
        
        self.apply_sigmoid = apply_sigmoid
        self.weights = weights
        
        if self.weights != None:
            if not self.weights in model_urls.keys():
                raise Exception("weights value must be in {}".format(list(model_urls.keys())))
                
            # set to be what this model is trained to predict
            self.pathologies = model_urls[weights]["labels"]
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
            
            url = model_urls[weights]["weights_url"]
            weights_filename = os.path.basename(url)
            weights_storage_folder = os.path.expanduser(os.path.join("~",".torchxrayvision","models_data"))
            self.weights_filename_local = os.path.expanduser(os.path.join(weights_storage_folder,weights_filename))
              
            if not os.path.isfile(self.weights_filename_local):
                print("Downloading weights...")
                print("If this fails you can run `wget {} -O {}`".format(url, self.weights_filename_local))
                pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
                download(url, self.weights_filename_local)
            
            try:
                savedmodel = torch.load(self.weights_filename_local, map_location='cpu')
                # patch to load old models https://github.com/pytorch/pytorch/issues/42242
                for mod in savedmodel.modules():
                    if not hasattr(mod, "_non_persistent_buffers_set"):
                        mod._non_persistent_buffers_set = set()

                self.load_state_dict(savedmodel.state_dict())
            except Exception as e:
                print("Loading failure. Check weights file:", weights_filename_local)
                raise(e)
            
            self.eval()
            
            if "op_threshs" in model_urls[weights]:
                self.op_threshs = torch.tensor(model_urls[weights]["op_threshs"])
                
    def __repr__(self):
        if self.weights != None:
            return "XRV-DenseNet121-{}".format(self.weights)
        else:
            return "XRV-DenseNet"
                
    def features2(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out
    
    def forward(self, x):
        features = self.features2(x)
        out = self.classifier(features)
        
        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)
        
        if hasattr(self,"op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out

def op_norm(outputs, op_threshs):
    """normalize outputs according to operating points for a given model.
    Args: 
        outputs: outputs of self.classifier(). torch.Size(batch_size, num_tasks) 
        op_threshs_arr: torch.Size(batch_size, num_tasks) with self.op_threshs expanded.
    Returns:
        outputs_new: normalized outputs, torch.Size(batch_size, num_tasks)
    """
    #expand to batch size so we can do parallel comp
    op_threshs = op_threshs.expand(outputs.shape[0],-1)
    
    #initial values will be 0
    outputs_new = torch.zeros(outputs.shape, device = outputs.device)
    
    # only select non-nan elements otherwise the gradient breaks
    mask_leq = (outputs<op_threshs) & ~torch.isnan(op_threshs)
    mask_gt = ~(outputs<op_threshs) & ~torch.isnan(op_threshs)
    
    #scale outputs less than thresh
    outputs_new[mask_leq] = outputs[mask_leq]/(op_threshs[mask_leq]*2)
    #scale outputs greater than thresh
    outputs_new[mask_gt] = 1.0 - ((1.0 - outputs[mask_gt])/((1-op_threshs[mask_gt])*2))
    
    return outputs_new
    
def get_densenet_params(arch):
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

                                      
import sys
import requests

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
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
                                      
