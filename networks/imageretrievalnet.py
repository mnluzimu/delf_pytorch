import os
import pdb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision

from layers.pooling import SPoC, GeM
from layers.normalization import L2N, PowerLaw
from datasets.genericdataset import ImagesFromList
from utils.general import get_data_root
from networks.score_function import ScoreFunction


# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is
# higher for them
FEATURES = {
    'vgg16': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}


# possible global pooling layers, each on of these can be made regional
POOLING = {
    'spoc': SPoC,
    'gem': GeM
}


# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet': 256,
    'vgg11': 512,
    'vgg13': 512,
    'vgg16': 512,
    'vgg19': 512,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,  # largest densenet
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,
}


class ImageRetrievalNet(nn.Module):

    def __init__(self, features, pool, meta, is_attention=False, score_function=None):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        self.meta = meta
        self.is_attention = is_attention
        self.score_function = score_function

    def forward(self, x):
        # x -> features
        o = self.features(x)
        
        # features -> pool -> norm
        if not self.is_attention:
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        else:
            alpha = self.score_function(o)
            o = o.mul(alpha)
            o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):
    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    pooling = params.get('pooling', 'spoc')
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)
    attention = params.get('attention', False)
    path = params.get('path', None)
    extraction = params.get('extraction', False)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    score_function = None
    if attention:
        score_function = ScoreFunction(dim, dim)

    # loading networks from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with networks pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained networks
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize pooling
    pool = POOLING[pooling]()

    # create meta information to be stored in the networks
    meta = {
        'architecture': architecture,
        'pooling': pooling,
        'mean': mean,
        'std': std,
        'outputdim': dim,
    }

    # create a generic image retrieval networks
    net = ImageRetrievalNet(features, pool, meta, is_attention=attention, score_function=score_function)

    # initialize features with custom pretrained networks if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
              .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))
        
    if attention and not extraction:
        model_ft = torch.load(path)
        net.features.load_state_dict(model_ft['state_dict'])
        
    if extraction:
        model_ft = torch.load(path)
        net.load_state_dict(model_ft['state_dict'])
        

    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving networks to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)

            if (i + 1) % print_freq == 0 or (i + 1) == len(images):
                print('\r>>>> {}/{} done...'.format((i + 1), len(images)), end='')
        print('')

    return vecs


def extract_ss(net, input):
    return net(input).cpu().data.squeeze()


def extract_ms(net, input, ms, msp):
    v = torch.zeros(net.meta['outputdim'])

    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()

    v /= len(ms)
    v = v.pow(1. / msp)
    v /= v.norm()

    return v



