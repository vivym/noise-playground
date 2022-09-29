from functools import partial
from typing import Union, List, Dict, Any, cast
import sys

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

cifar10_pretrained_weight_urls = {
    'vgg11': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        self.noise_layer_idx = -1
        self.noise_factor = 0.1
        self.apply_feature_noise = False
        self.apply_weight_noise = False
        self._original_weight = None

        def pre_hook(idx, module, inputs):
            if idx == self.noise_layer_idx:
                with torch.no_grad():
                    if self.apply_feature_noise:
                        inputs = map(
                            lambda x: x * (1 + torch.randn_like(x) * self.noise_factor),
                            inputs,
                        )
                        inputs = tuple(inputs)
                    if self.apply_weight_noise:
                        self._original_weight = module.weight
                        module.weight = nn.Parameter(self._original_weight * (
                            1 + torch.randn_like(module.weight) * self.noise_factor
                        ))
                return inputs

        def after_hook(idx, module, inputs, outputs):
            if idx == self.noise_layer_idx:
                with torch.no_grad():
                    if self.apply_weight_noise:
                        module.weight = self._original_weight

        idx = 0
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                m.register_forward_pre_hook(partial(pre_hook, idx))
                m.register_forward_hook(partial(after_hook, idx))
                idx += 1

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.register_forward_pre_hook(partial(pre_hook, idx))
                m.register_forward_hook(partial(after_hook, idx))
                idx += 1

        self.num_layers = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool,
         model_urls: Dict[str, str],
         pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_vgg11(*args, **kwargs) -> VGG: pass
def cifar10_vgg13(*args, **kwargs) -> VGG: pass
def cifar10_vgg16(*args, **kwargs) -> VGG: pass
def cifar10_vgg19(*args, **kwargs) -> VGG: pass


def cifar100_vgg11(*args, **kwargs) -> VGG: pass
def cifar100_vgg13(*args, **kwargs) -> VGG: pass
def cifar100_vgg16(*args, **kwargs) -> VGG: pass
def cifar100_vgg19(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11", "vgg13", "vgg16", "vgg19"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(
                _vgg,
                arch=model_name,
                cfg=cfg,
                batch_norm=True,
                model_urls=model_urls,
                num_classes=num_classes
            )
        )
