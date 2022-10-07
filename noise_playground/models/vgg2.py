from functools import partial

import torch
from torch import nn

from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights
from torchvision.models._utils import handle_legacy_interface


def register_noise_hooks(self):
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


@handle_legacy_interface(weights=("pretrained", VGG16_BN_Weights.IMAGENET1K_V1))
def vgg16(*, weights = None, progress: bool = True, **kwargs):
    model = vgg16_bn(weights=weights, progress=progress, **kwargs)
    register_noise_hooks(model)
    return model
