from typing import Optional, Any
from functools import partial

import torch
from torch import nn
from torchvision.models.resnet import ResNet, ResNet34_Weights, BasicBlock
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param


class ResNetWrapper(ResNet):
    def register_importance_hooks(self):
        def after_hook(idx, module, inputs, outputs):
            with torch.no_grad():
                # batch_size, num_channels, h, w
                importance = outputs.flatten(2).abs().mean(-1)
                self.importances[idx].append(importance)

        idx = 0
        for m in [self.conv1]:
            m.register_forward_hook(partial(after_hook, idx))
            idx += 1

        for blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in blocks:
                for m in [block.conv1, block.conv2]:
                    m.register_forward_hook(partial(after_hook, idx))
                    idx += 1

        print("num_layers", idx)
        self.num_layers = idx
        self.importances = [[] for _ in range(self.num_layers)]

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
        for m in [self.conv1]:
            m.register_forward_pre_hook(partial(pre_hook, idx))
            m.register_forward_hook(partial(after_hook, idx))
            idx += 1

        for blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in blocks:
                for m in [block.conv1, block.conv2]:
                    m.register_forward_pre_hook(partial(pre_hook, idx))
                    m.register_forward_hook(partial(after_hook, idx))
                    idx += 1

        for m in [self.fc]:
            m.register_forward_pre_hook(partial(pre_hook, idx))
            m.register_forward_hook(partial(after_hook, idx))
            idx += 1

        self.num_layers = idx

    def get_importances(self):
        return self.importances


def _resnet(
    block,
    layers,
    weights,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNetWrapper(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(weights=("pretrained", ResNet34_Weights.IMAGENET1K_V1))
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
