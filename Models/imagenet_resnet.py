# Based on code taken from https://pytorch.org/docs/stable/torchvision/models.html

import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from Layers import layers
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layers.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layers.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_num_out_channel(self, c, list_E, V_out_select):
        if self.downsample is not None:
            for layer in reversed(self.downsample):
                if hasattr(layer, 'get_num_out_channel'):
                    shortcut = self.downsample.get_num_out_channel(c, list_E, V_out_select)
        else:
            shortcut = c
        c = self.conv2.get_num_out_channel(c, list_E, V_out_select)
        c = self.conv1.get_num_out_channel(c, list_E, V_out_select)
        return max(shortcut, c)
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Branch')
        print(f'Main 1')
        out = self.conv1.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        print(f'Main 2')
        out = self.conv2.forward_set_scores(x=out, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        if self.downsample is not None:
            for layer in self.downsample:
                if hasattr(layer, 'forward_set_scores'):
                    print(f'Downsample')
                    out += layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
                    out = 1.0 * (out != 0)
        else:
            print(f'Skip connection')
            out += x
            out = 1.0 * (out != 0)
        return out

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        if is_forward:
            _, out_x = self.conv1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.conv2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            shortcut = x
            if self.downsample is not None:
                for layer in self.downsample:
                    if hasattr(layer, 'modify_effective_sparsity'):
                        _, shortcut = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=shortcut)
                        shortcut    = 1.0 * (shortcut != 0)
            return None, 1.0 * ((out_x + shortcut) != 0)
        else:
            print(f'out2')
            out, _ = self.conv2.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            print(f'out1')
            out, _ = self.conv1.modify_effective_sparsity(connected_channels=out, is_forward=is_forward, x=None)
            if self.downsample is not None:
                assert len(self.downsample) == 2
                assert isinstance(self.downsample[0], nn.Conv2d) and isinstance(self.downsample[1], nn.BatchNorm2d)
                for layer in reversed(self.downsample):
                    if hasattr(layer, 'modify_effective_sparsity') and not isinstance(layer, nn.BatchNorm2d):
                        print('shortcut')
                        shortcut, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            else:
                shortcut = connected_channels
            return torch.tensor(sorted(list(set(torch.cat([out, shortcut]).to('cpu').tolist()))), dtype=torch.int64).to(self.bn2.bias.device), None





class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_num_out_channel(self, c, list_E, V_out_select):
        if self.downsample is not None:
            for layer in reversed(self.downsample):
                if hasattr(layer, 'get_num_out_channel'):
                    shortcut = layer.get_num_out_channel(c, list_E, V_out_select)
        else:
            shortcut = c
        c = self.conv3.get_num_out_channel(c, list_E, V_out_select)
        c = self.conv2.get_num_out_channel(c, list_E, V_out_select)
        c = self.conv1.get_num_out_channel(c, list_E, V_out_select)
        return max(shortcut, c)
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Branch')
        print(f'Main 1')
        out = self.conv1.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        print(f'Main 2')
        out = self.conv2.forward_set_scores(x=out, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        print(f'Main 3')
        out = self.conv3.forward_set_scores(x=out, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        if self.downsample is not None:
            for layer in self.downsample:
                if hasattr(layer, 'forward_set_scores'):
                    print(f'Downsample')
                    out += layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
                    out = 1.0 * (out != 0)
        else:
            print(f'Skip connection')
            out += x
            out = 1.0 * (out != 0)
        return out

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        if is_forward:
            _, out_x = self.conv1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.conv2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.conv3.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn3.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            shortcut = x
            if self.downsample is not None:
                for layer in self.downsample:
                    if hasattr(layer, 'modify_effective_sparsity'):
                        _, shortcut = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=shortcut)
                        shortcut    = 1.0 * (shortcut != 0)
            return None, 1.0 * ((out_x + shortcut) != 0)
        else:
            print(f'out3')
            out, _ = self.conv3.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            print(f'out2')
            out, _ = self.conv2.modify_effective_sparsity(connected_channels=out, is_forward=is_forward, x=None)
            print(f'out1')
            out, _ = self.conv1.modify_effective_sparsity(connected_channels=out, is_forward=is_forward, x=None)
            if self.downsample is not None:
                assert len(self.downsample) == 2
                assert isinstance(self.downsample[0], nn.Conv2d) and isinstance(self.downsample[1], nn.BatchNorm2d)
                for layer in reversed(self.downsample):
                    if hasattr(layer, 'modify_effective_sparsity') and not isinstance(layer, nn.BatchNorm2d):
                        print('shortcut')
                        shortcut, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            else:
                shortcut = connected_channels
            return torch.tensor(sorted(list(set(torch.cat([out, shortcut]).to('cpu').tolist()))), dtype=torch.int64).to(self.bn2.bias.device), None




class ResNet(nn.Module):

    def __init__(self, block, layer_list, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layers.BatchNorm2d
        self._norm_layer = norm_layer

        self.num_classes=num_classes
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layers.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layer_list[0])
        self.layer2 = self._make_layer(block, 128, layer_list[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layer_list[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layer_list[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = layers.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layers.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layers.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layer_list = []
        layer_list.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layer_list)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def set_num_out_channel(self, list_E, V_out_select):
        c = self.fc.get_num_out_channel(self.num_classes, list_E, V_out_select)
        for layer in reversed(self.layer4):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)
        for layer in reversed(self.layer3):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)
        for layer in reversed(self.layer2):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)
        for layer in reversed(self.layer1):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)
        c = self.conv1.get_num_out_channel(c, list_E, V_out_select)

    def set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        x = self.conv1.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
        x = 1.0 * (x != 0)
        x = self.maxpool(x)
        x = 1.0 * (x != 0)
        for layer in self.layer1:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                x = 1.0 * (x != 0)
        for layer in self.layer2:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                x = 1.0 * (x != 0)
        for layer in self.layer3:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                x = 1.0 * (x != 0)
        for layer in self.layer4:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                x = 1.0 * (x != 0)
        x = self.avgpool(x)
        x = 1.0 * (x != 0)
        x = x.view(x.size(0), -1)
        x = self.fc.forward_set_scores(x=x, scores=scores, shuffle_V_out=True, set_only_V_in=set_only_V_in)

    def set_effective_sparsity(self, is_forward, x=None):
        self.eval()
        assert self.training == False
        if is_forward:
            _, x = self.conv1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            x    = 1.0 * (x != 0)
            _, x = self.bn1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            x    = 1.0 * (x != 0)
            x    = self.maxpool(x)
            for layer in self.layer1:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    x    = 1.0 * (x != 0)
            for layer in self.layer2:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    x    = 1.0 * (x != 0)
            for layer in self.layer3:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    x    = 1.0 * (x != 0)
            for layer in self.layer4:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    x    = 1.0 * (x != 0)
            x    = self.avgpool(x)
            x    = 1.0 * (x != 0)
            x    = torch.flatten(x, 1)
            _, x = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)        
        else:
            connected_channels, _ = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=None)
            for layer in reversed(self.layer4):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            for layer in reversed(self.layer3):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            for layer in reversed(self.layer2):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            for layer in reversed(self.layer1):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            connected_channels, _ = self.conv1.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)








def _resnet(arch, block, layer_list, pretrained, progress, **kwargs):
    model = ResNet(block, layer_list, **kwargs)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                                   progress=progress)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def wide_resnet50_2(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
