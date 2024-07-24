# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import layers

class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))
    
    def get_num_out_channel(self, c, list_E, V_out_select):
        return self.conv.get_num_out_channel(c, list_E, V_out_select)

    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        x = self.conv.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        return 1.0 * (x != 0)
    
    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        raise NotImplementedError
    

class ConvBNModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.bn = layers.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
    
    def get_num_out_channel(self, c, list_E, V_out_select):
        return self.conv.get_num_out_channel(c, list_E, V_out_select)
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        x = self.conv.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        return 1.0 * (x != 0)
    
    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        if is_forward:
            assert self.bn.weight.data.dim() == 1
            _, x = self.conv.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            x    = 1.0 * (x != 0)
            _, x = self.bn.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            return None, x
        else:
            out, _ = self.conv.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            return out, None

    def calculate_path_prob(self, x, in_d):
        return self.conv.calculate_path_prob(x, in_d)
    
    def forward_check_V_in_loss(self, x):
        x, n_loss = self.conv.forward_check_V_in_loss(x)
        _, x      = self.bn.modify_effective_sparsity(connected_channels=None, is_forward=True, x=x)
        return x, n_loss

class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3
        self.num_classes = num_classes
        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        # self.avgpool2d = nn.AvgPool2d(2)
        if num_classes == 1000: # imagenet
            self.fc = layers.Linear(512*7*7, num_classes)
        else:
            self.fc = layers.Linear(512, num_classes)
        if dense_classifier:
            raise NotImplementedError
            self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # x = nn.AvgPool2d(2)(x)
        # x = self.avgpool2d(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def set_num_out_channel(self, list_E, V_out_select):
        if self.num_classes == 1000:
            c = self.fc.get_num_out_channel(self.num_classes, list_E, V_out_select, (7, 7))
        else:    
            c = self.fc.get_num_out_channel(self.num_classes, list_E, V_out_select)
        for layer in reversed(self.layers):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)

    def set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        for layer in self.layers:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                assert not (torch.isinf(x).all() or torch.isnan(x).all())
                x = 1.0 * (x != 0)
            elif isinstance(layer, nn.MaxPool2d):
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x) 
                x = 1.0 * (x != 0)
        print(f'before avg : {x.size()=}')
        # x = self.avgpool2d(x)
        x = F.avg_pool2d(x, x.size()[3]) 
        print(f'after avg : {x.size()=}')
        x = 1.0 * (x != 0)
        x = x.view(x.size(0), -1)
        x = self.fc.forward_set_scores(x, scores, shuffle_V_out=True, set_only_V_in=set_only_V_in)

    def set_effective_sparsity(self, is_forward, x=None, pseudo_kernel_size=(1, 1)):
        self.eval()
        assert self.training == False
        if is_forward:
            for layer in self.layers:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    assert not (torch.isinf(x).all() or torch.isnan(x).all())
                    x = 1.0 * (x != 0)
                elif isinstance(layer, nn.MaxPool2d):
                    x = nn.MaxPool2d(kernel_size=2, stride=2)(x) 
                    x = 1.0 * (x != 0)
            # x    = self.avgpool2d(x)
            x = F.avg_pool2d(x, x.size()[3]) 
            x    = 1.0 * (x != 0)
            x    = x.view(x.size(0), -1)
            _, x = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
        else:
            connected_channels, _ = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=None, pseudo_kernel_size=pseudo_kernel_size)
            for layer in reversed(self.layers):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)

    def calculate_path_prob(self, x, in_d):
        for layer in self.layers:
            if hasattr(layer, 'calculate_path_prob'):
                x, in_d = layer.calculate_path_prob(x, in_d)
            elif isinstance(layer, nn.MaxPool2d):
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
                in_d = 1 - (1 - in_d)**4
        # x = self.avgpool2d(x)
        x = F.avg_pool2d(x, x.size()[3])
        in_d = in_d = 1 - (1 - in_d)**4
        x = x.view(x.size(0), -1)
        return self.fc.calculate_path_prob(x, in_d)

    def check_V_in_loss(self, x, get_sparsities=False):
        n_losses = []
        n_zeros  = 0
        n_params = 0
        for layer in self.layers:
            if hasattr(layer, 'forward_set_scores'):
                n_zeros += (x == 0).sum()
                n_params += x.numel()
                x, n_loss = layer.forward_check_V_in_loss(x)
                n_losses.append(n_loss)
                assert not (torch.isinf(x).all() or torch.isnan(x).all())
                x = 1.0 * (x != 0)
            elif isinstance(layer, nn.MaxPool2d):
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x) 
                x = 1.0 * (x != 0)
        # x = self.avgpool2d(x)
        x = F.avg_pool2d(x, x.size()[3]) 
        x = 1.0 * (x != 0)
        x = x.view(x.size(0), -1)
        n_zeros += (x == 0).sum()
        n_params += x.numel()
        x, n_loss = self.fc.forward_check_V_in_loss(x)
        n_losses.append(n_loss)
        if get_sparsities:
            return n_losses, n_zeros, n_params
        return n_losses

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _plan(num):
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg11(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg13(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg13_bn(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg16(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg19(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg19_bn(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)
