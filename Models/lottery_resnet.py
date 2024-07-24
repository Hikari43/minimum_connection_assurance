# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import layers
import copy


class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample=False, use_skip_connection=True):
        super(Block, self).__init__()
        self.use_skip_connection = use_skip_connection
        stride = 2 if downsample else 1
        self.conv1 = layers.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layers.BatchNorm2d(f_out)
        self.conv2 = layers.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layers.BatchNorm2d(f_out)

        # No parameters for shortcut connections.
        if self.use_skip_connection:
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    layers.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    layers.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = layers.Identity2d(f_in)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip_connection:
            out += self.shortcut(x)
        return F.relu(out)
    
    def get_num_out_channel(self, c, list_E, V_out_select):
        shortcut_param = 0
        main_param     = 0
        if self.use_skip_connection:
            if isinstance(self.shortcut, nn.Sequential):
                for layer in reversed(self.shortcut):
                    if hasattr(layer, 'get_num_out_channel'):
                        shortcut = layer.get_num_out_channel(c, list_E, V_out_select)
            else:
                shortcut = self.shortcut.get_num_out_channel(c, list_E, V_out_select)
        c  = self.conv2.get_num_out_channel(c, list_E, V_out_select)
        c  = self.conv1.get_num_out_channel(c, list_E, V_out_select)
        if self.use_skip_connection:
            return max(shortcut, c)
        else:
            return c
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Branch')
        print(f'Main 1')
        out = self.conv1.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        print(f'Main 2')
        out = self.conv2.forward_set_scores(x=out, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        out = 1.0 * (out != 0)
        if self.use_skip_connection:
            if isinstance(self.shortcut, nn.Sequential):
                for layer in self.shortcut:
                    if hasattr(layer, 'forward_set_scores'):
                        print(f'Downsample')
                        out += layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
                        out = 1.0 * (out != 0)
            else:
                print(f'Skip connection')
                out += self.shortcut.forward_set_scores(x=x, scores=scores, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
                out = 1.0 * (out != 0)
        return out

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None, use_skip_connection=True):
        if is_forward:
            _, out_x = self.conv1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn1.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.conv2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            _, out_x = self.bn2.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=out_x)
            out_x    = 1.0 * (out_x != 0)
            if use_skip_connection:
                if isinstance(self.shortcut, nn.Sequential):
                    assert len(self.shortcut) == 2
                    assert isinstance(self.shortcut[0], nn.Conv2d)
                    print('shortcut')
                    _, shortcut = self.shortcut[0].modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
                    shortcut    = 1.0 * (shortcut != 0)
                    assert isinstance(self.shortcut[1], nn.BatchNorm2d)
                    _, shortcut = self.shortcut[1].modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=shortcut)
                    shortcut    = 1.0 * (shortcut != 0)
                else:
                    shortcut = x
                return None, 1.0 * ((out_x + shortcut) != 0)
            else:
                return None, out_x
        else:
            print(f'out2')
            out, _ = self.conv2.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
            print(f'out1')
            out, _ = self.conv1.modify_effective_sparsity(connected_channels=out, is_forward=is_forward, x=None)
            if use_skip_connection:
                if isinstance(self.shortcut, nn.Sequential):
                    print('shortcut')
                    assert len(self.shortcut) == 2
                    assert isinstance(self.shortcut[0], nn.Conv2d)
                    shortcut, _ = self.shortcut[0].modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)
                else:
                    shortcut = connected_channels
                return torch.tensor(sorted(list(set(torch.cat([out, shortcut]).to('cpu').tolist()))), dtype=torch.int64).to(self.bn2.bias.device), None
            else:
                return out, None

    def calculate_path_prob(self, x, in_d):
        print(f'conv1')
        out, mid_d = self.conv1.calculate_path_prob(x, in_d)
        print(f'conv2')
        out, mid_d = self.conv2.calculate_path_prob(out, mid_d)
        if isinstance(self.shortcut, nn.Sequential):
            for layer in self.shortcut:
                if hasattr(layer, 'calculate_path_prob'):
                    print('shortcut')
                    shortcut, shortcut_d = layer.calculate_path_prob(x, in_d)
        else:
            shortcut, shortcut_d = x, in_d
        return out + shortcut, 1 - ((1 - in_d) + in_d * (1 - mid_d) * (1 - shortcut_d))

    def forward_check_V_in_loss(self, x, use_skip_connection=True, get_sparsities=False):
        n_losses = []
        n_zeros  = 0
        n_params = 0
        print(f'Branch')
        print(f'Main 1')
        n_zeros += (x == 0).sum()
        n_params += x.numel()
        out, n_loss = self.conv1.forward_check_V_in_loss(x)
        n_losses.append(n_loss)
        out = 1.0 * (out != 0)
        print(f'Main 2')
        n_zeros += (out == 0).sum()
        n_params += out.numel()
        out, n_loss = self.conv2.forward_check_V_in_loss(out)
        n_losses.append(n_loss)
        out = 1.0 * (out != 0)
        if use_skip_connection:
            if isinstance(self.shortcut, nn.Sequential):
                for layer in self.shortcut:
                    if hasattr(layer, 'forward_check_V_in_loss'):
                        print(f'Downsample')
                        n_zeros += (x == 0).sum()
                        n_params += x.numel()
                        sout, n_loss = layer.forward_check_V_in_loss(x)
                        n_losses.append(n_loss)
                        out += sout
                        out = 1.0 * (out != 0)
            else:
                print(f'Skip connection')
                out += x
                out = 1.0 * (out != 0)
        if get_sparsities:
            return out, n_losses, n_zeros, n_params
        else:
            return out, n_losses

class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""
    
    def __init__(self, plan, num_classes, dense_classifier, use_skip_connection=True):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = layers.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = layers.BatchNorm2d(current_filters)
        if not use_skip_connection:
            print(f'This model do not use skip connection.')

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample, use_skip_connection))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = layers.Linear(plan[-1][0], num_classes)
        if dense_classifier:
            self.fc = nn.Linear(plan[-1][0], num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def set_num_out_channel(self, list_E, V_out_select):
        c = self.fc.get_num_out_channel(self.num_classes, list_E, V_out_select)
        for layer in reversed(self.blocks):
            if hasattr(layer, 'get_num_out_channel'):
                c = layer.get_num_out_channel(c, list_E, V_out_select)
        c = self.conv.get_num_out_channel(c, list_E, V_out_select)

    def set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        x = self.conv.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
        x = 1.0 * (x != 0)
        for layer in self.blocks:
            if hasattr(layer, 'forward_set_scores'):
                x = layer.forward_set_scores(x=x, scores=scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)
                x = 1.0 * (x != 0)
        x = F.avg_pool2d(x, x.size()[3])
        x = 1.0 * (x != 0)
        x = x.view(x.size(0), -1)
        x = self.fc.forward_set_scores(x=x, scores=scores, shuffle_V_out=True, set_only_V_in=set_only_V_in)

    def set_effective_sparsity(self, is_forward, x=None, use_skip_connection=True):
        self.eval()
        assert self.training == False
        if is_forward:
            _, x = self.conv.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            x    = 1.0 * (x != 0)
            _, x = self.bn.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)
            x    = 1.0 * (x != 0)
            for layer in self.blocks:
                if hasattr(layer, 'modify_effective_sparsity'):
                    _, x = layer.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x, use_skip_connection=use_skip_connection)
                    x    = 1.0 * (x != 0)
            x    = F.avg_pool2d(x, x.size()[3])
            x    = 1.0 * (x != 0)
            x    = x.view(x.size(0), -1)
            _, x = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=x)        
        else:
            connected_channels, _ = self.fc.modify_effective_sparsity(connected_channels=None, is_forward=is_forward, x=None)
            for layer in reversed(self.blocks):
                if hasattr(layer, 'modify_effective_sparsity'):
                    connected_channels, _ = layer.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None, use_skip_connection=use_skip_connection)
            connected_channels, _ = self.conv.modify_effective_sparsity(connected_channels=connected_channels, is_forward=is_forward, x=None)

    def calculate_path_prob(self, x, in_d):
        x, in_d = self.conv.calculate_path_prob(x, in_d)
        for layer in self.blocks:
            if hasattr(layer, 'calculate_path_prob'):
                x, in_d = layer.calculate_path_prob(x, in_d)
        in_d = in_d = 1 - (1 - in_d)**x.size()[3]
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        return self.fc.calculate_path_prob(x, in_d)

    def check_V_in_loss(self, x, use_skip_connection=True, get_sparsities=False):
        all_losses = []
        n_zeros  = 0
        n_params = 0 
        n_zeros += (x == 0).sum()
        n_params += x.numel()
        x, n_loss = self.conv.forward_check_V_in_loss(x)
        all_losses.append(n_loss)
        x = 1.0 * (x != 0)
        for layer in self.blocks:
            if hasattr(layer, 'forward_check_V_in_loss'):
                if get_sparsities:
                    x, n_losses, n_z, n_p = layer.forward_check_V_in_loss(x, use_skip_connection=use_skip_connection, get_sparsities=get_sparsities)
                    n_zeros  += n_z
                    n_params += n_p
                else:
                    x, n_losses = layer.forward_check_V_in_loss(x, use_skip_connection=use_skip_connection, get_sparsities=get_sparsities)
                all_losses.extend(n_losses)
                x = 1.0 * (x != 0)
        x = F.avg_pool2d(x, x.size()[3])
        x = 1.0 * (x != 0)
        x = x.view(x.size(0), -1)
        n_zeros  += (x == 0).sum()
        n_params += x.numel()
        x, n_loss = self.fc.forward_check_V_in_loss(x)
        all_losses.append(n_loss)
        if get_sparsities:
            return all_losses, n_zeros, n_params
        else:
            return all_losses

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _plan(D, W):
    """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

    The ResNet is structured as an initial convolutional layer followed by three "segments"
    and a linear output layer. Each segment consists of D blocks. Each block is two
    convolutional layers surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and each layer in the
    third segment has 64W filters.

    The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
    N is the total number of layers in the network: 2 + 6D.
    The default value of W is 16 if it isn't provided.

    For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
    linear layer, there are 18 convolutional layers in the blocks. That means there are nine
    blocks, meaning there are three blocks per segment. Hence, D = 3.
    The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
    """
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]

    return plan

def _resnet(arch, plan, num_classes, dense_classifier, pretrained, use_skip_connection=True):
    model = ResNet(plan, num_classes, dense_classifier, use_skip_connection)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# ResNet Models
def resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False, use_skip_connection=True):
    plan = _plan(20, 16)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained, use_skip_connection)

def resnet32(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 16)
    return _resnet('resnet32', plan, num_classes, dense_classifier, pretrained)

def resnet44(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 16)
    return _resnet('resnet44', plan, num_classes, dense_classifier, pretrained)

def resnet56(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 16)
    return _resnet('resnet56', plan, num_classes, dense_classifier, pretrained)

def resnet110(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 16)
    return _resnet('resnet110', plan, num_classes, dense_classifier, pretrained)

def resnet1202(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 16)
    return _resnet('resnet1202', plan, num_classes, dense_classifier, pretrained)

# Wide ResNet Models
def wide_resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 32)
    return _resnet('wide_resnet20', plan, num_classes, dense_classifier, pretrained)

def wide_resnet32(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 32)
    return _resnet('wide_resnet32', plan, num_classes, dense_classifier, pretrained)

def wide_resnet44(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 32)
    return _resnet('wide_resnet44', plan, num_classes, dense_classifier, pretrained)

def wide_resnet56(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 32)
    return _resnet('wide_resnet56', plan, num_classes, dense_classifier, pretrained)

def wide_resnet110(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 32)
    return _resnet('wide_resnet110', plan, num_classes, dense_classifier, pretrained)

def wide_resnet1202(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 32)
    return _resnet('wide_resnet1202', plan, num_classes, dense_classifier, pretrained)