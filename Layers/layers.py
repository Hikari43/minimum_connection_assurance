import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import numpy as np
from numba import jit, prange

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)        
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return F.linear(input, W, b)

    def get_num_out_channel(self, c, list_E, V_out_select, kernel_size=(1, 1)):
        print(f'Linear : set out channel')
        print(f'V_out select : {V_out_select}')
        self.num_E     = list_E[id(self.weight)]
        self.num_V_out = max(0, min([c, self.num_E, self.weight.data.size()[0]]))
        if kernel_size != (1, 1):
            print(f'NOTE : This Linear layer has pseudo_kernel_size!')
        self.pseudo_kernel_size = kernel_size
        out = 0

        if self.num_V_out == 0:
            print(f'num_V_out is zero!')
        elif (self.num_E / self.num_V_out) <= int(self.num_E / (kernel_size[0] * kernel_size[1])):
            if V_out_select   == 'max':
                out = int(self.num_E / (kernel_size[0] * kernel_size[1]))
            elif V_out_select == 'min':
                out = math.ceil(self.num_E / self.num_V_out)
            elif V_out_select == 'rand':
                out = torch.randint(math.ceil(self.num_E / self.num_V_out), int(self.num_E / (kernel_size[0] * kernel_size[1])) + 1, (1,)).item()
                print(f'rand : {math.ceil(self.num_E / self.num_V_out)} <= {out} <= {self.num_E / (kernel_size[0] * kernel_size[1])}')
            elif V_out_select == 'ignore':
                print(f'ignore : pass select V_out phase')
            else:
                raise ValueError
        else:
            out = math.ceil(self.num_E / self.num_V_out)

        if V_out_select == 'ignore':
            out = torch.randint(1, self.weight.data.size()[1] + 1, (1,)).item()
            print(f'ignore V_out range!')

        print(f'num_V_out : {self.num_V_out}')
        print(f'out : min({out}, {self.weight.data.size()[1]})')
        return min(out, self.weight.data.size()[1])
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Linear : ')
        x = x.view(x.size(0), -1)
        V_in_inds = x[0].nonzero()[:, 0]
        scores[id(self.weight)] = get_modified_scores(m=self, V_in_inds=V_in_inds, num_E=self.num_E, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        if not self.num_E < 1:
            zero = torch.tensor([0.]).to(self.weight_mask.device)
            one  = torch.tensor([1.]).to(self.weight_mask.device)
            self.weight_mask.copy_(torch.where(scores[id(self.weight)] >= self.num_E, zero, one))
            print(f'|E| :              {self.num_E}')
            print(f'|Input| :          {(x[0] != 0).sum().item()}')
            print(f'|Input cap V_in| : {((x[0] != 0) & (self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=0) != 0)).sum().item()}')
            print(f'valid |V_in| :     {(self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=0) != 0).sum().item()}')
            print(f'valid |V_out| :    {(self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=1) != 0).sum().item()}')
            if self.weight_mask.sum() == self.num_E:
                print(f'edges placed correctly. |E| = {self.num_E} = {self.weight_mask.sum()}')
            else:
                raise ValueError(f'|E| = {self.num_E}, but the number of successfully placed edges is {self.weight_mask.sum()}')
        else:
            print(f'|E| = 0!')
            self.weight_mask.copy_(torch.zeros_like(self.weight_mask).to(self.weight_mask.device))
        return F.linear(input=x, weight=self.weight_mask, bias=None) 

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None, pseudo_kernel_size = (1, 1)):
        print(f'Linear :')
        if is_forward:
            modified_mask = torch.zeros_like(self.weight_mask).to(self.weight_mask.device)
            V_in_inds = x[0].nonzero()[:, 0]
            modified_mask[:, V_in_inds] = 1

            print(f'Remaining param : {self.weight_mask.sum().item()} -> {(self.weight_mask * (self.weight != 0) * modified_mask).sum().item()}')
            self.weight_mask = self.weight_mask * (self.weight != 0) * modified_mask 

            if self.bias != None:
                return None, F.linear(input=x, weight=self.weight_mask, bias=1.0 * (self.bias != 0))
            else:
                return None, F.linear(input=x, weight=self.weight_mask, bias=None)
        else:
            k = np.prod(pseudo_kernel_size)                
            if isinstance(connected_channels, torch.Tensor):
                modified_mask = torch.zeros_like(self.weight_mask).to(self.weight_mask.device)
                modified_mask[connected_channels, :] = 1
                print(f'Remaining param : {self.weight_mask.sum().item()} -> {(self.weight_mask * modified_mask).sum().item()}')
                self.weight_mask = self.weight_mask * (self.weight != 0) * modified_mask
            if k != 1:
                print(f'Pseudo kernel size : {pseudo_kernel_size}')
                out = torch.zeros(self.weight_mask.size()[0], self.weight_mask.size()[1]).to(self.weight_mask.device)
                for i in range(self.weight_mask.size()[1]):
                    out[:, i] = self.weight_mask[:, i * k:(i + 1) * k].sum(dim=1)
                return (out.sum(dim=0) != 0).nonzero()[:, 0].to(torch.int64), None
            else:
                return (self.weight_mask.sum(dim=0) != 0).nonzero()[:, 0].to(torch.int64), None

    def calculate_path_prob(self, x, in_d):
        print(f'in_d : {in_d.item()}')
        c_in = x.size()[1]
        density = self.weight_mask.sum() / self.weight_mask.numel()
        out_d = 1 - (1 - in_d * density)**c_in
        x = F.linear(input=x, weight = self.weight_mask, bias=None)
        print(f'out_d : {out_d.item()}')
        return x, out_d
    
    def forward_check_V_in_loss(self, x):
        print(f'Linear : ')
        x = x.view(x.size(0), -1)
        assert x.size(0) == 1
        n_loss = (x[0] != 0).sum() - ((x[0] != 0) * (self.weight_mask.sum(dim=0) != 0)).sum()
        n_loss = n_loss.item()
        return F.linear(input=x, weight=self.weight_mask, bias=None), n_loss



class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias, padding_mode)
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        W = self.weight_mask * self.weight
        if self.bias is not None:
            b = self.bias_mask * self.bias
        else:
            b = self.bias
        return self._conv_forward(input, W, b)
    
    def get_num_out_channel(self, c, list_E, V_out_select):
        print(f'Conv2d : set out channel')
        print(f'V_out select : {V_out_select}')
        self.num_E     = list_E[id(self.weight)]
        self.num_V_out = max(0, min([c, self.num_E, self.weight.data.size()[0]]))
        out = 0

        if self.num_V_out == 0:
            print(f'num_V_out is zero!')
        elif (self.num_E / self.num_V_out) <= int(self.num_E / (self.weight.data.size()[2] * self.weight.data.size()[3])):
            if V_out_select == 'max':
                out = int(self.num_E / (self.weight.data.size()[2] * self.weight.data.size()[3]))
            elif V_out_select == 'min':
                out = math.ceil(self.num_E / self.num_V_out)
            elif V_out_select == 'rand':
                out = torch.randint(math.ceil(self.num_E / self.num_V_out), int(self.num_E / (self.weight.data.size()[2] * self.weight.data.size()[3])) + 1, (1,)).item()
                print(f'rand : {math.ceil(self.num_E / self.num_V_out)} <= {out} <= {int(self.num_E / (self.weight.data.size()[2] * self.weight.data.size()[3]))}')
            elif V_out_select == 'ignore':
                print(f'ignore : pass select V_out phase')
            else:
                raise ValueError
        else:
            out = math.ceil(self.num_E / self.num_V_out)

        if V_out_select == 'ignore':
            out = torch.randint(1, self.weight.data.size()[1] + 1, (1,)).item()
            print(f'ignore V_out range!')

        print(f'num_V_out : {self.num_V_out}')
        print(f'out : min({out}, {self.weight.data.size()[1]})')
        return min(out, self.weight.data.size()[1])

    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Conv2d : ')
        stride      = self.stride[0]
        padding     = self.padding[0]
        dilation    = self.dilation[0]
        groups      = self.groups
        kernel_size = self.kernel_size[0]
        if (self.stride[0] != self.stride[1]) \
        or (self.padding[0] != self.padding[1]) \
        or (self.dilation[0] != self.dilation[1] or self.dilation != (1, 1)) \
        or (self.groups != 1):
            raise NotImplementedError      

        deg_w = get_degree_weight(
            x, 
            kernel_size= kernel_size, 
            stride     = stride, 
            padding    = padding
            ).reshape((-1,))
        V_in_inds = deg_w.nonzero()[:, 0]
        scores[id(self.weight)] = get_modified_scores(m=self, V_in_inds=V_in_inds, num_E=self.num_E, shuffle_V_out=shuffle_V_out, set_only_V_in=set_only_V_in)
        if not self.num_E < 1:
            zero = torch.tensor([0.]).to(self.weight_mask.device)
            one  = torch.tensor([1.]).to(self.weight_mask.device)
            self.weight_mask.copy_(torch.where(scores[id(self.weight)] >= self.num_E, zero, one))
            print(f'|E| :              {self.num_E}')
            print(f'|Input| :          {(deg_w != 0).sum().item()}')
            print(f'|Input cap V_in| : {((deg_w.to(self.weight.device) != 0) & (self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=0) != 0)).sum().item()}')
            print(f'valid |V_in| :     {(self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=0) != 0).sum().item()}')
            print(f'valid |V_out| :    {(self.weight_mask.reshape(self.weight_mask.size()[0], -1).sum(dim=1) != 0).sum().item()}')                
            if self.weight_mask.sum() == self.num_E:
                print(f'edges placed correctly. |E| = {self.num_E} = {self.weight_mask.sum()}')
            else:
                raise ValueError(f'|E| = {self.num_E}, but the number of successfully placed edges is {self.weight_mask.sum()}')
        else:
            print(f'|E| = 0!')
            self.weight_mask.copy_(torch.zeros_like(self.weight_mask).to(self.weight_mask.device))
        return F.conv2d(
            input=x, weight=self.weight_mask, bias=None, stride=self.stride, 
            padding=self.padding, dilation=self.dilation, groups=self.groups)

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        print(f'Conv2d :')
        if is_forward:
            stride      = self.stride[0]
            padding     = self.padding[0]
            dilation    = self.dilation[0]
            groups      = self.groups
            kernel_size = self.kernel_size[0]
            if (self.stride[0] != self.stride[1]) \
            or (self.padding[0] != self.padding[1]) \
            or (self.dilation[0] != self.dilation[1] or self.dilation != (1, 1)) \
            or (self.groups != 1):
                raise NotImplementedError      

            deg_w = get_degree_weight(
                x, 
                kernel_size= kernel_size, 
                stride     = stride, 
                padding    = padding
                ).reshape((-1,))
            print(f'{self.weight.size()=}')
            V_in_inds = deg_w.nonzero()[:, 0]
            modified_mask = torch.zeros_like(self.weight_mask).to(self.weight_mask.device)
            modified_mask = modified_mask.reshape((modified_mask.size()[0], -1))
            modified_mask[:, V_in_inds] = 1

            modified_mask = modified_mask.reshape(self.weight_mask.size())
            print(f'Remaining param : {self.weight_mask.sum().item()} -> {(self.weight_mask * (self.weight != 0) * modified_mask).sum().item()}')
            self.weight_mask = self.weight_mask * (self.weight != 0) * modified_mask 

            if self.bias != None:
                return None, F.conv2d(
                    input=x, weight=self.weight_mask, bias=1.0 * (self.bias != 0), stride=self.stride, 
                    padding=self.padding, dilation=self.dilation, groups=self.groups)
            else:
                return None, F.conv2d(
                    input=x, weight=self.weight_mask, bias=None, stride=self.stride, 
                    padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            k   = self.weight_mask.size()[2] * self.weight_mask.size()[3]
            # print(self.weight_mask.size())
            # print(k)
            if isinstance(connected_channels, torch.Tensor):
                modified_mask = torch.zeros_like(self.weight_mask).to(self.weight_mask.device)
                modified_mask = modified_mask.reshape((modified_mask.size()[0], -1))
                modified_mask[connected_channels, :] = 1
                modified_mask = modified_mask.reshape(self.weight_mask.size())
                print(f'Remaining param : {self.weight_mask.sum().item()} -> {(self.weight_mask * modified_mask).sum().item()}')
                self.weight_mask = self.weight_mask * (self.weight != 0) * modified_mask
            reshaped_mask = self.weight_mask.reshape((self.weight_mask.size()[0], -1))
            out = torch.zeros(self.weight_mask.size()[0], self.weight_mask.size()[1]).to(self.weight_mask.device)
            for i in range(self.weight_mask.size()[1]):
                out[:, i] = reshaped_mask[:, i * k:(i + 1) * k].sum(dim=1)
            return (out.sum(dim=0) != 0).nonzero()[:, 0].to(torch.int64), None

    def calculate_path_prob(self, x, in_d):
        if isinstance(in_d, torch.Tensor):
            print(f'in_d : {in_d.item()}')
        else:
            print(f'in_d : {in_d}')
        c_in = x.size()[1]
        h = x.size()[2] + 2 * self.padding[0]
        w = x.size()[3] + 2 * self.padding[1]
        density = self.weight_mask.sum() / self.weight_mask.numel()
        in_d = (in_d * x.size()[1] * x.size()[2] * x.size()[3]) / (c_in * h * w) 
        if isinstance(in_d, torch.Tensor):
            print(f'after padding in_d : {in_d.item()}')
        else:
            print(f'after padding in_d : {in_d}')
        out_d = 1 - (1 - in_d * density)**(c_in * self.kernel_size[0] * self.kernel_size[1])
        x = F.conv2d(
            input=x, weight=self.weight_mask, bias=None, stride=self.stride, 
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        print(f'out_d : {out_d.item()}')
        return x, out_d

    def forward_check_V_in_loss(self, x):
        print(f'Conv2d : ')
        stride      = self.stride[0]
        padding     = self.padding[0]
        dilation    = self.dilation[0]
        groups      = self.groups
        kernel_size = self.kernel_size[0]
        if (self.stride[0] != self.stride[1]) \
        or (self.padding[0] != self.padding[1]) \
        or (self.dilation[0] != self.dilation[1] or self.dilation != (1, 1)) \
        or (self.groups != 1):
            raise NotImplementedError      

        deg_w = get_degree_weight(
            x, 
            kernel_size= kernel_size, 
            stride     = stride, 
            padding    = padding
            ).reshape((-1,)).to(x.device)
        n_loss = (deg_w != 0).sum() - ((deg_w != 0) * (self.weight_mask.reshape((self.weight_mask.size()[0], -1)).sum(dim=0) != 0)).sum()
        n_loss = n_loss.item()
        return F.conv2d(
            input=x, weight=self.weight_mask, bias=None, stride=self.stride, 
            padding=self.padding, dilation=self.dilation, groups=self.groups), n_loss



class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        if self.affine:     
            self.register_buffer('weight_mask', torch.ones(self.weight.shape))
            self.register_buffer('bias_mask', torch.ones(self.bias.shape))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.affine:
            W = self.weight_mask * self.weight
            b = self.bias_mask * self.bias
        else:
            W = self.weight
            b = self.bias

        return F.batch_norm(
            input, self.running_mean, self.running_var, W, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def modify_effective_sparsity(self, connected_channels=None, is_forward=True, x=None):
        if is_forward:
            x    = x * (1.0 * (self.weight.data.reshape(1, x.size()[1], 1, 1) != 0))
            x    = 1.0 * (x != 0)
            x = x + (1.0 * ((-self.weight.data * self.running_mean.data / (self.running_var + self.eps).sqrt() + self.bias.data).reshape(1, x.size()[1], 1, 1) != 0))
            x    = 1.0 * (x != 0)
            return None, x
        else:
            raise NotImplementedError("You don't have to use this function when is_forward=False.")


class Identity1d(nn.Module):
    def __init__(self, num_features):
        super(Identity1d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W


class Identity2d(nn.Module):
    def __init__(self, num_features):
        super(Identity2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features, 1, 1))
        self.register_buffer('weight_mask', torch.ones(self.weight.shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        W = self.weight_mask * self.weight
        return input * W

    def get_num_out_channel(self, c, list_E, V_out_select):
        self.num_E     = 0
        self.num_V_out = c
        return c
    
    def forward_set_scores(self, x, scores, shuffle_V_out=False, set_only_V_in=False):
        print(f'Identity2d : ')
        return x

    def modify_effective_sparsity(self, connected_channels=None):
        return connected_channels


def get_modified_scores(m, V_in_inds, num_E, shuffle_V_out=False, set_only_V_in=False):
    weight_size = np.prod(m.weight.size())
    scores = torch.randperm(weight_size, dtype=torch.int64).reshape(m.weight.size()).to(m.weight.device)
    scores = scores.reshape((scores.size()[0], -1))

    if set_only_V_in and len(V_in_inds) != 0: 
        print(f'set only V_in')
        # Increase the connection priority of input nodes
        # Note: MiCA implementations are pruned by the higher score.
        #       Different from implementations such as SynFlow.
        scores = scores + scores[:, V_in_inds].numel()
        scores[:, V_in_inds] = torch.randperm(scores[:, V_in_inds].numel(), dtype=torch.int64).reshape(scores[:, V_in_inds].size()).to(m.weight.device)

    elif len(V_in_inds) != 0: 
        print(f'mica process')
        # Increase the connection priority of input nodes
        scores = scores + scores[:, V_in_inds].numel()
        scores[:, V_in_inds] = torch.randperm(scores[:, V_in_inds].numel(), dtype=torch.int64).reshape(scores[:, V_in_inds].size()).to(m.weight.device)

        # Select output nodes
        num_V_out  = int(max(1, min([scores.size()[0], num_E, m.num_V_out])))
        if shuffle_V_out:
            V_out_inds = torch.randperm(scores.size()[0])[:num_V_out].to(m.weight.device)
            print(f'After Shuffling V_out ind : {V_out_inds}')
        else:
            V_out_inds = torch.arange(num_V_out).to(m.weight.device)
        
        # Increase the connection priority of selected output nodes
        scores = scores + num_V_out * V_in_inds.numel()
        scores[V_out_inds.expand(V_in_inds.numel(), -1).flatten(), V_in_inds.repeat_interleave(num_V_out)] \
            = torch.randperm(num_V_out * V_in_inds.numel(), dtype=torch.int64).to(m.weight.device)
        
        # (i to iii) MiCA steps
        # Increase the connection priority of input and output node combinations
        modified_V_in_non_zero_inds = V_in_inds[torch.randperm(V_in_inds.numel())].to(m.weight.device)
        if isinstance(m, nn.Conv2d) or (hasattr(m, 'pseudo_kernel_size') and m.pseudo_kernel_size != (1, 1)):
            # (i)
            if isinstance(m, nn.Linear):
                print(f'This is Linear layer')
                print(f'Use pseudo_kernel_size : {m.pseudo_kernel_size}')
                kernel = np.prod(m.pseudo_kernel_size)
            else:
                kernel = np.prod(m.kernel_size)
            override_inds = -1 * torch.ones(int(scores.size()[1] / kernel), device=m.weight.device)
            override_bools = torch.ones(modified_V_in_non_zero_inds.numel(), device=m.weight.device)
            for i in range(modified_V_in_non_zero_inds.numel()):
                if override_inds[int(modified_V_in_non_zero_inds[i] / kernel)] == -1:
                    override_inds[int(modified_V_in_non_zero_inds[i] / kernel)] = modified_V_in_non_zero_inds[i]
                    override_bools[i] = 0
                if (override_inds > -1).all():
                    break
            override_inds = override_inds[override_inds != -1]
            modified_V_in_non_zero_inds = torch.cat((override_inds, modified_V_in_non_zero_inds.masked_select(override_bools==1))).to(dtype=torch.int).to(m.weight.device)
        modified_V_out_inds = V_out_inds
        if num_V_out > V_in_inds.numel():
            prob_dist = torch.zeros(scores.size()[1]).to(m.weight.device)
            prob_dist[V_in_inds] = 1
            prob_dist = prob_dist / prob_dist.sum()
            modified_V_in_non_zero_inds = torch.cat((modified_V_in_non_zero_inds, torch.multinomial(prob_dist, num_V_out - V_in_inds.numel(), replacement=True).to(m.weight.device)))
        elif num_V_out < V_in_inds.numel():
            prob_dist = torch.zeros(scores.size()[0]).to(m.weight.device)
            prob_dist[V_out_inds] = 1
            prob_dist = prob_dist / prob_dist.sum()
            modified_V_out_inds = torch.cat((modified_V_out_inds, torch.multinomial(prob_dist, V_in_inds.numel() - num_V_out, replacement=True).to(m.weight.device)))
        scores = scores + modified_V_out_inds.numel()
        scores[modified_V_out_inds, modified_V_in_non_zero_inds] = torch.arange(0, modified_V_out_inds.numel(), dtype=torch.int64).to(m.weight.device)

    scores = scores.reshape((-1,))
    scores[scores.argsort()] = torch.arange(scores.numel(), dtype=torch.int64, device=scores.device)
    scores = scores.reshape(m.weight.size())

    return scores


def get_degree_weight(x, kernel_size, stride, padding):
    in_channels, in_height, in_width = x.shape[1:]
    out_height = int((in_height + 2 * padding - kernel_size) / stride + 1)
    out_width  = int((in_width + 2 * padding - kernel_size)  / stride + 1)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding)).to('cpu').detach().numpy()
    out = jit_executor(
        x_padded, 
        in_channels,
        out_height, 
        out_width,
        kernel_size,
        stride,
        padding
        )
    return torch.from_numpy(out.astype(np.float32)).clone()
    
@jit('f4[:, :, :](f4[:, :, :, :], i8, i8, i8, i8, i8)', nopython=True)
def out_width_loop(x, h_out, out_width, in_channels, stride, kernel_size):
    w_out_patch = np.zeros((in_channels, kernel_size, kernel_size), dtype='float32') 
    for w_out in range(out_width):
        h_start = h_out * stride
        w_start = w_out * stride
        h_end = h_start + kernel_size
        w_end = w_start + kernel_size
        w_out_patch += x[0, :, h_start:h_end, w_start:w_end]
    return w_out_patch # size: (in_c, k, k)

@jit('f4[:, :, :](f4[:, :, :, :], i8, i8, i8, i8, i8, i8)', nopython=True, parallel = True)
def jit_executor(x, in_channels, out_height, out_width, kernel_size, stride, padding):
    h_w_out_patch = np.zeros((in_channels, kernel_size, kernel_size), dtype='float32') 
    for h_out in prange(out_height):
        h_w_out_patch += out_width_loop(x, h_out, out_width, in_channels, stride, kernel_size)
    return h_w_out_patch # size: (in_c, k, k)

