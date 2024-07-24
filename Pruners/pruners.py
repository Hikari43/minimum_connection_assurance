import torch
import torch.nn as nn
import numpy as np
from numba import jit, prange
import torch.nn.functional as F
import math
import _ctypes
import copy
from tqdm import tqdm
from Utils import generator

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
        raise NotImplementedError

    def _global_mask(self, density):
        r"""Updates masks of model with scores by density level globally.
        """
        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - density) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            # print(f'Number of duplicate scores : {(global_scores == threshold).sum()}')
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
                # print(f'|E| = {mask.sum()}')
    
    def _local_mask(self, density):
        r"""Updates masks of model with scores by density level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - density) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, density, scope):
        r"""Updates masks of model with scores by density according to scope.
        """
        if scope == 'global':
            self._global_mask(density)
        if scope == 'local':
            self._local_mask(density)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self, shuffle_vertices=False):
        print(f'shuffle the mask')
        if shuffle_vertices:
            print(f'shuffle vertices')
            for mask, param in self.masked_parameters:
                m = mask.sum()
                reshaped_mask = mask.reshape(mask.size()[0], -1)
                V_out, V_in = reshaped_mask.size() 
                perm_in  = torch.randperm(V_in)
                perm_out = torch.randperm(V_out)
                reshaped_mask = reshaped_mask[:, perm_in]
                reshaped_mask = reshaped_mask[perm_out, :]
                if m != reshaped_mask.sum():
                    raise ValueError
                mask.copy_(reshaped_mask.reshape(mask.size()))
        else:
            for mask, param in self.masked_parameters:
                m = mask.sum()
                shape = mask.shape
                perm = torch.randperm(mask.nelement())
                mask.copy_(mask.reshape(-1)[perm].reshape(shape))

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data * m.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad * m.data).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
    
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

class EdgePopup(Pruner):
    def __init__(self, masked_parameters):
        super(EdgePopup, self).__init__(masked_parameters)
        self.signed_scores = {}
        for m, p in self.masked_parameters:
            self.signed_scores[id(p)] = torch.ones_like(m)
            nn.init.kaiming_normal_(self.signed_scores[id(p)])
            self.scores[id(p)] = self.signed_scores[id(p)].abs()

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
        # compute gradient
        fixed_masks = {}
        for m, p in self.masked_parameters:
            fixed_masks[id(p)] = torch.clone(m).detach()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            self.mask(density, scope)
            # allow masks to have gradient
            for m, _ in self.masked_parameters:
                m.requires_grad = True

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()
            for m, p in self.masked_parameters:
                self.signed_scores[id(p)] -= lr * torch.clone(m.grad).detach()
                self.scores[id(p)] = self.signed_scores[id(p)].abs() * fixed_masks[id(p)]
                p.grad.data.zero_()
                m.grad.data.zero_()
                m.requires_grad = False



class MinimumConnectionAssurance(Pruner):
    def __init__(self, masked_parameters):
        super(MinimumConnectionAssurance, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, 
                density=None, scope=None, lr=None, sparsity_distribution=None, mask_file=None, V_out_select='max', set_only_V_in=False):
        print(f'NOTE : in this algorithm, this score function even performs mask construction.')            
        edges = get_layer_wise_num_edge(
            model=model, loss=loss, dataloader=dataloader, masked_parameters=self.masked_parameters, 
            sparsity_distribution=sparsity_distribution, density=density, mask_file=mask_file)
        list_E = {}
        for i, (mask, p) in enumerate(self.masked_parameters):
            list_E[id(p)] = int(edges[i])

        model.set_num_out_channel(list_E, V_out_select)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        x = torch.ones([1] + input_dim).to(device)
        print(f'check model architecture')
        with torch.no_grad():
            model.set_scores(x=x, scores=self.scores, shuffle_V_out=False, set_only_V_in=set_only_V_in)

    def _global_mask(self, density):
        raise NotImplementedError
        
    def _local_mask(self, density):
        r"""Updates masks of model with scores by density level parameter-wise.
        """
        total_edge = 0
        for i, (mask, param) in enumerate(self.masked_parameters):
            print(f'Layer {i} : |E| = {mask.sum()}')
            total_edge += mask.sum()
        print(f'number of remaining edges : {total_edge}')


def get_layer_wise_num_edge(
    model, loss, dataloader, masked_parameters, sparsity_distribution, density, mask_file):
    if mask_file != None:
        print(f'Override mask from {mask_file}')
        mask_model = copy.deepcopy(model)
        mask_dict  = torch.load(mask_file)
        masks      = {}
        for k, v in mask_dict.items():
            print(f'param : {k}')
            masks[k] = v
        model_dict = mask_model.state_dict()
        model_dict.update(masks)
        mask_model.load_state_dict(model_dict)
        mask_model_masked_parameters = list(generator.masked_parameters(mask_model, False, False, False))
        edges = []
        for m, _ in mask_model_masked_parameters:
            edges.append(m.sum())
        edges = torch.tensor(edges)
    else:
        print(f'Sparsity distribution : {sparsity_distribution}')
        if sparsity_distribution == 'snip':
            edges = get_snip(
                model=model, loss=loss, dataloader=dataloader, 
                masked_parameters=masked_parameters, density=density)
        elif sparsity_distribution == 'grasp':
            edges = get_grasp(
                model=model, loss=loss, dataloader=dataloader, 
                masked_parameters=masked_parameters, density=density)
        elif sparsity_distribution == 'synflow':
            edges = get_synflow(
                model=model, loss=loss, dataloader=dataloader, 
                masked_parameters=masked_parameters, density=density)
        elif sparsity_distribution == 'erk':
            edges = get_erk(masked_parameters=masked_parameters, density=density)
        elif sparsity_distribution == 'igq':
            edges = get_igq(masked_parameters=masked_parameters, density=density) 
        elif sparsity_distribution == 'epl':
            edges = get_epl(masked_parameters=masked_parameters, density=density)
        else:
            raise ValueError
    
    edges = torch.floor(edges)

    print(f'Edge setting is complete')
    total_edge = 0
    for i, (_, p) in enumerate(masked_parameters):
        total_edge += p.numel()
        print(f'Layer {i} :')
        print(f'|E| = {edges[i]} (/ {p.numel()})')

    print(f'1. total number of edges :')
    print(f'|E| = {total_edge} * {density} = {total_edge*density} (/ {total_edge})')
    print(f'2. final number of edges :')
    print(f'|E| = {edges.sum()}')
    print('finish')
    return edges


def get_snip(model, loss, dataloader, masked_parameters, density):
    for _, p in masked_parameters:
        device = p.device
        break
    pruner = SNIP(masked_parameters=[])
    pruner.masked_parameters = masked_parameters
    pruner.score(
            model=model, loss=loss, dataloader=dataloader, device=device)
    pruner.mask(density, 'global')
    edges = []
    for m, p in pruner.masked_parameters:
        edges.append(m.sum())
    return torch.tensor(edges)


def get_grasp(model, loss, dataloader, masked_parameters, density):
    for _, p in masked_parameters:
        device = p.device
        break
    pruner = GraSP(masked_parameters=[])
    pruner.masked_parameters = masked_parameters
    pruner.score(
            model=model, loss=loss, dataloader=dataloader, device=device)
    pruner.mask(density, 'global')
    edges = []
    for m, p in pruner.masked_parameters:
        edges.append(m.sum())
    return torch.tensor(edges)

def get_synflow(model, loss, dataloader, masked_parameters, density):
    # epoch 100, pruning_schedule exponential
    for _, p in masked_parameters:
        device = p.device
        break
    pruner = SynFlow(masked_parameters=[])
    pruner.masked_parameters = masked_parameters
    epochs = 100
    for epoch in tqdm(range(epochs)):
        pruner.score(
                model=model, loss=loss, dataloader=dataloader, device=device)
        dense = density**((epoch + 1) / epochs)
        pruner.mask(dense, 'global')
    edges = []
    for m, p in pruner.masked_parameters:
        edges.append(m.sum())
    return torch.tensor(edges)

def get_erk(masked_parameters, density):
    # We have to enforce custom sparsities and then find the correct scaling
    # factor.
    is_eps_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_eps_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
        divisor = 0
        rhs     = 0
        raw_probabilities = {}
        print(f'Loop : ')
        for _, p in masked_parameters:
            n_param = p.numel()
            n_zeros = int(p.numel() * (1 - density))
            if id(p) in dense_layers:
                print(f'{id(p)} is a dense layer')
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                n_ones = n_param - n_zeros
                rhs += n_ones
                assert id(p) not in raw_probabilities 
                print(f'{id(p)} raw_prob : {(sum(p.size()) / p.numel())}')
                raw_probabilities[id(p)] = (sum(p.size()) / p.numel())
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[id(p)] * n_param
        print()
        # All layer is dense
        if divisor == 0:
            is_eps_valid = True
            break
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        eps = rhs / divisor
        print(f'eps : {rhs} / {divisor} = {eps}')
        # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob     = np.max(list(raw_probabilities.values()))
        print(f'raw_prob_list : {list(raw_probabilities.values())}')
        print(f'max_prob : {max_prob}')
        max_prob_one = max_prob * eps
        print(f'max_prob_one : {max_prob_one}')
        if max_prob_one >= 1:
            is_eps_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f'Sparsity of layer {mask_name} had to be set to 0')
                    dense_layers.add(mask_name)
            print()
        else:
            is_eps_valid = True

    sparsities = []
    edges      = []
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for m, p in masked_parameters:
        n_param = p.numel()
        if id(p) in dense_layers:
            sparsities.append(0.)
            edges.append(p.numel())
        else:
            probability_one = eps * raw_probabilities[id(p)]
            sparsities.append(1. - probability_one)
            print(f'{p.numel()} * {probability_one} = {p.numel() * probability_one}')
            edges.append(int(p.numel() * probability_one))

    return torch.tensor(edges)


def get_igq(masked_parameters, density):
    def bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high):
        lengths_low          = [Length / (f_low / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_low = 1 - sum(lengths_low) / sum(Lengths)
        if abs(overall_sparsity_low - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_low, Lengths)]
        
        lengths_high          = [Length / (f_high / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity_high = 1 - sum(lengths_high) / sum(Lengths)
        if abs(overall_sparsity_high - target_sparsity) < tolerance:
            return [1 - length / Length for length, Length in zip(lengths_high, Lengths)]
            
        force            = float(f_low + f_high) / 2
        lengths          = [Length / (force / area + 1) for Length, area in zip(Lengths, areas)]
        overall_sparsity = 1 - sum(lengths) / sum(Lengths)
        f_low            = force if overall_sparsity < target_sparsity else f_low
        f_high           = force if overall_sparsity > target_sparsity else f_high
        return bs_force_igq(areas, Lengths, target_sparsity, tolerance, f_low, f_high)
    
    edges  = []
    counts = []
    for m, p in masked_parameters:
        counts.append(p.numel())
    tolerance = 1./sum(counts)
    areas     = [1./count for count in counts]
    sparsities = bs_force_igq(
        areas=areas, Lengths=counts, target_sparsity=1-density, 
        tolerance=tolerance, f_low=0, f_high=1e20)
    for i, (m, p) in enumerate(masked_parameters):
        edges.append(int(p.numel() * (1 - sparsities[i])))

    return torch.tensor(edges)

def get_epl(masked_parameters, density):
    shapes = []
    for _, p in masked_parameters:
        shapes.append(p.size())
        device = p.device
    num_layer      = len(shapes) # shapes[0] = (out, in, k1, k2)
    layer_sizes    = [(s[0], torch.prod(torch.tensor(s[1:]))) for s in shapes]
    max_edges      = torch.tensor([float(s[0] * s[1]) for s in layer_sizes], device=device)
    num_curr_edge  = torch.floor(max_edges.sum() * density).to(device)
    e              = num_curr_edge
    edges = torch.tensor([0.] * num_layer, device=device)
    list_edge_filled         = torch.tensor([False] * num_layer, device=device)
    num_unfilled_layer       = num_layer
    while True:
        num_unfilled_layer = num_layer - list_edge_filled.sum()
        edges[torch.logical_not(list_edge_filled).to(device)] += e / num_unfilled_layer
        list_edge_filled = torch.logical_or(
            max_edges <= edges,
            torch.isclose(max_edges, edges)
        ).to(device)

        if torch.any(list_edge_filled):
            e = ((edges - max_edges)[list_edge_filled]).sum()
            edges[list_edge_filled] = max_edges[list_edge_filled]
            if torch.all(list_edge_filled) or e == 0:
                print(f'All layers are filled!')
                break
        elif torch.all(edges < 1):
            print(f'Edge alignment results')
            print(edges)
            raise ValueError(f'The edges cannot be placed uniformly at this compression rate! : {np.log10(1 / density)}')
        else:
            print(f'All layers are unfilled!')
            break
    return edges
