import torch
import torch.nn as nn

import numpy as np


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mean_no_none(l):
    l_no_none = [el for el in l if el is not None]
    return sum(l_no_none) / len(l_no_none)


def aggregate_dicts(dicts, operation=mean_no_none):
    all_keys = set().union(*[el.keys() for el in dicts])
    return {k: operation([dic.get(k, None) for dic in dicts]) for k in all_keys}


def aggregate_info_dicts(dicts):
    agg_dict = aggregate_dicts(dicts)

    return {k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in agg_dict.items()}
