from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
from metanas.models import ops, search_cnn
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

""" Genotypes
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""

"""
Based on https://github.com/khanrc/pt.darts
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""


""" Genotype of the Auto-Meta paper https://arxiv.org/pdf/1806.06927.pdf
Note: According to their figues, it should be normal_concat =
reduce_concat = range(0,7), which is currently not supporeted (fully) by
our implemention)
"""

genotype_auto_meta = Genotype(
    normal=[
        [("conv_3x3", 1), ("max_pool_3x3", 1)],
        [("conv_1x5_5x1", 2), ("max_pool_3x3", 2)],
        [("conv_3x3", 2), ("conv_1x5_5x1", 3)],
        [("conv_3x3", 3), ("conv_3x3", 3)],
        [("conv_1x5_5x1", 3), ("conv_3x3", 2)],
    ],
    normal_concat=range(2, 7),
    reduce=[
        [("conv_3x3", 1), ("max_pool_3x3", 1)],
        [("conv_1x5_5x1", 2), ("max_pool_3x3", 2)],
        [("conv_3x3", 2), ("conv_1x5_5x1", 3)],
        [("conv_3x3", 3), ("conv_3x3", 3)],
        [("conv_1x5_5x1", 3), ("conv_3x3", 2)],
    ],
    reduce_concat=range(2, 7),
)


genotype_metanas_v1 = Genotype(
    normal=[
        [("conv_3x3", 0), ("conv_3x3", 1)],
        [("conv_1x5_5x1", 0), ("dil_conv_3x3", 2)],
        [("conv_1x5_5x1", 0), ("max_pool_3x3", 3)],
    ],
    normal_concat=range(2, 5),
    reduce=[
        [("dil_conv_3x3", 0), ("sep_conv_3x3", 1)],
        [("conv_3x3", 0), ("max_pool_3x3", 1)],
        [("dil_conv_3x3", 2), ("conv_1x5_5x1", 3)],
    ],
    reduce_concat=range(2, 5),
)


genotype_metanas_in_v2 = Genotype(
    normal=[
        [("conv_1x5_5x1", 0), ("conv_3x3", 1)],
        [("conv_3x3", 0), ("sep_conv_3x3", 2)],
        [("dil_conv_3x3", 1), ("conv_1x5_5x1", 3)],
    ],
    normal_concat=range(2, 5),
    reduce=[
        [("sep_conv_3x3", 0), ("conv_3x3", 1)],
        [("conv_3x3", 0), ("sep_conv_3x3", 2)],
        [("conv_3x3", 0), ("dil_conv_3x3", 1)],
    ],
    reduce_concat=range(2, 5),
)


genotype_metanas_og_v2 = Genotype(
    normal=[
        [("dil_conv_3x3", 0), ("sep_conv_3x3", 1)],
        [("avg_pool_3x3", 0), ("sep_conv_3x3", 1)],
        [("conv_1x5_5x1", 0), ("conv_3x3", 2)],
    ],
    normal_concat=range(2, 5),
    reduce=[
        [("skip_connect", 0), ("sep_conv_3x3", 1)],
        [("conv_3x3", 0), ("dil_conv_3x3", 1)],
        [("dil_conv_3x3", 0), ("conv_1x5_5x1", 2)],
    ],
    reduce_concat=range(2, 5),
)

PRIMITIVES = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",  # identity
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "none",
]


PRIMITIVES_FEWSHOT = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",  # identity
    "conv_1x5_5x1",
    "conv_3x3",
    "sep_conv_3x3",
    # "sep_conv_5x5",  # remove
    "dil_conv_3x3",
    # "dil_conv_5x5",  # remove
    # "none",  # remove
]

PRIMITIVES_SHARP = [
    "avg_pool_3x3",
    "max_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "flood_conv_3x3",
    "dil_conv_3x3",
    "dil_flood_conv_3x3",
    # "choke_conv_3x3", # Remove choke, due to long training time
    # "dil_choke_conv_3x3",
    # "none",  # These are not described
    # "dil_conv_5x5",
    # "conv_7x1_1x7",
    # "sep_conv_5x5",
    # "flood_conv_5x5",
    # "sep_conv_7x7",
    # "dil_flood_conv_5x5",
]


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            # Identity does not use drop path
            if not isinstance(op, ops.Identity):
                op = nn.Sequential(op, ops.DropPath_())
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    """generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """
    genotype = eval(s)
    return genotype


def parse(alpha, k, primitives=PRIMITIVES_FEWSHOT):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """
    gene = []
    # assert PRIMITIVES_FEWSHOT[-1] == "none"  # assume last PRIMITIVE
    # is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing
    #    top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(
            edges[:, :], 1
        )  # ignore 'none' ##removed none
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = primitives[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_pairwise(alpha, alpha_pairwise, primitives=PRIMITIVES_FEWSHOT):
    """Parse continous alpha to a discrete gene, is deprecated?

    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    alpha_pairwise is ParameterList for pairwise inputs per node:
    ParameterList [
        Parameter(1,)
        Parameter(3,)
        Parameter(6,)
        ...
        Parameter(n_previous_nodes choose 2)
    ]



    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    # get sparse alpha pw
    alpha_pairwise = search_cnn.sparsify_pairwise_alphas(alpha_pairwise)
    gene = []

    for edges, pw_edges in zip(alpha, alpha_pairwise):  # iterate through nodes
        # edges: Tensor(n_edges, n_ops)
        # pw_edges: Tensor(n_input_nodes)

        # find strongest edge for each input
        edge_max, primitive_indices = torch.topk(
            edges[:, :], 1)  # ignore 'none'
        node_gene = []

        top_inputs = []
        pw_idx = 0  # find the best two inputs from pairwise alphas

        # iterate through possible inputs and check which to use
        # (correct one = only combination without zero alpha)
        for input_1 in range(len(edges)):
            for input_2 in range(input_1 + 1, len(edges)):
                if pw_edges[pw_idx] > 0:
                    top_inputs = torch.tensor([input_1, input_2])
                pw_idx = pw_idx + 1

        for edge_idx in top_inputs:
            prim_idx = primitive_indices[edge_idx]
            prim = primitives[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def convert_tensor_alphas(alpha_concat, nodes=3):
    alphas = []
    for a_i in get_edge_indices(nodes):
        alphas.append(
            torch.Tensor(alpha_concat[a_i[0]:a_i[1]]))

    # print(alpha_concat, alphas)
    return alphas


def get_edge_indices(nodes=3):
    # Amount of nodes for each edge
    j = [i for i in range(2, nodes+2)]

    prev = 0
    indices = []
    for i in j:
        if prev != 0:
            indices.append((sum(j[:j.index(prev)+1]),
                            sum(j[:j.index(i)+1])))
        else:
            indices.append((0, sum(j[:j.index(i)+1])))
        prev = i
    return indices


def find_indice(row_idx, nodes=3):
    edges = get_edge_indices(nodes)

    for i, (lower, upper) in enumerate(edges):
        if row_idx >= lower and row_idx <= upper:
            # return index
            idx_alpha = 0 if row_idx-lower == 0 else row_idx-lower-1
            return i, idx_alpha


def limit_skip_connections(alphas, primitives, k=2,
                           num_of_skip_connections=2):
    sk_idx = primitives.index("skip_connect")

    # Number of skip-connections enabled in switches
    gene = parse(alphas, k)

    alphas = [a.detach().cpu().numpy() for a in alphas]
    alpha_concat = np.concatenate(alphas, axis=0)

    num_sk_enabled = sum([1 for edge in gene
                          for op in edge if op[0] == "skip_connect"])

    sk_a = alpha_concat[:, sk_idx-1:sk_idx]
    if num_sk_enabled < num_of_skip_connections:
        alphas = convert_tensor_alphas(alpha_concat)
        gene = parse(alphas, k)
        return gene
    else:
        while num_sk_enabled > num_of_skip_connections:
            # Pick skip-connection index with lowest alpha
            # value
            row_idx = np.argmin(sk_a)
            sk_a[row_idx] = float("inf")

        #     # set alphas to -inf to make sure, prevent it from
        #     # being picked.
            alpha_concat[row_idx][sk_idx] = float("-inf")
            alphas = convert_tensor_alphas(alpha_concat)

            gene = parse(alphas, k)
            num_sk_enabled = sum([1 for edge in gene
                                  for op in edge if op[0] == "skip_connect"])

            if num_sk_enabled <= num_of_skip_connections:
                # return the new switches
                return gene


def limit_skip_connections_pw(alphas, alphas_pairwise, primitives,
                              num_of_skip_connections=2):
    sk_idx = primitives.index("skip_connect")

    # Number of skip-connections enabled in switches
    gene = parse_pairwise(alphas, alphas_pairwise)

    alphas = [a.detach().cpu().numpy() for a in alphas]
    alpha_concat = np.concatenate(alphas, axis=0)

    num_sk_enabled = sum([1 for edge in gene
                          for op in edge if op[0] == "skip_connect"])

    sk_a = alpha_concat[:, sk_idx-1:sk_idx]
    if num_sk_enabled < num_of_skip_connections:
        alphas = convert_tensor_alphas(alpha_concat)
        gene = parse_pairwise(alphas, alphas_pairwise)
        return gene
    else:
        while num_sk_enabled > num_of_skip_connections:
            # Pick skip-connection index with lowest alpha
            # value
            row_idx = np.argmin(sk_a)
            sk_a[row_idx] = float("inf")

        #     # set alphas to -inf to make sure, prevent it from
        #     # being picked.
            alpha_concat[row_idx][sk_idx] = float("-inf")
            alphas = convert_tensor_alphas(alpha_concat)

            gene = parse_pairwise(alphas, alphas_pairwise)
            num_sk_enabled = sum([1 for edge in gene
                                  for op in edge if op[0] == "skip_connect"])

            if num_sk_enabled <= num_of_skip_connections:
                # return the new switches
                return gene


def limit_skip_connections_alphas(alpha, primitives, k=2, nodes=3,
                                  num_of_skip_connections=2):
    """Mutate the alpha in-place and return the corresponding gene"""
    gene = parse(alpha, k=k)
    num_sk_enabled = sum([1 for edge in gene
                          for op in edge if op[0] == "skip_connect"])
    epsilon = 0.0001

    if num_sk_enabled < num_of_skip_connections:
        return gene
    else:
        sk_idx = primitives.index("skip_connect")
        alphas_concat = np.concatenate(
            [a.detach().cpu().numpy() for a in alpha],
            axis=0)
        sk_alphas = alphas_concat[:, sk_idx-1:sk_idx]

        for i in range(len(sk_alphas)):
            # Pick skip-connection index with lowest alpha value
            row_idx = np.argmin(sk_alphas)
            # Set to inf so we don't pick this again
            sk_alphas[row_idx] = float("inf")

            edge_idx, row_idx = find_indice(row_idx, nodes)

            # Set the actual alpha value to min - epsilon
            alpha[edge_idx][row_idx][sk_idx] = torch.min(
                alpha[edge_idx][row_idx]) - epsilon
            gene = parse(alpha, k=k)

            num_sk_enabled = sum([1 for edge in gene
                                  for op in edge if op[0] == "skip_connect"])
            if num_sk_enabled < num_of_skip_connections:
                return gene
