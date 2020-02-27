from itertools import chain

import torch
import dgl


def h5group_to_dict(h5group):
    group_dict = {k: v[()] for k, v in chain(h5group.items(), h5group.attrs.items())}
    return group_dict


def sub_dict(full_dict, *keys, to_tensor):
    return {k: torch.tensor(full_dict[k]).float() if to_tensor else full_dict[k] for k in keys if k in full_dict}


def build_graph_from_dict(g_dict, to_tensor=True):
    g = dgl.DGLGraph()
    g.add_nodes(g_dict['num_nodes'], data=sub_dict(g_dict, 'x', 'y', to_tensor=to_tensor))
    g.add_edges(*g_dict['edge_index'], data=sub_dict(g_dict, 'edge_attr', to_tensor=to_tensor))
    return g