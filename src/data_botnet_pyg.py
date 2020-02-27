import numpy as np
import torch
from torch_geometric.data import Data


class GraphData(Data):
    """
    A single static graph data object.
    Args:
        graph_dict (dict): a dictionary containing a single static graph, with attributes containing
            'x', 'edge_index', 'edge_attr', 'y', etc. (the data could be numpy.ndarray)
    """
    def __init__(self, graph_dict, to_tensor=True):
        if 'num_edges' in graph_dict:
            del graph_dict['num_edges']  # conflict with Data's property method 'num_edges'
        if 'num_evils' in graph_dict:
            del graph_dict['num_evils']  # otherwise error from Batch.from_data_list (figure out later)
        super().__init__(**graph_dict)
        if to_tensor:
            self.to_tensor()

    def to_tensor(self):
        for k, v in self():    # Data.__call__(*keys) which iterates all attributes in the graph data
            if isinstance(v, np.ndarray):
                self[k] = torch.tensor(v)