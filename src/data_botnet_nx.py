import networkx as nx


def nxgraph(g_dict):
    g = nx.Graph()
    g.add_nodes_from(range(g_dict['num_nodes']))
    g.add_edges_from(zip(g_dict['edge_index'][0], g_dict['edge_index'][1]))
    return g
