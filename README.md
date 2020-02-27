# botnet-detection
Topological botnet detection datasets and graph neural network applications

## To Load Botnet Data

Include the directory `src` in your Python search path (can be done via `sys.path.insert(0, 'path_to_src')`), and import `BotnetDataset` class by `from src.dataset_botnet import BotnetDataset`.

Load the botnet dataset, which can be compatible with most of the graph learning libraries by specifying the `graph_format` argument:
```
botnet_dataset_train = BotnetDataset(name='chord', split='train', graph_format='pyg')
botnet_dataset_val = BotnetDataset(name='chord', split='val', graph_format='pyg')
botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')
```

The choices for `name` are (indicating different botnet topologies):
- 'chord'
- 'debru'
- 'kadem'
- 'leet'

The choices for `graph_format` are (for different graph data format according to different graph libraries):
- 'pyg' for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- 'dgl' for [DGL](https://github.com/dmlc/dgl) 
- 'nx' for [NetworkX](https://github.com/networkx/networkx)
- 'dict' for plain python dictionary
