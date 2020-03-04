# botnet-detection

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Topological botnet detection datasets and automatic detection with graph neural networks.

## To Load the Botnet Data

Include the directory `src` in your Python search path (can be done via `sys.path.insert(0, 'path_to_src')`), and import `BotnetDataset` class by `from src.data.dataset_botnet import BotnetDataset`.

Load the botnet dataset, which can be compatible with most of the graph learning libraries by specifying the `graph_format` argument:
```
from src.data.dataset_botnet import BotnetDataset

botnet_dataset_train = BotnetDataset(name='chord', split='train', graph_format='pyg')
botnet_dataset_val = BotnetDataset(name='chord', split='val', graph_format='pyg')
botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')
```

The choices for `name` are (indicating different botnet topologies):
- 'chord' (synthetic, 10k botnet nodes)
- 'debru' (synthetic, 10k botnet nodes)
- 'kadem' (synthetic, 10k botnet nodes)
- 'leet' (synthetic, 10k botnet nodes)
- 'c2' (real, ~3k botnet nodes)
- 'p2p' (real, ~3k botnet nodes)

The choices for `graph_format` are (for different graph data format according to different graph libraries):
- 'pyg' for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- 'dgl' for [DGL](https://github.com/dmlc/dgl) 
- 'nx' for [NetworkX](https://github.com/networkx/networkx)
- 'dict' for plain python dictionary

Based on different choices of the above argument, when indexing the botnet dataset object, it will return a corresponding graph data object defined by the specified graph library.

Construct the data loader with automatic batching (agnostic to the specific graph learning library):
```
from src.data.dataloader import GraphDataLoader

train_loader = GraphDataLoader(botnet_dataset_train, batch_size=2, shuffle=False, num_workers=0)
val_loader = GraphDataLoader(botnet_dataset_val, batch_size=1, shuffle=False, num_workers=0)
test_loader = GraphDataLoader(botnet_dataset_test, batch_size=1, shuffle=False, num_workers=0)
```

## To Evaluate a Model Predictor

Include the directory `src` in your Python search path (can be done via `sys.path.insert(0, 'path_to_src')`), and load the dataset class and the evaluation function as below:

```
from src.data.dataset_botnet import BotnetDataset
from src.eval.evaluation import eval_predictor
```

Then define a simple wrapper of your model as a predictor function which takes in a graph from the dataset and output the prediction probabilities for the positive class (as well as the loss from the forward pass, optionally). Some examples are [here](src/eval/evaluation.py#L99).

We compare evaluations on the test set, as below:

```
from src.eval.evaluation import PygModelPredictor

botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')
predictor = PygModelPredictor(model)    # 'model' is some graph learning model
result_dict_avg, loss_avg = eval_predictor(botnet_dataset_test, predictor)

print(f'Testing --- loss: {loss_avg:.5f}')
print(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict_avg.items()]))

test_f1 = result_dict_avg['f1']
```

And we mainly compare the average F1 score to compare across models.

## To Train a Graph Neural Network for Topological Botnet Detection

We implemented a set of graph convolutional neural network models [here](./src/models_pyg) with PyTorch Geometric, and provide an example training script [here](./train_botnet.py).

One can use our main [model API](./src/models_pyg/gcn_model.py#L9) to construct various basic GNN models, by specifing different number of layers, how in each layer node representations are updated (e.g. with direct message passing, MLP, or with graph attention), different choices of non-linear activation functions, whether to use residual connections and how many hops to connect, whether to add a final projection layer or not, etc. For a complete list of model configuration arguments, check our [example training script](./train_botnet.py#L71).

We run graph neural network models on each of the topologies, and our results are as below:

<!--| Topology | Chord | de Bruijn | Kademlia | LEET-Chord | C2 | P2P |-->
<!--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-->
<!--| Test F1 | | | | | | |-->
<!--| Average Over Topologies <td colspan=6> 0 </td>|-->

<table align="center">
  <tr>
    <td> Topology </td>
    <td> Chord </td>
    <td> de Bruijn </td>
    <td> Kademlia </td>
    <td> LEET-Chord </td>
    <td> C2 </td>
    <td> P2P </td>
  </tr>
    
  <tr>
    <td> Test F1 </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
  </tr>
  <tr>
    <td style="text-align:center"> Average </td>
    <td colspan="6"> 0 </td>
  </tr>
</table>

## Citing

```
@article{zhou2020auto,
  title={Automating Botnet Detection with Graph Neural Networks},
  author={Jiawei Zhou*, Zhiying Xu*, Alexander M. Rush, and Minlan Yu},
  journal={AutoML for Networking and Systems Workshop of MLSys 2020 Conference},
  year={2020}
}
```
