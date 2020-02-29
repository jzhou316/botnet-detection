# botnet-detection
Topological botnet detection datasets and graph neural network applications

## To Load the Botnet Data

Include the directory `src` in your Python search path (can be done via `sys.path.insert(0, 'path_to_src')`), and import `BotnetDataset` class by `from src.dataset_botnet import BotnetDataset`.

Load the botnet dataset, which can be compatible with most of the graph learning libraries by specifying the `graph_format` argument:
```
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

## To Evaluate a Model Predictor

Include the directory `src` in your Python search path (can be done via `sys.path.insert(0, 'path_to_src')`), and load the dataset class and the evaluation function as below:

```
from src.dataset_botnet import BotnetDataset
from src.evaluation import eval_predictor
```

Then define a simple wrapper of your model as a predictor function which takes in a graph from the dataset and output the prediction probabilities for the positive class (as well as the loss from the forward pass, optionally). Some examples are [here](./src/evaluation.py#L99).

We compare evaluations on the test set, as below:

```
from src.evaluation import PygModelPredictor

botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')
predictor = PygModelPredictor(model)    # 'model' is some graph learning model
result_dict_avg, loss_avg = eval_predictor(botnet_dataset_test, predictor)

print(f'Testing --- loss: {loss_avg:.5f}')
print(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict_avg.items()]))

test_auroc = result_dict_avg['auroc']
```

And we mainly compare the AUROC (area under the ROC curve) metric to compare across models.

## Graph Neural Network Results

We run graph neural network models on each of the topologies, and our results are as below:

<!--| Topology | Chord | de Bruijn | Kademlia | LEET-Chord | C2 | P2P |-->
<!--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-->
<!--| Test AUROC | | | | | | |-->
<!--| Average Over Topologies <td colspan=6> 0 </td>|-->

<table>
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
    <td> Test AUROC </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
  </tr>
  
  <tr>
    <td> Average </td>
    <td colspan="6"> 0 </td>
  </tr>
</table>
