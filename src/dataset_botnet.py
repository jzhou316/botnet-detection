import os
import os.path as osp
import pickle

import h5py
import deepdish as dd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .url_utils import makedirs, decide_download, download_url, extract_tar
from .data_utils import (h5group_to_dict, build_graph_from_dict_pyg,
                         build_graph_from_dict_dgl, build_graph_from_dict_nx)


def files_exist(files):
    return all([osp.exists(f) for f in files])


# TODO: add degree calculation in preprocessing
class BotnetDataset(Dataset):
    """
    Botnet detection graph dataset, containing different botnet topologies and train/val/test splits.
    The graphs are stored in HDF5 files.
    
    Args:
        path (str): path of the HDF5 file containing a series of graph data
        num_graphs (int, optional): number of graphs in the dataset
        in_memory (bool, optional): whether to read all the graphs into memory. Default: False
    """

    url = 'https://sandbox.zenodo.org/record/503403/files/botnet_chord.tar.gz'

    def __init__(self, name='chord', root='data/botnet', split='train', split_idx=None, add_nfeat_ones=True,
                 in_memory=True, graph_format='pyg'):
        super().__init__()
        assert name in ['chord', 'debru', 'kadem', 'leet']
        assert split in ['train', 'val', 'test']
        assert graph_format in ['pyg', 'dgl', 'nx', 'dict']

        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))

        self.name = name
        self.root = root
        self.split = split
        self.split_idx = split_idx
        self.add_nfeat_ones = add_nfeat_ones

        self.download()

        self.process()

        self.in_memory = in_memory
        self._graph_format = graph_format
        if split == 'train':
            self.path = self.processed_paths[0]
        elif split == 'val':
            self.path = self.processed_paths[1]
        elif split == 'test':
            self.path = self.processed_paths[2]

        if in_memory:
            self.data = dd.io.load(self.path)  # dictionary
            self.data_type = 'dict'
            self.num_graphs = self.data['num_graphs']
        else:
            self.data = h5py.File(self.path, 'r')
            self.data_type = 'file'
            self.num_graphs = self.data.attrs['num_graphs']

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return ['botnet_' + self.name + '.tar.gz', self.name + '_raw.hdf5', self.name + '_split_idx.pkl']

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return [self.name + '_' + s + '.hdf5' for s in ('train', 'val', 'test')]

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return [osp.join(self.processed_dir, f) for f in self.processed_file_names]

    @property
    def graph_format(self):
        return self._graph_format

    def download(self):
        # breakpoint()
        if osp.exists(self.raw_paths[0]) or files_exist(self.raw_paths[1:3]):
            return

        if files_exist(self.processed_paths):
            return

        makedirs(self.raw_dir)

        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            extract_tar(path, self.raw_dir)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        # breakpoint()
        if files_exist(self.processed_paths):
            return

        if not files_exist(self.raw_paths[1:3]):
            assert osp.exists(self.raw_paths[0])
            path = extract_tar(self.raw_paths[0], self.raw_dir)
        #             os.unlink(self.raw_paths[0])

        print('Processing...')
        makedirs(self.processed_dir)

        if self.split_idx is None:
            # default data split
            split_idx = pickle.load(open(self.raw_paths[2], 'rb'))

        with h5py.File(self.raw_paths[1], 'r') as f:
            for path, split in zip(self.processed_paths, ('train', 'val', 'test')):
                print(f'writing {split} set ' + '-' * 10)
                ori_graph_ids = split_idx[split]
                with h5py.File(path, 'w') as g:
                    num_nodes_sum = 0
                    num_edges_sum = 0
                    num_evils_sum = 0
                    if 'num_evil_edges_avg' in f.attrs:
                        num_evil_edges_sum = 0
                        num_evil_edges_flag = True
                    else:
                        num_evil_edges_sum = None
                        num_evil_edges_flag = False

                    for n, i in tqdm(enumerate(ori_graph_ids)):
                        f.copy(str(i), g, name=str(n))
                        if self.add_nfeat_ones:
                            g[str(n)].create_dataset('x',
                                                     shape=(g[str(n)].attrs['num_nodes'], 1),
                                                     dtype='f4',
                                                     data=np.ones((g[str(n)].attrs['num_nodes'], 1)))

                        num_nodes_sum += f[str(i)].attrs['num_nodes']
                        num_edges_sum += f[str(i)].attrs['num_edges']
                        num_evils_sum += f[str(i)].attrs['num_evils']
                        if num_evil_edges_flag:
                            num_evil_edges_sum += f[str(i)].attrs['num_evil_edges']

                    g.attrs['num_graphs'] = n + 1
                    g.attrs['num_nodes_avg'] = num_nodes_sum / (n + 1)
                    g.attrs['num_edges_avg'] = num_edges_sum / (n + 1)
                    g.attrs['num_evils_avg'] = num_evils_sum / (n + 1)
                    if num_evil_edges_flag:
                        g.attrs['num_evil_edges_avg'] = num_evil_edges_sum / (n + 1)
                    g.attrs['is_directed'] = f.attrs['is_directed']
                    g.attrs['contains_self_loops'] = f.attrs['contains_self_loops']
                    g.attrs['ori_graph_ids'] = ori_graph_ids

                print('{} split --- number of graphs: {}, data saved at {}.'.format(split, n + 1, path))

        print('Done!')

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, index):
        if self.data_type == 'dict':
            graph_dict = self.data[str(index)]
        elif self.data_type == 'file':
            graph_dict = h5group_to_dict(self.data[str(index)])
        else:
            raise ValueError

        if self.graph_format == 'pyg':
            return build_graph_from_dict_pyg(graph_dict)
        elif self.graph_format == 'dgl':
            return build_graph_from_dict_dgl(graph_dict)
        elif self.graph_format == 'nx':
            return build_graph_from_dict_nx(graph_dict)
        elif self.graph_format == 'dict':
            return graph_dict

    def __repr__(self):
        return f'{self.__class__.__name__}(botnet topology: {self.name} | data split: {self.split} | ' \
               f'number of graphs: {len(self)} | graph format: {self.graph_format})'


if __name__ == '__main__':
    dataset = BotnetDataset(split='train')
    print(len(dataset))
    print(dataset[0])
    breakpoint()
