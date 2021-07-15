#%%
import os
import sys
import requests

import dgl
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch

import warnings
warnings.filterwarnings("ignore")
os.path.abspath('.')

DTYPE = np.float32
IDTYPE = np.int32

data_dir = '/home/flower/projects/def-laurence/flower'
# data_dir = '../data'
residue2idx = torch.load('../data/res2idx_dict_core.pt')
# residue2count = torch.load('../data/res2count_dict.pt')

#%%
# def blockPrinting(func):
#     def func_wrapper(*args, **kwargs):
#         # block all printing to the console
#         sys.stdout = open(os.devnull, 'w')
#         # call the method in question
#         value = func(*args, **kwargs)
#         # enable all printing to the console
#         sys.stdout = sys.__stdout__
#         # pass the return value of the method back
#         return value

#     return func_wrapper

class ProtProcess:
    @staticmethod
    def download_pdb(pdb_id, outfile):
        # check if the pdb file has been downloaded in the outfile
        if os.path.exists(outfile):
            # print(f"The file has been downloaded...")
            return 1

        page = 'http://files.rcsb.org/view/{}.pdb'.format(pdb_id)
        req = requests.get(page)
        if req.status_code == 200:
            response = req.text
            if outfile:
                with open(outfile, 'w') as f:
                    f.write(response)
            return 1
        else:
            return 0

    @staticmethod
    def get_residue_feature(chain):
        # init
        res_list = []
        coo_list = []

        # record sequential info
        for res in chain:
            if res.get_resname() == 'HOH':
                continue

            # record residue sequence
            res_name = res.get_resname()
            if residue2idx.get(res_name) is None:
                continue                            # with res2idx_dict_core.pt
                # residue2idx[res_name] = len(residue2idx)
                # residue2count[res_name] = 0                 # TODO: remove later

            res_list.append(residue2idx[res_name])
            # residue2count[res_name] += 1                    # TODO: remove later


            # compute coo of residue by averaging its atoms' coos
            coo = np.concatenate([i.get_coord() for i in res]).reshape((-1, 3)).mean(axis=0)
            coo_list.append(coo)

        return np.array(res_list).astype(IDTYPE), np.concatenate(coo_list).reshape((-1, 3)).astype(DTYPE)

    @staticmethod
    def get_edge_set(dis_cut, l, atom_list, level='R'):
        ns = NeighborSearch(atom_list)
        ns_list = ns.search_all(dis_cut, level=level)

        edge = set()
        for ai, aj in ns_list:
            i, j = ai.get_id()[1]-1, aj.get_id()[1]-1
            if i < j-1 and j < l and i > 0:
                edge.add((i, j))

        return edge

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q

#%%
class ProtFunctDataset(Dataset):
    atom_feature_size = len(residue2idx)

    def __init__(self, file_path, mode: str='train', if_transform: bool=True, dis_cut: list=[3.0, 3.5]):
        """Create a dataset object

        Args:
            file_path (str): path to data
            mode (str, optional): {train/test/val}. Defaults to 'train'.
            if_transform (bool, optional): if applying data augmentation function. Defaults to True.
        """
        self.file_path = file_path
        self.mode = mode

        self.dis_cut = dis_cut
        self.num_bonds = len(dis_cut) + 1
        self.bond2idx = {'covalent':0, 'neighbor<{dis_cut[0]}':1, 'neighbor<{dis_cut[1]}': 2}

        self.transform = RandomRotation() if if_transform else None
        
        self.__load_data()
        self.len = len(self.targets)

    def __load_data(self):
        """Load preprocessed dataset and parser for input protein structure."""
        data = torch.load(self.file_path)[self.mode]
        # s, e = 100, 500
        self.inputs = data['input_list']               # TODO: remove
        self.targets = data['target_list']

        self.parser = PDBParser()

    def __len__(self):
        return self.len

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)),data] = 1
        return one_hot

    def __getitem__(self, idx):
        pdb, y = self.inputs[idx], self.targets[idx]

        # parse protein structure
        p, c = pdb.split('.')           # pdb ID and chain ID
        ProtProcess.download_pdb(p, f'{data_dir}/pdb/{p}.pdb')
        structure = self.parser.get_structure('a', f'{data_dir}/pdb/{p}.pdb')
        chain = structure[0][c]

        # generate node features
        try:
            res, x = ProtProcess.get_residue_feature(chain)
        except:
            print('error pdb: ', pdb)
        num_residues = res.shape[0]
        res = self.to_one_hot(res, len(residue2idx))[...,None]

        # augmentation on the coordinates(
        if self.transform:
            x = self.transform(x).astype(DTYPE)

        # generate edge features
        src, dst, w = self.__connect_partially([i for i in chain.get_atoms()], num_residues)
        w = self.to_one_hot(w, self.num_bonds).astype(DTYPE)

        # create protein representation graph
        G = dgl.DGLGraph((src, dst))
        # add node feature
        x = torch.Tensor(x)
        G.ndata['x'] = x
        G.ndata['f'] = torch.Tensor(res)
        # add edge feature
        G.edata['d'] = x[dst] - x[src]
        G.edata['w'] = torch.Tensor(w)
    
        return G, y, pdb
        # return torch.Tensor([0, 0])

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to QM9 representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        # x = self.unit_conversion[self.task] * x
        return x


    def __connect_partially(self, atom_list, num_residues):
        # initialize edges satisfy the different distance cutoffs
        adjacency = {}
        for c in range(1, self.num_bonds):
            for i, j in ProtProcess.get_edge_set(self.dis_cut[-c], num_residues, atom_list):
                adjacency[(i, j)] = self.num_bonds - c
                adjacency[(j, i)] = self.num_bonds - c
        
        # add covalent bonds
        for i in range(1, num_residues):
            adjacency[(i-1, i)] = 0
            adjacency[(i, i-1)] = 0

        # convert to numpy arrays
        src, dst, w = [], [], []
        for edge, bond in adjacency.items():
            src.append(edge[0])
            dst.append(edge[1])
            w.append(bond)

        return np.array(src).astype(IDTYPE), np.array(dst).astype(IDTYPE), np.array(w)
        
def collate(samples):
    graphs, y, pdb = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y), pdb

def to_np(x):
    return x.cpu().detach().numpy()

#%%
# test
if __name__ == '__main__':
    try:
        dataset = ProtFunctDataset('../data/ProtFunct.pt', mode='train')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

        for i, data in enumerate(dataloader):
            print(i)
    finally:
        torch.save(residue2idx, 'res2idx_dict.pt')
        # torch.save(residue2count, 'res2count_dict.pt')

