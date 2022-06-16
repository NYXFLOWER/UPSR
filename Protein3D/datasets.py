#%%
import os
from shutil import Error
import requests

import dgl
import torch
import numpy as np
from collections import OrderedDict

from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from dgl.data.utils import save_graphs, load_graphs

# from equivariant_attention.utils_profiling import * # load before other local module

import warnings
warnings.filterwarnings("ignore")

DTYPE = np.float32
IDTYPE = np.int32

# data_dir = "/global/home/hpc4590/share/protein"
# data_dir = '../data'


# dir_path = os.path.dirname(os.path.realpath(__file__))
# residue2idx = torch.load(f'../data/res2idx_dict_core.pt')
# residue2count = torch.load('../data/res2count_dict.pt')
residue2idx = OrderedDict([('GLY', 0), ('PRO', 1), ('LEU', 2), ('SER', 3), ('MET', 4), ('LYS', 5), ('ASP', 6), ('ILE', 7), ('ASN', 8), ('TYR', 9), ('VAL', 10), ('GLU', 11), ('ARG', 12), ('THR', 13), ('GLN', 14), ('PHE', 15), ('CYS', 16), ('ALA', 17), ('HIS', 18), ('TRP', 19)])


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
    def download_pdb(pdb_id, outfile, replace=False):
        # check if the pdb file has been downloaded in the outfile
        if not replace:
            if os.path.exists(outfile):
                # print(f"The file has been downloaded...")
                # check if the file is empty
                if os.stat(outfile).st_size:
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
    def get_residue_feature(residues):
        # init
        res_list = []
        coo_list = []
        atom_list = []

        current_chain = residues[0].get_full_id()[2]
        chain_end_idx = []

        # record sequential info
        for res in residues:
            if res.get_resname() == 'HOH':
                continue

            # record residue sequence
            res_name = res.get_resname()
            if residue2idx.get(res_name) is None:
                continue                            # with res2idx_dict_core.pt
                # residue2idx[res_name] = len(residue2idx)
                # residue2count[res_name] = 0                 # TODO: remove later

            tmp_chain = res.get_full_id()[2]
            if tmp_chain != current_chain:
                chain_end_idx.append(len(res_list)-1)
                current_chain = tmp_chain

            res_list.append(residue2idx[res_name])
            # residue2count[res_name] += 1                    # TODO: remove later
            atom_list.extend([i for i in res.get_atoms()])

            # compute coo of residue by averaging its atoms' coos
            coo = np.concatenate([i.get_coord() for i in res]).reshape((-1, 3)).mean(axis=0)
            coo_list.append(coo)

        return np.array(res_list).astype(IDTYPE), np.concatenate(coo_list).reshape((-1, 3)).astype(DTYPE), np.array(atom_list), chain_end_idx

    @staticmethod
    def get_edge_set(dis_cut, atom_list, level='R'):
        ns = NeighborSearch(atom_list)
        ns_list = ns.search_all(dis_cut, level=level)

        edge = set()
        for ai, aj in ns_list:
            i, j = ai.get_id()[1]-1, aj.get_id()[1]-1
            if i < j-1 and i > 0:
                edge.add((i, j))

        return edge

class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


class ProtFunctDatasetMultiClass(Dataset):
    # IE baseline dataset
    
    atom_feature_size = len(residue2idx)

    def __init__(self, inputs: np.array, outputs: np.array, dis_cut: list, if_transform: bool=False, data_dir = "/home/flower/projects/def-laurence/flower"):
        """Create a SIFTS dataset object"""
        self.inputs = inputs
        self.dis_cut = dis_cut
        self.data_dir = data_dir
        self.transform = RandomRotation() if if_transform else None

        # recast EC numbers to index
        self.get_class_recaster = {}
        for i, c in enumerate(np.unique(outputs)):
            self.get_class_recaster[c] = i
        self.outputs = np.array([self.get_class_recaster[i] for i in outputs])

        # set edge types
        self.dis_cut = dis_cut
        self.num_bonds = len(dis_cut) + 1
        self.bond2idx = {'covalent':0}
        for i, c in enumerate(dis_cut):
            self.bond2idx[f'neighbor<{dis_cut[i]}'] = i+1 
        
        # set pdb parser
        self.parser = PDBParser()

        # set dataset info
        self.len = self.inputs.shape[0]
        self.num_class = len(self.get_class_recaster)

        # set cache data directory for storing pre-constructed protein graph
        self.cache_dir = f"{data_dir}/graph/{'-'.join([str(i) for i in dis_cut])}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        print(f'Data summary -> {self.num_class} protein classes, and {self.len} protein samples\n')


    def __init__(self, file_path, mode: str='train', if_transform: bool=True, dis_cut: list=[3.0, 3.5], use_classes: list=None, data_dir = "/home/flower/projects/def-laurence/flower"):
        """Create an IE dataset object

        Args:
            file_path (str): path to data
            mode (str, optional): {train/test/valid}. Defaults to 'train'.
            if_transform (bool, optional): if applying data augmentation function. Defaults to True.
        """
        self.file_path = file_path
        self.mode = mode

        self.dis_cut = dis_cut
        self.num_bonds = len(dis_cut) + 1
        self.bond2idx = {'covalent':0}
        for i, c in enumerate(dis_cut):
            self.bond2idx[f'neighbor<{dis_cut[i]}'] = i+1

        self.transform = RandomRotation() if if_transform else None

        self.use_classes = use_classes
        self.get_class_recaster = {}
        if use_classes:
            for i, c in enumerate(use_classes):
                self.get_class_recaster[c] = i
        
        self.__load_data_ie()

        self.parser = PDBParser()

        self.len = self.inputs.shape[0]

        self.data_dir = data_dir

        self.cache_dir = f"{data_dir}/graph/{'-'.join([str(i) for i in dis_cut])}/"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        print(f'Data summary -> {len(use_classes) if use_classes else 384} protein classes, and {self.inputs.shape[0]} protein samples')

    def __load_data_ie(self):
        """Load preprocessed dataset and parser for input protein structure."""
        data = torch.load(self.file_path)[self.mode]
        self.inputs = np.array(data['input_list']) #[2393*8 :] 
        self.targets = np.array(data['target_list']) #[2393*8 :] 

        # if self.mode == 'train':
        #     self.inputs = self.inputs[3500*8:]
        #     self.targets = self.targets[3500*8:]

        if self.use_classes:
            self.__use_selected_classes()


    def __use_selected_classes(self):
        mask = np.vstack([self.targets == i for i in self.use_classes]).any(axis=0)
        self.inputs = self.inputs[mask]
        self.targets = self.targets[mask]

    def __len__(self):
        return self.len

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)),data] = 1
        return one_hot

    def __prepare_item__(self, pdb, fp):

        # parse protein structure
        p, c = pdb.split('.')           # pdb ID and chain ID
        ProtProcess.download_pdb(p, f'{self.data_dir}/pdb/{p}.pdb')
        flag = True
        while(flag):
            # generate node features
            try:
                structure = self.parser.get_structure('a', f'{self.data_dir}/pdb/{p}.pdb')
                residues = list(structure[0].get_residues())
                res, x, atoms, chain_end_idx = ProtProcess.get_residue_feature(residues)
                num_residues = res.shape[0]

                # assert num_residues == 0
                flag = False
            except Error as err:
                print('error: ', err)
                print('error pdb: ', pdb)
                ProtProcess.download_pdb(p, f'{self.data_dir}/pdb/{p}.pdb', replace=True)

        res = self.to_one_hot(res, len(residue2idx))[...,None]

        # augmentation on the coordinates(
        if self.transform:
            x = self.transform(x).astype(DTYPE)

        # generate edge features
        src, dst, w = self.__connect_partially(atoms, num_residues, chain_end_idx)
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

        save_graphs(fp, G)
    
        return G

    # @profile
    def __getitem__(self, idx):
        pdb, y = self.inputs[idx], self.targets[idx]
        # print(self.data_dir)
        fp = f"{self.cache_dir}{pdb}.bin"
        # os.path.join(self.data_dir, 'graph', '-'.join(self.dis_cut), f'{pdb}.bin')
        if os.path.exists(fp):
            G, _ = load_graphs(fp)
            G = G[0]
        else:
            G = self.__prepare_item__(pdb, fp)

        G.readonly()

        if self.use_classes:
            y = self.get_class_recaster.get(y)
    
        return G, y, pdb

    def norm2units(self, x, denormalize=True, center=True):
        # Convert from normalized to QM9 representation
        if denormalize:
            x = x * self.std
            # Add the mean: not necessary for error computations
            if not center:
                x += self.mean
        # x = self.unit_conversion[self.task] * x
        return x


    def __connect_partially(self, atom_list: list, num_residues: int, chain_end_idx=[]):
        # initialize edges satisfy the different distance cutoffs
        adjacency = {}
        for c in range(1, self.num_bonds):
            for i, j in ProtProcess.get_edge_set(self.dis_cut[-c], atom_list):
                adjacency[(i, j)] = self.num_bonds - c
                adjacency[(j, i)] = self.num_bonds - c
        
        # add covalent bonds
        for i in range(num_residues-1):
            if i in chain_end_idx:
                continue
            adjacency[(i, i+1)] = 0
            adjacency[(i+1, i)] = 0

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

def collate_ns(samples):
    graphs, y, pdb, graphs_ns, y_ns, pdb_ns = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs+graphs_ns)
    return batched_graph, torch.tensor(y+y_ns), pdb+pdb_ns


#%%
# test
# if __name__ == '__main__':
    # try:
    #     for mode in ['train', 'test', 'valid']:
    #         dataset = ProtFunctDatasetBinary('../data/ProtFunct.pt', mode='test', class_idx=0)
    #         dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_ns)

    #         for i, data in enumerate(dataloader):
    #             print(f'{i}')
    # finally:
    #     torch.save(residue2idx, 'res2idx_dict_tmp.pt')
    #     # torch.save(residue2count, 'res2count_dict.pt')

    # for mode in ['train', 'test', 'valid']:
    #     dataset = ProtFunctDataset('../data/ProtFunct.pt', mode=mode)
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
    #     for i, data in enumerate(dataloader):
    #         if not i % 1000:
    #             print(i)
    #         continue
    #         # print(f'{mode}: {i}')