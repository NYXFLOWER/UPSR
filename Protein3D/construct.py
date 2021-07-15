#%%
import os
import pickle
import torch
import requests
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch

import numpy as np
import pandas as pd

os.path.abspath(os.curdir)
os.chdir('/home/flower/github/Protein3D/Protein3D')
os.path.abspath(os.curdir)

data_dir = '../data'

class PreProcess:
    @staticmethod
    def download_pdb(pdb_id, outfile):
        # check if the pdb file has been downloaded in the outfile
        if os.path.exists(outfile):
            print(f"The file has been downloaded...")
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

#%%
residue2idx = {}
bond2idx = {'covalent':0, 'neighbor<2.5':1, 'neighbor<3.5': 2}
dis_cutoff = [3, 3.5]

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
            residue2idx[res_name] = len(residue2idx)
        res_list.append(residue2idx[res_name])

        # compute coo of residue by averaging its atoms' coos
        coo = np.concatenate([i.get_coord() for i in res]).reshape((-1, 3)).mean(axis=0)
        coo_list.append(coo)

    return res_list, np.concatenate(coo_list).reshape((-1, 3))

def get_edge_set(dis_cut, l, level='R'):
    ns_list = ns.search_all(dis_cut, level=level)

    edge = set()
    for ai, aj in ns_list:
        i, j = ai.get_id()[1]-1, aj.get_id()[1]-1
        if i < j-1 and j < l and i > 0:
            edge.add((i, j))

    return edge


#%%
mod = 'train'
out = {'train': {}, 'test':{}, 'valid':{}}

# load train samples
mod = 'train'

#%%
datasp = {}
pdb_list = pd.read_csv(f'{data_dir}/ProtFunct/training.txt', header=None)[0].values
#%%
chain_func = pd.read_csv(f'{data_dir}/ProtFunct/chain_functions.txt', header=None, sep=',')
pdb2func = {k:v for k, v in chain_func.values}
#%%
pdb = pdb_list[0]

data_pdb = {}
data_pdb['pdb_id'] = pdb

p, c = pdb.split('.') 

parser = PDBParser()
if PreProcess.download_pdb(p, f'{data_dir}/pdb/{p}.pdb'):
    structure = parser.get_structure('a', f'{data_dir}/pdb/{p}.pdb')
    chain = structure[0][c]

    res_list, coo_list = get_residue_feature(chain)
    data_pdb['res_list'] = res_list
    data_pdb['coo_ndarray'] = coo_list

    atom_list = [i for i in chain.get_atoms()]
    ns = NeighborSearch(atom_list)
    l = len(res_list)

    # edge_set_list = [{(i-1, i) for i in range(1, l)}]
    edge_set_list = [get_edge_set(cut, l) for cut in dis_cutoff]
    edge_set_list[1] = edge_set_list[1] - edge_set_list[0] 

target = pdb2func[pdb]


#%%
# Dataset
from torch.utils.data import Dataset, DataLoader

#%%
class ProtFunctDataset(Dataset):
    def __init__(self, file_path, mode: str='train', if_transform: bool=True):
        """Create a dataset object

        Args:
            file_path (str): path to data
            mode (str, optional): {train/test/val}. Defaults to 'train'.
            if_transform (bool, optional): if applying data augmentation function. Defaults to True.
        """

        self.file_path = file_path
        self.mode = mode
        
        self.__load_data()
        self.len = len(self.targets)     # TODO

    def __load_data(self):
        """Load preprocessed dataset
        """
        data = torch.load(self.file_path)[self.mode]
        self.inputs = data['input_list']
        self.targets = data['target_list']

    def __len__(self):
        return self.len

#%%
# test
dataset = ProtFunctDataset('../data/ProtFunct.pt', mode='train')





    


    


    



# %%



# %%
