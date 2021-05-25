#%%
import os
import pickle
import torch
import requests
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch

import numpy as np

os.path.abspath(os.curdir)
os.chdir('/home/flower/github/Protein3D/Protein3D')
os.path.abspath(os.curdir)

class PreProcess:
    @staticmethod
    def download_pdb(pdb_id, outfile):
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
bond2idx = {'covalent':0, 'neighbor':1, 'SSBOND': 2, 'LINK': 3}

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


# %%
p, c = "4fae", "B"
parser = PDBParser()
if PreProcess.download_pdb(p, f'{p}.pdb'):
    structure = parser.get_structure('a', f'{p}.pdb')
    chain = structure[0][c]

    res_list, coo_list = get_residue_feature(chain)

    atom_list = [i for i in chain.get_atoms()]
    ns = NeighborSearch(atom_list)
    ns_list = ns.search_all(3.3, level='R')

    edge, l = [], len(res_list)
    for ai, aj in ns_list:
        i, j = ai.get_id()[1]-1, aj.get_id()[1]-1
        if i < j-1 and j < l:
            edge.append([i, j, 1])

#%%
mod = 'train'
out = {'train': {}, 'test':{}, 'valid':{}}
out[mod]['edge'] = 






    


    


    



# %%



# %%
