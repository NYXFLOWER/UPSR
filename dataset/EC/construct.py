#%%
import os
import pickle
import requests

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


ec3 = [3, 4, 24]
in_name = 'ec_pdb_{}_{}_{}.pkl'.format(ec3[0], ec3[1], ec3[2])

with open(os.path.join(in_name), 'rb') as file:
    ec_pdb = pickle.load(file)


# %%
from Bio.PDB import PDBParser

parser = PDBParser()
if download_pdb('1nqd', '1nqd.pdb'):
    structure = parser.get_structure('aaaaa1', '1nqd.pdb')
    print(structure)

# %%
