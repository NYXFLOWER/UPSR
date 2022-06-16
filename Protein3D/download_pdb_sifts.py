from datasets import ProtProcess, data_dir
from Bio.PDB import PDBParser
import pandas as pd
from shutil import Error
import os
import torch

parser = PDBParser()
df = pd.read_csv('./ec1.csv', usecols=[0])

# df.CHAIN = [str(i).upper() for i in df.CHAIN]
df = df.drop_duplicates()

es_name = 'error_set2.pt'
if os.path.exists(es_name):
    error_set = torch.load(es_name)
else:
    error_set = set()

print(error_set)

df = df.values.reshape((-1,))
a = 65000
# for p, c in df.values[a:, :]:
try:
    for p in df[a:]:
        fp = f'{data_dir}/pdb/{p}.pdb'
        ProtProcess.download_pdb(p, fp)
        flag = True

        if not os.path.exists(fp):
            ProtProcess.download_pdb(p, fp, replace=True)
            if not os.path.exists(fp):
                print(p)
                error_set.add(p)
                continue
        
        while(flag):
            # generate node features
            try:
                structure = parser.get_structure('a', fp)
                # chain = structure[0][c]
                # res, x = ProtProcess.get_residue_feature(chain)
                # num_residues = res.shape[0]

                # assert num_residues == 0
                flag = False
            except Error as err:
                print('error: ', err)
                print(f'error pdb: {p}.{c}')
                ProtProcess.download_pdb(p, f'{data_dir}/pdb/{p}.pdb', replace=True)

        if a % 100 == 0:
            print(f'{a} / {df.shape[0]}')

        a += 1

except:
    torch.save(error_set, es_name)