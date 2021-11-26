from datasets import ProtProcess, data_dir
from Bio.PDB import PDBParser
import pandas as pd
from shutil import Error

parser = PDBParser()
df = pd.read_csv('./ec1.csv', usecols=[0])

# df.CHAIN = [str(i).upper() for i in df.CHAIN]
df = df.drop_duplicates()

df = df.values.reshape((-1,))
a = 0
# for p, c in df.values[a:, :]:
for p in df[a:]:
    ProtProcess.download_pdb(p, f'{data_dir}/pdb/{p}.pdb')
    flag = True
    while(flag):
        # generate node features
        try:
            structure = parser.get_structure('a', f'{data_dir}/pdb/{p}.pdb')
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