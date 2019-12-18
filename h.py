import shutil
import pandas as pd
import os

data = pd.read_csv('./data/dataset_bio.csv')
pdb = data['PDB_index']
count = 0

for i in pdb:
    in_file = '/Users/nyxfer/Docu/PS_data/data/pdb/{}.pdb'.format(i)
    if os.path.isfile(in_file):
        out_file = '/Users/nyxfer/Docu/PS/data/ppp/{}.pdb'.format(i)
        shutil.move(in_file, out_file)
        count += 1

print(count)