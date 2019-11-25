"""" Used for PS-dataset construction"""

from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PPBuilder
from Bio.PDB import Selection


pkl_dir = '../PS_data/data/pdb-pkl'
pdb_dir = '../PS_data/data/pdb'
aa_dir = '../PS_data/data/deepsol/'

# ##############################################################################
# load amino acid sequence and its solubility
# ##############################################################################
aa_seq = np.concatenate((np.loadtxt(aa_dir + 'train_src', dtype='str'),
                        np.loadtxt(aa_dir + 'test_src', dtype='str'),
                        np.loadtxt(aa_dir + 'val_src', dtype='str')), axis=0)
sol = np.concatenate((np.loadtxt(aa_dir + 'train_tgt', dtype=int),
                      np.loadtxt(aa_dir + 'test_tgt', dtype=int),
                      np.loadtxt(aa_dir + 'val_tgt', dtype=int)), axis=0)

# len_aa_max = max([len(a) for a in aa_seq])

# ##############################################################################
# construct dataset table
# ##############################################################################
files = os.listdir(pkl_dir)

# index map from ps_dataset to deepsol
idx_map = [int(file[:-4]) for file in files]
dataset_size = len(idx_map)

np.savetxt('./data/map_to_deepsol.txt', idx_map, fmt='%d')

# place holder for dataset
data = np.zeros(dataset_size, dtype=[('PDB_index', 'U4'),
                                          ('If_soluble', '<i1'),
                                          ('Amino_acid_sequence', 'U1700'),
                                          ('BLAST_result', 'U4000')])

# dataset construction
for i in range(dataset_size):
    with open(os.path.join(pkl_dir, files[i]), 'rb') as f:
        tmp = pickle.load(f)
    data[i][0], data[i][3] = tmp
    data[i][1], data[i][2] = sol[i], aa_seq[i]

data = pd.DataFrame(data)

# save to csv
data.to_csv('./data/dataset.csv')

# ##############################################################################
# positive and negative sample processing
# ##############################################################################
data = pd.read_csv('./data/dataset_bio.csv')

pdb_idx = data['PDB_index']
parse = PDBParser(PERMISSIVE=1)

# for i in range(4):
i = 1
structure = parse.get_structure(i, os.path.join(pdb_dir, pdb_idx[i]+'.pdb'))


# p = PDBParser(PERMISSIVE=1)
# s = p.get_structure('1cz4', '1CZ4.pdb')
#
a_list = Selection.unfold_entities(structure, 'A')
# r_list = Selection.unfold_entities(s, 'R')
# p = np.array([a.get_coord() for a in a_list])
# rr = [r.get_resname() for r in r_list]
