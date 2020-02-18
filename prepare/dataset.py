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
out_dir = './data'

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
# plot samples' distribution
# ##############################################################################
with open('./prepare/extracted_data/pos_data.pkl', 'br') as f:
    pos_data = pickle.load(f)
with open('./prepare/extracted_data/neg_data.pkl', 'br') as f:
    neg_data = pickle.load(f)

s = 50000
a = np.array([i['aa_seq'].shape[0] for i in pos_data.values()])
b = np.array([i['aa_seq'].shape[0] for i in neg_data.values()])
a, b = a[a < s], b[b < s]
tt = np.append(a, b)
_, bins, _ = plt.hist(tt, 500, alpha=0.9, label='All')
plt.hist(a, bins, facecolor='g', alpha=0.5, label='Pos')
plt.title('Atom Length Histograms with threshold {}'.format(s))
plt.legend()
plt.savefig('./prepare/fig/hist{}.png'.format(s))
plt.show()


# ##############################################################################
# positive and negative sample processing
# ##############################################################################
from sklearn.neighbors import NearestNeighbors

with open('tmp.pkl', 'bw') as f:
    pickle.dump(pdb_data, f)

'''
    data.n_node: number of nodes
    data.n_node_type: number of node types (aka number of atom type)
    data.n_edge: number of edges

    data.node_type: int array of the shape (n_node,). Element i is the atom type of i th atom.
    data.edge_index: int array of the shape (2, n_edge). There is an edge from node i to node j if node i is the node j's d th nearest neighour and d < k.
    data.edge_direction: float array of the shape (3, n_edge). The i th column is the 3d vector from s[i] to t[i]. s[i] and t[i] are the target and source of i th edge.
'''

