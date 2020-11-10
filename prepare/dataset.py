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

threshold = 50000
a = np.array([i['aa_seq'].shape[0] for i in pos_data.values()])
b = np.array([i['aa_seq'].shape[0] for i in neg_data.values()])
a, b = a[a < threshold], b[b < threshold]
tt = np.append(a, b)
_, bins, _ = plt.hist(tt, 500, alpha=0.9, label='All')
plt.hist(a, bins, facecolor='g', alpha=0.5, label='Pos')
plt.title('Atom Length Histograms with threshold {}'.format(threshold))
plt.legend()
plt.savefig('./prepare/fig/hist{}.png'.format(threshold))
plt.show()


# ##############################################################################
# positive and negative sample processing
# ##############################################################################
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch

pdb = pos_data['5L9U']
coo = pdb['coo']

k_neigh = 3
threshold = 20000
neigh = NearestNeighbors(k_neigh)
neigh.fit(coo)
neigh_idx = neigh.kneighbors(coo, return_distance=False)[:, 1:]

data = Data()

data.prot_name = pdb
data.n_nodes = coo.shape[0]

data.node_type = pdb['at_seq']
data.n_node_type = np.unique(data.node_type).shape[0]

data.n_edges = k_neigh * data.n_nodes
data.edge_index = np.concatenate((np.array([[i] * k_neigh for i in range(data.n_nodes)]).reshape((1, -1)),
                                 neigh_idx.reshape((1, -1))),
                                 axis=0)
data.edge_direction = coo[data.edge_index[0]] - coo[data.edge_index[1]]
torch.save([data, data], './data/{}-{}.pt'.format(k_neigh, threshold))

aaa = torch.load('./data/tmp.pt')

'''
    data.n_node: number of nodes
    data.n_node_type: number of node types (aka number of atom type)
    data.n_edge: number of edges

    data.node_type: int array of the shape (n_node,). Element i is the atom type of i th atom.
    data.edge_index: int array of the shape (2, n_edge). There is an edge from node i to node j if node i is the node j's d th nearest neighour and d < k.
    data.edge_direction: float array of the shape (3, n_edge). The i th column is the 3d vector from s[i] to t[i]. s[i] and t[i] are the target and source of i th edge.
'''






import plotly.graph_objects as go
import pandas as pd
import numpy as np
fig = go.Figure(data=go.Scatter3d(
    x=coo[:, 0], y=coo[:, 1], z=coo[:, 2],
    marker=dict(
        size=4,
        color=pdb['at_seq']
    )
))

fig.show()