from multiprocessing import Pool, current_process, cpu_count
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import numpy as np
import torch
import sys, os, time, pickle

# _, pre, root, k_neigh, threshold = sys.argv     # pos ./data/processed 2 100
pre, root, k_neigh, threshold = 'pos', './data/processed', '2', '10000000'
k_neigh, threshold = int(k_neigh), int(threshold)

with open('./data/raw_sample/{}_data.pkl'.format(pre), 'br') as f:
    raw_data = pickle.load(f)

raw_data = {'3DKW': raw_data['3DKW'], '2VO9': raw_data['2VO9']}

print("Number of cpu : ", cpu_count())


def process_pdb(pdb_coo):
    pdb = pdb_coo[0]
    coo = pdb_coo[1]['coo']

    if coo.shape[0] < threshold:
        neigh = NearestNeighbors(k_neigh + 1)
        neigh.fit(coo)
        neigh_idx = neigh.kneighbors(coo, return_distance=False)[:, 1:]

        data = Data()
        data.pdb = pdb
        data.n_nodes = coo.shape[0]
        data.node_type = pdb_coo[1]['at_seq']
        data.n_node_type = np.unique(data.node_type).shape[0]
        data.n_edges = k_neigh * data.n_nodes
        data.edge_index = np.concatenate((
            np.array([[i] * k_neigh for i in range(data.n_nodes)]).reshape(
                (1, -1)),
            neigh_idx.reshape((1, -1))),
            axis=0)
        data.edge_direction = coo[data.edge_index[0]] - coo[data.edge_index[1]]

        return data
    else:
        return


def pool_handler():
    p = Pool(100)
    result = p.map(process_pdb, raw_data.items())
    return result

if __name__ == '__main__':
    a = pool_handler()


