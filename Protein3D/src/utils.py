import pickle
import runpy
import torch
import sys
import os


def load_dataset(root='.', k_neigh=5, threshold=20000):
    """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        root : string, root path of this repo
        k_neigh : int, optional (default = 5)
            Number of neighbors to use Mixin for k-neighbors searches

        threshold : int, optional (default = 20000)
            Maximum number of atoms in a protein

        Returns
        -------
        pos_data_list : list of {data : torch_geometric.data.Data}
        neg_data_list : list of {data : torch_geometric.data.Data}
            - data.n_node: number of nodes
            - data.n_node_type: number of node types (atom type)
            - data.n_edge: number of edges
            - data.node_type: torch.Tensor, int, shape (n_node,).
                Element i is the atom type of i th atom.
            - data.edge_index: torch.Tensor, int, shape (2, n_edge).
                There is an edge from node i to node j if node i is the
                node j's d th nearest neighour and d < k.
            - data.edge_direction: torch.Tensor, int, shape (3, n_edge).
                The i th column is the 3d vector from s[i] to t[i].
                s[i] and t[i] are the target and source of i th edge.

        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]
    """

    name = '{}_{}'.format(k_neigh, threshold)
    name_path = os.path.join(root, 'data', 'processed', name)

    if not os.path.exists(name_path):
        sys.argv = ['_', 'arg1 arg2']
        runpy.run_path(root + '/src/' + 'dataset_mp.py', run_name='__main__')

    return torch.load(os.path.join(name_path, 'pos.pt')), \
        torch.load(os.path.join(name_path, 'neg.pt'))




