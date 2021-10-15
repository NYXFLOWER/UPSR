import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

# import pytorch_lightning as pl
import torchmetrics as tm

from datasets import *

EPS = 1e-13

# ##################### Hyperpremeter Setting #########################
class ExpSetting(object):
    def __init__(self, 
    distance_cutoff=[3, 3.5], 
    data_address='../data/ProtFunct.pt', 
    log_file=None, 
    log_dir = 'log/', 
    batch_size=4, 
    lr=1e-3, 
    num_epochs=2, 
    num_workers=4, 
    num_layers=2, 
    atom_feature_size=3, # SS
    num_degrees=3, 
    num_channels=20, 
    num_nlayers=0, 
    pooling='avg', 
    head=1, 
    div=4, 
    seed=0, 
    num_class=384, 
    use_classes=None, 
    hyperparameter=None, 
    decoder_mid_dim=60): 
        self.distance_cutoff = distance_cutoff
        self.data_address = data_address
        self.log_file = log_file
        self.log_dir = log_dir
        self.hyperparameter = hyperparameter

        self.batch_size = batch_size
        self.lr = lr          		 	  # learning rate
        self.num_epochs = num_epochs          
        self.num_workers = num_workers
        self.atom_feature_size = atom_feature_size # SS

        self.num_layers = num_layers      # number of equivariant layer
        self.num_degrees = num_degrees    # number of irreps {0,1,...,num_degrees-1}
        self.num_channels = num_channels  # number of channels in middle layers
        self.num_nlayers = num_nlayers    # number of layers for nonlinearity
        self.pooling = pooling        	  # choose from avg or max
        self.head = head                  # number of attention heads
        self.div = div                    # low dimensional embedding fraction
        self.decoder_mid_dim = decoder_mid_dim

        # self.num_class = num_class     # number of class in multi-class decoder
        self.num_class = len(use_classes) # SS
        self.max_class_label = num_class

        self.use_classes = use_classes

        self.seed = seed                  # random seed for both numpy and pytorch

        self.n_bounds = len(distance_cutoff) + 1
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')				 # Automatically choose GPU if available


class SE3TransformerEncoder(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, 
                num_layers: int,
                atom_feature_size: int, # SS
                num_channels: int, 
                num_nlayers: int=1, 
                num_degrees: int=4, 
                edge_dim: int=4, 
                div: float=4, 
                pooling: str='avg', 
                n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.aa_embed = nn.Linear(atom_feature_size, num_channels) # SS

        self.fibers = {'in': Fiber(1, self.num_channels), #SS
                    'mid': Fiber(num_degrees, self.num_channels),
                    'out': Fiber(1, num_degrees*self.num_channels)}

        self.Gblock = self._build_gcn(self.fibers, 1)
        # print(self.Gblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        return nn.ModuleList(Gblock)
    
    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        
        # SS
        # h = {'0': G.ndata['f']}
        m, _, _ = G.ndata['f'].size()
        z = self.aa_embed(G.ndata['f'].squeeze()).view(m, -1, 1)
        h = {'0': z}
        # SS

        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        return h


class MultiClassInnerProductLayer(nn.Module):
    def __init__(self, in_dim, num_class):
        super(MultiClassInnerProductLayer, self).__init__()
        self.num_class = num_class
        self.in_dim = in_dim

        self.embedding = nn.Parameter(torch.Tensor(self.in_dim, self.num_class))
        self.reset_parameters()

    def __repr__(self):
        return f'MultiClassInnerProductLayer(structure=[(batch_size, {self.num_class})]'

    def forward(self, z):
        # value = (z[node_list] * self.weight[node_label]).sum(dim=1)
        # value = torch.sigmoid(value) if sigmoid else value

        pred = torch.matmul(z, self.embedding)
        return pred

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.embedding.size(-2) + self.embedding.size(-1)))
        self.embedding.data.uniform_(-stdv, stdv)
        # self.weight.data.normal_()


class ProtMultClassModel(nn.Module):
    def __init__(self, setting: ExpSetting):
        super().__init__()
        self.setting = setting
        self.model = self.__build_model()
        self.class_dic = {}
        i = 0
        for c in self.setting.use_classes:
            self.class_dic[c] = i
            i += 1

        self.loss_history = []

    def class_recast(self, x):
        y = torch.tensor([self.class_dic[i.tolist()] for i in x])
        return y.cuda()

    def __build_model(self):
        model = []

        model.append(
            SE3TransformerEncoder(
            self.setting.num_layers, 
            self.setting.atom_feature_size, # SS 
            self.setting.num_channels, 
            self.setting.num_nlayers, 
            self.setting.num_degrees, 
            edge_dim=3, 
            n_bonds=self.setting.n_bounds, 
            div=self.setting.div, 
            pooling=self.setting.pooling, 
            head=self.setting.head)
            )

        mid_dim = model[0].fibers['out'].n_features

        # model.append(nn.Linear(mid_dim, self.setting.decoder_mid_dim))
        # model.append(nn.ReLU(inplace=True))

        model.append(MultiClassInnerProductLayer(mid_dim, self.setting.num_class))

        return nn.ModuleList(model)

    def forward(self, g):
        """get model prediction"""
        prob = self._run_step(g)

        return prob

    def _run_step(self, g):
        """compute forward"""
        z = g
        for layer in self.model:
            z = layer(z)
        return z

    # def __to_onehot(self, y_list):
    #     # convert class number to onehot representation

    #     return F.one_hot(y_list, num_classes=self.setting.num_class)

    def __compute_epoch_metrics(self, mode):
        outputs = self.metrics_dict[mode].compute()
        self.metrics_dict[mode].reset()

        return outputs

    def forward(self, g, mode='train'):
                
        # targets_recast = self.class_recast(targets)

        z = g
        for layer in self.model:
            z = layer(z)

        return z


def load_data(setting):
    tmp = []
    for mode in ['train', 'valid', 'test']:
        dataset = ProtFunctDatasetMultiClass(
            setting.data_address, 
            mode=mode, 
            if_transform=False, 
            dis_cut=setting.distance_cutoff,
            use_classes=setting.use_classes)

        loader = DataLoader(
            dataset, 
            batch_size=setting.batch_size, 
            shuffle=True, 
            collate_fn=collate, 
            num_workers=setting.num_workers)
        
        tmp.append(loader)
    
    return tmp


def get_class_recaster(use_classes):
    dic = {}
    for i, c in enumerate(use_classes):
        dic[c] = i
    
    def tmp(x):
        y = [dic[c.detach().tolist()] for c in x]
        return torch.tensor(y)
    
    return tmp


