from models import ExpSetting
from datasets import collate, ProtFunctDatasetMultiClass
from torch.utils.data import DataLoader
import numpy as np
import torch


def load_data_ie(setting: ExpSetting):
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

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_class_recaster(use_classes):
    dic = {}
    for i, c in enumerate(use_classes):
        dic[c] = i
    
    def tmp(x):
        y = [dic[c.detach().tolist()] for c in x]
        return y[0] if len(y) == 1 else torch.tensor(y)
    
    return tmp