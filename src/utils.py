import numpy as np
import pandas as pd
import torch
import pickle


# ############## io #################
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


# ############## PDB file processing #################
class Pdb:


