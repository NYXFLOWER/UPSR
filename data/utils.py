import numpy as np
import pandas as pd
import torch
import pickle
import requests
from os import path
from colorama import Style, Fore
from Bio.PDB import parse_pdb_header


# ############## io #################
def save_pickle(obj, out_dir):
    with open(out_dir, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(out_dir):
    with open(out_dir, 'rb') as f:
        obj = pickle.load(f)
    return obj


def error(string):
    print(Fore.RED + 'Error: ' + string)


def high(string):
    print(Fore.YELLOW + 'Highlight:' + string)


def download_from_webpage(page, outfile, idx):
    req = requests.get(page)
    if req.status_code == 200:
        response = req.text

        if outfile:
            with open(outfile, 'w') as f:
                f.write(response)
        else:
            error('invalid output file! - {}'.format(idx))

        return response
    else:
        error('website no response! - {}'.format(idx))
        return None


# ############## PDB file processing #################
def get_resolutions(pdb_files, outdir_pdb):
    reso = []
    for pdb in pdb_files:
        structure = parse_pdb_header(path.join(outdir_pdb, str(pdb)+'.pdb'))
        reso.append(structure['resolution'])

    return reso