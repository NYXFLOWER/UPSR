import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import sys
from os import path
import requests
from colorama import Style, Fore

in_file = "dataset/deepsol/pos_aas.csv"
aa_seqs = pd.read_csv(in_file).to_numpy().flatten().tolist()


def error(string):
    print(Fore.RED + 'Error: ' + string)


def download_from_webpage(aa_sequence, index):
    page = """https://search.rcsb.org/rcsbsearch/v1/query?json=
        {"query": {"type": "terminal", "service": "sequence",
                   "parameters": {"evalue_cutoff": 1, 
                                  "identity_cutoff": 1, 
                                  "target": "pdb_protein_sequence",
                                  "value": \"""" + str(aa_sequence) + """\"}},
        "request_options": {"scoring_strategy": "sequence"},
        "return_type": "polymer_entity"}"""

    req = requests.get(page)
    if req.status_code == 200:
        response = req.text
        outfile = path.join('dataset/pdb_blast_json', str(index) + '.txt')

        if outfile:
            with open(outfile, 'w') as f:
                f.write(response)
        else:
            error('invalid output file! - {}'.format(index))

        return response
    else:
        error('website no response! - {}'.format(index))
        return None


for i in range(0, aa_seqs):
    seq = aa_seqs[i]
    response = download_from_webpage(seq, i)

    try:
        results = json.loads(response)['result_set']
        for r in results:
            info = r['services'][0]['nodes'][0]['match_context'][0]
            if info['mismatches'] == 0 & info['gaps_opened'] == 0 & info['query_length'] == info['subject_length']:

    except:
        pass



