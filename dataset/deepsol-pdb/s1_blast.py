import json
from os import makedirs, path

import pandas as pd
import requests
from Bio.PDB import parse_pdb_header
from colorama import Style, Fore

in_file = "../deepsol/pos_aas.csv"
aa_seqs = pd.read_csv(in_file).to_numpy().flatten().tolist()

outdir_pdb = '../pdb'
if not path.exists(outdir_pdb):
    makedirs(outdir_pdb)


def error(string):
    print(Fore.RED + 'Error: ' + string + f'{Style.RESET_ALL}')


evalue = float('1E-20')


def blast_aa_seq(aa_sequence, index):
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
        return req.text
    else:
        error('website no response for the AAs BLAST! - {}'.format(index))
        return None


for i in range(0, len(aa_seqs)):
    seq = aa_seqs[i]
    response = blast_aa_seq(seq, i)
    pdb_reso = []

    try:
        results = json.loads(response)['result_set']
        for r in results:
            pdb_id = None
            pdb_full = None
            # match
            info = r['services'][0]['nodes'][0]['match_context'][0]
            if info['mismatches'] == 0 and info['gaps_opened'] == 0 and info['query_length'] == info['subject_length']:
                pdb_full = r['identifier']
                pdb_id = pdb_full.split('_')[0]

            # if match, download pdb file
            if pdb_id and pdb_full:
                page = 'http://files.rcsb.org/view/{}.pdb'.format(pdb_id)
                req = requests.get(page)
                if req.status_code == 200:
                    response = req.text
                    outfile = path.join(outdir_pdb, str(pdb_id) + '.pdb')
                    if outfile:
                        with open(outfile, 'w') as f:
                            f.write(response)
                        # parse to get the resolution
                        structure = parse_pdb_header(outfile)
                        pdb_reso.append((pdb_full, structure['resolution']))

        # append to dataset file
        if pdb_reso:
            # find the pdb with best resolution
            tmp_dict = {r: p for p, r in pdb_reso}
            best_pdb_id = tmp_dict[max(tmp_dict.keys())]
            # write to file
            with open('./dataset_pos.csv', 'a') as f:
                f.write("{}, {}, {}\n".format(best_pdb_id, seq, pdb_reso))
                print("{} - {}".format(i, pdb_reso))
    except:
        pass



