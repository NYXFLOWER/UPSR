import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import re
from Bio.Blast import NCBIWWW
from ssbio.pipeline.gempro import GEMPRO
from os import path
from data.utils import *
from tqdm import tqdm
import sys
from time import sleep

dir = './data/deepsol/'
aa_seq = np.concatenate((np.loadtxt(dir+'train_src', dtype='str'),
                        np.loadtxt(dir+'test_src', dtype='str'),
                        np.loadtxt(dir+'val_src', dtype='str')), axis=0)
# sol = np.concatenate((np.loadtxt(dir+'train_tgt', dtype=int),
#                       np.loadtxt(dir+'test_tgt', dtype=int),
#                       np.loadtxt(dir+'val_tgt', dtype=int)), axis=0)
# assert aa_seq.shape[0] == sol.shape[0]
print('There are {:d} samples in the DeepSol datasets.'.format(aa_seq.shape[0]))

# l = []
# for i in aa_seq:
#     l.append(len(i))
# print('The length of amino acid is ranged from {} to {}.'.format(min(l), max(l)))

# evalue=0.0001
evalue = float('1E-20')

outdir_pdb = './data/pdb'

# ################### place holder ##################
# aa_seq | pdb_idx | solubility | matched pdb | if_reviewed
# data = np.chararray(shape=[aa_seq.shape[0], 5], itemsize=1700)
# , unicode=True
start, end = int(sys.argv[1]), int(sys.argv[2])

# ######## query and get matched pdb indices ########
with tqdm(total=end-start, file=sys.stdout) as pbar:
    for idx in range(start, end):
        out = []
        seq = aa_seq[idx]

        # write to dataset table
        # aa_seq | pdb_idx | solubility | matched pdb | if_reviewed
        # data[idx, 0] = seq
        # data[idx, 1] = sol[idx]

        # query on pdb-blast
        page = 'http://www.rcsb.org/pdb/rest/getBlastPDB1?' \
               'sequence={}&' \
               'eCutOff={}&' \
               'maskLowComplexity=yes&' \
               'matrix=BLOSUM62&' \
               'outputFormat=HTML'.format(seq, evalue)
        outfile = path.join('./data/pdb-blast', str(idx) + '.txt')
        response = download_from_webpage(page, outfile, idx)
        if response:
            response = re.split(r'name', response)

            # retrieve fully matched protein structure indices which have 100% identity
            matched_pbds = []
            if response.__len__() == 1:
                # error('no retrieved pdb file - {}'.format(idx))
                pass
            else:
                #         print(response.__len__())
                for i in range(1, response.__len__()):
                    try:  # parsing blast result
                        st = re.search('Identities = ', response[i]).span()[1]
                        ed = re.search(', Positives', response[i]).span()[0]
                        if re.search('100', response[i][st: ed]):
                            lo = re.search('<\/a>', response[i]).span()[1]
                            matched_pbds.append(response[i][lo:(lo + 4)])
                    except:
                        # error('paring response false - {}'.format(idx))
                        pass

                # download matched pdb files and parse the resolution
                pdb_files = []
                if matched_pbds:
                    for pdb in matched_pbds:
                        page = 'http://files.rcsb.org/view/{}.pdb'.format(pdb)
                        req = requests.get(page)
                        if req.status_code == 200:
                            response = req.text
                            outfile = path.join(outdir_pdb, str(pdb) + '.pdb')
                            if outfile:
                                with open(outfile, 'w') as f:
                                    f.write(response)
                                pdb_files.append(pdb)
                            else:
                                # error(
                                #     'can not write to file - idx:{}, pdb:{}'.format(
                                #         pdb, idx))
                                continue
                        else:
                            # error(
                            #     'web page no response for pdb download - {}'.format(
                                    # idx))
                            continue

                    if pdb_files:
                        reso = get_resolutions(pdb_files, outdir_pdb)
                        if reso:
                            tmp = np.array(reso).astype(float)
                            i_max = np.argmax(np.nan_to_num(tmp))
                            #         print(pdb_files[i_max], reso[i_max])
                            out.append(str(pdb_files[i_max]))  # write to dataset table
                            out.append(str(list(zip(pdb_files, reso))))
                            with open('./data/pdb-pkl/{}.pkl'.format(idx), 'wb') as f:
                                pickle.dump(out, f)
                        # print(idx)

        pbar.set_description('processed: %d' % (1 + idx))
        pbar.update(1)
        sleep(1)

# data_table = pd.DataFrame(data=data, columns=['aa_seq', 'solubility', 'pdb_idx', 'matched pdb', 'if_reviewed'])
# data_table.to_csv(r'./dataset.csv')

