from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import sys
import re

# ############# load index map #############
with open('./prepare/map/aa-sym-to-idx.pkl', 'br') as f:
    aa_sym_to_idx = pickle.load(f)
with open('./prepare/map/at-sym-to-idx.pkl', 'br') as f:
    at_sym_to_idx = pickle.load(f)

# ############# load processed deepsol dataset #############
data = pd.read_csv('./prepare/data/dataset_bio.csv', usecols=range(1, 5))
columns = data.columns.to_list()
data = data.to_numpy()

# ############# process #############
# with open('./prepare/extracted_data/err_idx_list.pkl', 'br') as f:
#     err_list = pickle.load(f)
err_list = []
with open('./prepare/extracted_data/pos_data.pkl', 'br') as f:
    pos_data = pickle.load(f)
with open('./prepare/extracted_data/neg_data.pkl', 'br') as f:
    neg_data = pickle.load(f)
# key: pdb;
# value: {aa_seq: ndarray(int8), at_seq: ndarray(int8), coo: ndarray(float16)}
end = data.shape[0]
start = 0
with tqdm(total=end - start, file=sys.stdout) as pbar:
    for idx in range(start, end):
        # init
        pdb, ifso, _, _ = data[idx]

        # load pdb data
        try:
            with open('../PS_jj/data/ppp/{}.pdb'.format(pdb), 'r') as f:
                pdb_data = f.read().replace('\n', ',')
            # match_obj = [i for i in re.finditer(' TER ', pdb_data)]
            # if len(match_obj) > 0:
            #     pdb_data = pdb_data[:match_obj[-1].span()[0]]
        except FileNotFoundError:
            pbar.set_description('processed: %d' % (1 + idx))
            pbar.update(1)
            continue
        # process text data
        try:
            # arrive at ATOM section
            match_obj = re.search('ATOM {4}[ \d][ \d]\d', pdb_data)
            pdb_data = pdb_data[match_obj.span()[0]:].split(',')

            # atom sequence and residue sequence
            aa_seq, at_seq = [], []
            for i in pdb_data:
                if i[:4] == 'ATOM':
                    m = i[76:78].replace(' ', '')
                    n = i[17:20].replace(' ', '')
                    try:
                        at_seq.append(at_sym_to_idx[m])
                    except KeyError:
                        j = len(at_sym_to_idx)
                        at_sym_to_idx[m] = j
                        at_seq.append(j)
                    try:
                        aa_seq.append(aa_sym_to_idx[n])
                    except KeyError:
                        k = len(aa_sym_to_idx)
                        aa_sym_to_idx[n] = k
                        aa_seq.append(k)
            at_seq = np.array(at_seq, dtype=np.int8)
            aa_seq = np.array(aa_seq, dtype=np.int8)

            # coordinate matrix - x y z
            coo = np.array([[float(i[30:38]), float(i[38:46]), float(i[46:54])]
                            for i in pdb_data if i[:4] == 'ATOM'],
                           dtype=np.float16)

            out = {'aa_seq': aa_seq, 'at_seq': at_seq, 'coo': coo}

            if ifso:
                pos_data[pdb] = out
            else:
                neg_data[pdb] = out
        except:
            err_list.append(idx)

        # update process bar
        pbar.set_description('processed: %d' % (1 + idx))
        pbar.update(1)


# update atom symbol index map
with open('./prepare/map/at-sym-to-idx.pkl', 'bw') as f:
    pickle.dump(at_sym_to_idx, f)

# save extracted_data info to path
with open('./prepare/extracted_data/pos_data.pkl', 'bw') as f:
    pickle.dump(pos_data, f)
with open('./prepare/extracted_data/neg_data.pkl', 'bw') as f:
    pickle.dump(neg_data, f)

# save unprocessed index in dataset_bio.csv file
with open('./prepare/extracted_data/err_idx_list.pkl', 'bw') as f:
    pickle.dump(err_list, f)
print('skip {} samples'.format(len(err_list)))

