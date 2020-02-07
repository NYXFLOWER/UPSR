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
start = int(sys.argv[1])
try:
    end = int(sys.argv[2])
except IndexError:
    end = data.shape[0]

# init out
pos_data, neg_data = OrderedDict(), OrderedDict()
err_list = []
# key: pdb;
# value: {aa_seq: ndarray(int8), at_seq: ndarray(int8), coo: ndarray(float16)}

with tqdm(total=end - start, file=sys.stdout) as pbar:
    for idx in range(start, end):
        # init
        pdb, ifso, _, _ = data[idx]

        # load pdb data
        try:
            with open('../PS_jj/data/ppp/{}.pdb'.format(pdb), 'r') as f:
                pdb_data = f.read().replace('\n', '')
        except FileNotFoundError:
            pbar.set_description('processed: %d' % (1 + idx))
            pbar.update(1)
            continue
        # process text data
        try:
            match_obj = re.search('ATOM {4}[ \d][ \d]\d', pdb_data)
            pdb_data = pdb_data[match_obj.span()[0]:]
            match_obj = re.search('TER', pdb_data)
            pdb_data = pdb_data[:match_obj.span()[0]].replace('0100.', '0 100.').split()
            pdb_data = np.array(pdb_data).reshape((-1, 12))

            aa_seq = np.array([aa_sym_to_idx[i] for i in pdb_data[:, 3]],
                              dtype=np.int8)
            coo = pdb_data[:, [6, 7, 8]].astype(np.float16)
            at_seq = []
            for i in pdb_data[:, 11]:
                try:
                    at_seq.append(at_sym_to_idx[i])
                except KeyError:
                    j = len(at_sym_to_idx)
                    at_sym_to_idx[i] = j
                    at_seq.append(j)
            at_seq = np.array(at_seq, dtype=np.int8)

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
#
# pdb = '3DKW'
# with open('../PS_jj/data/ppp/3DKW.pdb', 'r') as f:
#     pdb_data = f.read().replace('\n', '')
#
# match_obj = re.search('ATOM {4}[ \d][ \d]\d', pdb_data)
# if not match_obj:
#     pass  # //TODO
# pdb_data = pdb_data[match_obj.span()[0]:]
# match_obj = re.search('TER', pdb_data)
# pdb_data = pdb_data[:match_obj.span()[0]].replace('0100.', '0 100.').split()
# pdb_data = np.array(pdb_data).reshape((-1, 12))
#
# aa_seq = np.array([aa_sym_to_idx[i] for i in pdb_data[:, 3]], dtype=np.int8)
# coo = pdb_data[:, [6, 7, 8]].astype(np.float16)
#
# # //TODO: save and try
# at_seq = []
# for i in pdb_data[:, 11]:
#     try:
#         at_seq.append(at_sym_to_idx[i])
#     except KeyError:
#         j = len(at_sym_to_idx)
#         at_sym_to_idx[i] = j
#         at_seq.append(j)
# at_seq = np.array(at_seq, dtype=np.int8)
