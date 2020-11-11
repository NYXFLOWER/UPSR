import numpy as np
import pandas as pd

aa_dir = 'dataset/deepsol/'
aa_seq = np.concatenate((np.loadtxt(aa_dir + 'train_src', dtype='str'),
                        np.loadtxt(aa_dir + 'test_src', dtype='str'),
                        np.loadtxt(aa_dir + 'val_src', dtype='str')), axis=0)
sol = np.concatenate((np.loadtxt(aa_dir + 'train_tgt', dtype=int),
                      np.loadtxt(aa_dir + 'test_tgt', dtype=int),
                      np.loadtxt(aa_dir + 'val_tgt', dtype=int)), axis=0)

assert aa_seq.shape[0] == sol.shape[0]

pos_aas = pd.DataFrame(aa_seq[sol == 1], columns=['AAseq'])
pos_aas.to_csv(aa_dir + "pos_aas.csv", index=False)

neg_aas = pd.DataFrame(aa_seq[sol == 0], columns=['AAseq'])
neg_aas.to_csv(aa_dir + "neg_aas.csv", index=False)
