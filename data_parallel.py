from multiprocessing import Pool
import numpy as np
import sys

dir = './data/deepsol/'
aa_seq = np.concatenate((np.loadtxt(dir+'train_src', dtype='str'),
                        np.loadtxt(dir+'test_src', dtype='str'),
                        np.loadtxt(dir+'val_src', dtype='str')), axis=0)


def f(x):
    return x


start, end = int(sys.argv[1]), int(sys.argv[2])

with Pool(4) as p:
    print(p.map(f, aa_seq[start: end]))


