from py3pdb.download import download_pdb
import os
import pickle

ec3 = [3, 4, 24]
in_dir = 'ec_data/ec_{}/ec_{}_{}'.format(ec3[0], ec3[0], ec3[1])
in_name = 'ec_pdb_{}_{}_{}.pkl'.format(ec3[0], ec3[1], ec3[2])

with open(os.path.join(in_dir, in_name)) as file:
    ec_pdb = pickle.load(file)

