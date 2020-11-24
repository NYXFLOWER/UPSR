import py3pdb
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

root = './'
# root = './dataset/bindingDB/'
file_name = 'BindingDB_All_3D.sdf'
tmp_name = 'tmp.sdf'

out = PandasTools.LoadSDF(root + file_name)

ooo = pd.DataFrame({'PubChem CID': out['PubChem CID'],
                    'Target Name': out['Target Name Assigned by Curator or DataSource'],
                    'ki': out['Ki (nM)'],
                    'kd': out['Kd (nM)'],
                    'ic50': out['IC50 (nM)'],
                    'Target Organism': out['Target Source Organism According to Curator or DataSource'],
                    'Target PDB': out['PDB ID(s) of Target Chain'],
                    'Complex PDB': out['PDB ID(s) for Ligand-Target Complex'],
                    })
ooo = ooo[ooo['Target PDB'] != '']
ooo.to_csv(root+'bingdingDB.csv', index=False)

