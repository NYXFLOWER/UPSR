import pandas as pd

out = pd.read_csv("./BindingDB_All.tsv", sep='\t', error_bad_lines=False, nrows=2)

ooo = pd.DataFrame({'PubChem CID': out['PubChem CID'],
                    'Target Name': out['Target Name Assigned by Curator or DataSource'],
                    'ki': out['Ki (nM)'],
                    'kd': out['Kd (nM)'],
                    'ic50': out['IC50 (nM)'],
                    'Target PDB': out['PDB ID(s) of Target Chain'],
                    'Complex PDB': out['PDB ID(s) for Ligand-Target Complex'],
                    })

ooo.to_csv('bdb_all.tsv', index=False, sep='\t')


