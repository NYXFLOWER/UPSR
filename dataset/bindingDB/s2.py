# ######################### Columns Selection ############################
# import pandas as pd

# out = pd.read_csv("./BindingDB_All.tsv", sep='\t', error_bad_lines=False, nrows=2)
#
# ooo = pd.DataFrame({'PubChem CID': out['PubChem CID'],
#                     'Target Name': out['Target Name Assigned by Curator or DataSource'],
#                     'ki': out['Ki (nM)'],
#                     'kd': out['Kd (nM)'],
#                     'ic50': out['IC50 (nM)'],
#                     'Target PDB': out['PDB ID(s) of Target Chain'],
#                     'Complex PDB': out['PDB ID(s) for Ligand-Target Complex'],
#                     })
#
# ooo.to_csv('bdb_all.tsv', index=False, sep='\t')


# #############################################################################
# ######################### Parsing Web for protein name mapping ##############
# ###################### https://www.bindingdb.org/bind/BySequence.jsp ########
import pandas as pd
from bs4 import BeautifulSoup as bs

path = 'dataset/bindingDB/Sequence Search.html'

# empty list
data = []

# for getting the header from
# the HTML file
list_header = []
soup = bs(open(path), 'html.parser')
header = soup.find_all("table")[0].find("tr")

for items in header:
    try:
        list_header.append(items.get_text())
    except:
        continue

# for getting the data
HTML_data = soup.find_all("table")[0].find_all("tr")[1:]

for element in HTML_data:
    sub_data = []
    for sub_element in element:
        try:
            sub_data.append(sub_element.get_text())
        except:
            continue
    data.append(sub_data)

list_header = ['Sequence Name', 'Source Organism', 'KI', 'IC50', 'KD', 'EC50', 'Koff', 'Kon', 'other', 'ITC Data', 'Sequence']
dataFrame = pd.DataFrame(data=data[1:], columns=list_header)
dataFrame.to_csv('BindingDB_by_sequence.csv', index=False)

