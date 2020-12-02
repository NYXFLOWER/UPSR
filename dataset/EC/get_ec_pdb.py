import requests
import re
import os
from bs4 import BeautifulSoup as bs
import pickle

ec3 = [3, 4, 24]
n = 89


def get_4th_ec(ec4_code):
    pdb_id_list = []

    page = "https://www.ebi.ac.uk/thornton-srv/databases/cgi-bin/enzymes/GetPage.pl?ec_number=" + code
    out_path = "tmp.html"

    # download from EC database -- the 4th level
    response = requests.get(page)
    if response.status_code == 200:
        response = response.text
    else:
        return pdb_id_list

    re_result = re.search(r'There are <B>(\d+)</B>', response)
    if re_result:
        n_pdb = int(re_result.group(1))
        response = response[re_result.regs[0][0]:]
        with open("tmp.html", 'w') as f:
            f.write(response)

        # parsing the pdb table from the download HTML file
        data = parse_html_table(out_path)
        data = [i[0] for i in data if len(i) > 1]
        for pdb_id in data:
            re_result = re.match(r'\n(\w{4})\n', pdb_id)
            if re_result:
                pdb_id_list.append(re_result.group(1))

        if len(pdb_id_list) != n_pdb:
            print("{} -- Not all pdb code in the list".format(ec4_code))
        else:
            print("{} -- {} PDB entries".format(ec4_code, len(pdb_id_list)))

    return pdb_id_list


def parse_html_table(path):
    list_header = []
    data = []
    soup = bs(open(path), 'html.parser')
    header = soup.find_all("table")[0].find("tr")

    for items in header:
        try:
            list_header.append(items.get_text())
        except:
            continue

    html_data = soup.find_all("table")[0].find_all("tr")[1:]

    for element in html_data:
        sub_data = []
        for sub_element in element:
            try:
                sub_data.append(sub_element.get_text())
            except:
                continue
        data.append(sub_data)

    return data


ec_pdb = {}
for j in range(1, n+1):
    ec4 = ec3 + [j]
    code = str(ec4).replace(', ', '.')[1: -1]

    pdb_list = get_4th_ec(code)
    if pdb_list:
        ec_pdb[code] = pdb_list

out_dir = 'ec_data/ec_{}/ec_{}_{}'.format(ec3[0], ec3[0], ec3[1])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_name = 'ec_pdb_{}_{}_{}.pkl'.format(ec3[0], ec3[1], ec3[2])


with open(os.path.join(out_dir, out_name), 'wb') as file:
    pickle.dump(ec_pdb, file)
