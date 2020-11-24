import requests


def inchikey_to_smile(inchikey):
    r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON').json()
    smiles = r['PropertyTable']['Properties'][0]['CanonicalSMILES']
    return smiles

