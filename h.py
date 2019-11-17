
import sys
import logging
from ssbio.pipeline.gempro import GEMPRO

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # SET YOUR LOGGING LEVEL HERE #

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
handler.setFormatter(formatter)
logger.handlers = [handler]

# SET FOLDERS AND DATA HERE
import tempfile
ROOT_DIR = tempfile.gettempdir()

PROJECT = 'genes_and_sequences_GP'
GENES_AND_SEQUENCES = {'808': 'MSLTDSFTVRSIEGVCFRYPLATPVVTSFGKMLNRPAVFVRVVDEDGVEGWGEAWSNFPAPGAEHRARLINEVLAPGLVGRKLENPAQAFEVLSKGTEVLALQCGEPGPFAQAISGIDLALWDLFARRRNLPLWRLLGGQSSKIKVYASGINPGGAAQTAEAALKRGHRALKLKVGFGAETDIANLSALLTIVGAGMLAADANQGWSVDQALEMLPRLSEFNLRWLEEPIRADRPREEWRKLRANAKMPIAAGENISSVEDFEAALGDDVLGVIQPDIAKWGGLTVCVELARQILRVGKTFCPHYLGGGIGLLASAHLLAAVGRDGWLEVDANDNPLRDLFCGPVADVREGTIELNQNPGLGIVPDLSAIERYRSIEGHHHHHH'}
PDB_FILE_TYPE = 'mmtf'

# Create the GEM-PRO project
my_gempro = GEMPRO(gem_name=PROJECT, root_dir=ROOT_DIR, genes_and_sequences=GENES_AND_SEQUENCES, pdb_file_type=PDB_FILE_TYPE)


# Mapping using BLAST
my_gempro.blast_seqs_to_pdb(all_genes=True, evalue=0.00000000000000000000000000000000000000001)
my_gempro.df_pdb_blast
