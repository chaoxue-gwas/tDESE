# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/06/27 12:04
# @Function: Global Parameters for the Project.
import platform

# Common
DATA_DIR = '/home/xc/local/data'
if platform.system() == 'Windows':
    DATA_DIR = r'E:\WorkData\syncHPC\home\data'

# Input
## Real data and Simulate data
### age RNA-seq

### GWAS


### Age at onset
AGE_ONSET=f'{DATA_DIR}/project_data/timeDESE/age_at_onset/mol_psy.xlsx'


# Output
PROJ_NAME='timeDESE'
Result_ver='20250701-10' #-sex
# Result_ver='20250704'
PROJ_DIR=f'{DATA_DIR}/projects/{PROJ_NAME}/{Result_ver}'
## Real data and Simulate data
PROJ_REAL=f'{PROJ_DIR}/real'
PROJ_SIM=f'{PROJ_DIR}/simulate'
### age RNA-seq: include ind,predict;
REAL_RNA=f'{PROJ_REAL}/age_expr'
SIM_RNA=f'{PROJ_SIM}/age_expr'

### assoc-age: kggsum_para;phenotype
REAL_ASSOC=f'{PROJ_REAL}/assoc'
SIM_ASSOC=f'{PROJ_SIM}/assoc'

ENSG_annot=f'{DATA_DIR}/resources/GeneAnnotation/Human.GRCh38.ENSG2Name.txt'


