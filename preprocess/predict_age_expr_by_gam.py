# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/06/10 16:27
# @Function:
## install packages:  pip install tqdm joblib pygam pandas numpy
## main function: smooth_expression_by_age_parallel()
## api: smooth_expression_by_age_parallel_api()
import os

import pandas as pd
import numpy as np
import re
from pygam import GAM, s, LinearGAM, l, f
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from para import REAL_RNA
from util import unified_path, make_dir, log
import warnings
warnings.filterwarnings("ignore")

def extract_age(col):
    arr=str(col).split('.')
    if len(arr)<3:
        match = re.search(r'(\d+)\.', col)
        if match:
            return int(match.group(1))
        else:
            try:
                return int(col)
            except ValueError:
                raise ValueError(f"Cannot extract age from column name: {col}")
    else:
        return int(arr[1].split('_')[0])

def predict_with_se_0(gam, X_new):
    y_pred = gam.predict(X_new)
    intervals = gam.confidence_intervals(X_new, width=0.95)
    Z = 1.96
    se = (intervals[:, 1] - intervals[:, 0]) / (2 * Z)
    return y_pred, se

def fit_single_gene_0(gene_name, expr_values, ages, predicted_ages):
    try:
        gam = GAM(s(0)).fit(ages, expr_values)
        pred, se = predict_with_se(gam,predicted_ages)
    except:
        pred = np.full_like(predicted_ages, np.nan, dtype=float)
        se = np.full_like(predicted_ages, np.nan, dtype=float)
    return gene_name, pred, se

def predict_with_se(gam, X_new):
    pred = gam.partial_dependence(term=0, X=X_new, width=.95)
    y_pred = pred[0]
    intervals = pred[1]
    Z = 1.96
    se = (intervals[:, 1] - intervals[:, 0]) / (2 * Z)
    return y_pred, se

def fit_single_gene(gene_name, expr_values, covs, X_predict, cov_num=1):
    # try:
    if cov_num==3:
        gam = LinearGAM(s(0)+f(1)+l(2))
    if cov_num==2:
        gam = LinearGAM(s(0) + l(2))
    if cov_num==1:
        gam = LinearGAM(s(0))
    gam.fit(covs, expr_values)
    pred, se = predict_with_se(gam,X_predict)
    # except:
    #     pred = np.full_like(predicted_ages, np.nan, dtype=float)
    #     se = np.full_like(predicted_ages, np.nan, dtype=float)
    return gene_name, pred, se


def extract_covs_by_colnames(col_names):
    covs=[]
    for col in col_names:
        arr=str(col).split(';')
        covs.append([int(arr[i+1]) for i in range(2)]+[float(arr[5])])
    return np.array(covs)
def smooth_expression_by_age_parallel_api_with_se(input_path, n_jobs=6, cov_num=1):
    df = pd.read_csv(input_path, sep=None, engine='python')
    col_names = df.columns[1:]
    covs=extract_covs_by_colnames(col_names)
    ages = covs[:,0]
    min_age, max_age = min(ages), max(ages)
    age_range  = np.arange(min_age, max_age + 1)
    gene_names = df.iloc[:, 0].values
    expr_matrix = df.iloc[:, 1:].values.astype(float)
    sex_col = np.full_like(age_range, int(stats.mode(covs[:,1], keepdims=False).mode))
    rin_col = np.full_like(age_range, np.nanmean(covs[:,2]))
    init_covs = np.column_stack((age_range,sex_col,rin_col))
    X_predict = init_covs
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_gene)(
            gene_names[i],
            expr_matrix[i],
            covs,
            X_predict,
            cov_num
        )
        for i in tqdm(range(len(gene_names)), desc="Fitting GAM for each gene")
    )
    genes=[r[0] for r in results]
    pred_matrix = [r[1].tolist() for r in results]
    se_matrix = [r[2].tolist() for r in results]
    age_str=[f'{int(ag)}' for ag in age_range]
    return (genes, age_str, pred_matrix, se_matrix)

def smooth_expression_by_age_parallel_api(input_path, n_jobs=6):
    genes, ages, pred_matrix, se_matrix = smooth_expression_by_age_parallel_api_with_se(input_path,n_jobs)
    return (genes, ages, pred_matrix)

def smooth_expression_by_age_parallel(input_path, output_prefix, with_se=['se','mean'], n_jobs=6,cov_num=1):
    make_dir(os.path.dirname(output_prefix))
    genes, predicted_ages, pred_matrix, se_matrix = smooth_expression_by_age_parallel_api_with_se(input_path,n_jobs,cov_num)
    pred_matrix=np.array(pred_matrix)
    if len(predicted_ages) != pred_matrix.shape[1]:
        raise Exception(f'genes count is not equal to expr')
    if 'mean' in with_se:
        result_df = pd.DataFrame({'Gene': genes})
        for i, age in enumerate(predicted_ages):
            result_df[str(age)] = pred_matrix[:, i]
        result_df.to_csv(f'{output_prefix}-mean.tsv', sep='\t', index=False, lineterminator='\n', float_format='%.6f')
    if 'se' in with_se:
        se_matrix = np.array(se_matrix)
        result_df = pd.DataFrame({'Gene': genes})
        for i, age in enumerate(predicted_ages):
            result_df[f'{str(age)}.mean'] = pred_matrix[:, i]
            result_df[f'{str(age)}.SE'] = se_matrix[:, i]
        result_df.to_csv(f'{output_prefix}-se.tsv', sep='\t', index=False, lineterminator='\n', float_format='%.6f')

if __name__ == '__main__':
    # test_path=unified_path(f'project_data/timeDESE/gs_qc/ctf/ind/PsychEncode-LIBD_scControl.gene.ind.ctf.afterbirth.DFC.tsv')
    # input_path=unified_path(f'project_data/timeDESE/gs_qc/ctf/ind/PsychEncode-LIBD_scControl.tsv.gz')
    # output_path=unified_path(f'project_data/timeDESE/gs_qc/ctf/ind/t2.tsv')
    # # __reformat(test_path,input_path)
    # smooth_expression_by_age_parallel(input_path,output_path)
    ind_dir=f'{REAL_RNA}/ind'
    gam_dir=f'{REAL_RNA}/gam'
    for fn in os.listdir(ind_dir):
        log(f'start {fn}')
        for cov_num in range(1,4):
            f_prefix = '.'.join(fn.split('.')[:-1])+f'-cov{cov_num}'
            smooth_expression_by_age_parallel(f'{ind_dir}/{fn}',f'{gam_dir}/{f_prefix}',n_jobs=20, cov_num=cov_num)






