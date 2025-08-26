# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/18 21:08
# @Function:
import ast
import os
import re
from abc import ABC

import numpy as np
import pandas as pd
from rnanorm import TMM, CTF
from pycombat import Combat

from para import REAL_RNA
from util import unified_path, make_dir, log


class PreprocessGeneScore(ABC):
    def __init__(self):
        pass

    def qc_gs_sample(self, expr_df:pd.DataFrame, ages, min_neighbors=1, max_gap=3):
        neighbor_counts = np.array([
            np.sum(np.abs(ages - age_i) <= max_gap) - 1
            for age_i in ages
        ])
        keep_mask = neighbor_counts >= min_neighbors
        kept_expr_df = expr_df.loc[:, keep_mask]
        sample_names = expr_df.columns.to_numpy()
        removed_sample_names = sample_names[~keep_mask].tolist()
        log(f"(sample QC) remove {np.sum(~keep_mask)} isolate samples, remain {np.sum(keep_mask)} samples")
        if len(removed_sample_names)>0:
            log(f'removed sample names: {",".join(removed_sample_names)}')
        return kept_expr_df

    def qc_gs_gene(self,df:pd.DataFrame):
        raw_count=df.shape
        df = df.copy()
        df = df.map(lambda x: x if x >= 1e-4 else 0)
        df = df.map(lambda x: x if x <= 1e6 else 1e6)
        df = df.fillna(0)
        df = df.loc[(df > 0).sum(axis=1) >= 5]
        log(f'(gene QC) remain {df.shape} (raw {raw_count}).')
        return df

    def norm_sample(self,df:pd.DataFrame,method='ctf'):
        narr = None
        if method=='ctf':
            narr= CTF().fit_transform(df.values.T)
        if method=='tmm':
            tmm = TMM().fit(df.values.T)
            narr = tmm.transform(df.values.T)
        ndf = pd.DataFrame(narr.T, index=df.index, columns=df.columns)
        return ndf


class Brainspan(PreprocessGeneScore):
    def __init__(self,rna_dir='genes_matrix_csv',rna_type='gene'):
        self.gene_meta:pd.DataFrame = None
        self.cell_meta:pd.DataFrame = None
        self.data:pd.DataFrame = None
        self.wdir=unified_path(f'resources/Transcriptome/Brain/BrainSpan/{rna_dir}')
        self.output_dir = unified_path(f'project_data/timeDESE/gs_tpm')
        self.rna_type=rna_type
        make_dir(self.output_dir)

    def rpkm_to_tpm(self):
        df_rpkm=pd.read_csv(f'{self.wdir}/expression_matrix.csv',header=None,index_col=0)
        df_tpm = df_rpkm.div(df_rpkm.sum(axis=0), axis=1) * 1e6
        output_file=f'{self.wdir}/expression_matrix.tpm.csv'
        df_tpm.to_csv(output_file,header=False,float_format='%.6f')

    def load_data(self):
        self.data=pd.read_csv(f'{self.wdir}/expression_matrix.tpm.csv',header=None,index_col=0,dtype=str)
        self.cell_meta=pd.read_csv(f'{self.wdir}/columns_metadata.csv',index_col=0,dtype=str)
        # set 0-12 mos to 0 yrs
        self.cell_meta['age']=self.cell_meta['age'].map(lambda x:'0 yrs' if 'mos' in x else x)
        self.gene_meta=pd.read_csv(f'{self.wdir}/rows_metadata.csv',index_col=0,dtype=str)

    def after_birth_allregion(self):
        output_file=f'brainspan.{self.rna_type}.ind.afterbirth.allregion.tsv'
        self.load_data()
        filter_cell_idx=self.cell_meta.loc[self.cell_meta['age'].str.contains('yrs', na=False)].index
        col_names=self.cell_meta.loc[filter_cell_idx,['donor_id', 'age', 'gender','structure_acronym']].astype(str)\
            .agg('.'.join, axis=1).str.replace(r'\s+', '_', regex=True)
        df=self.data[filter_cell_idx-1]
        df.columns=col_names
        if self.rna_type=='exon':
            df.index= self.gene_meta.apply(lambda row: f"{row['start']}_{row['end']}:{row['ensembl_gene_id']}", axis=1)
        else:
            df.index=self.gene_meta['ensembl_gene_id']
        df.index.name='Gene'
        df.to_csv(f'{self.output_dir}/{output_file}',sep='\t',lineterminator='\n')
        pass

    def after_birth_splitregion(self):
        self.load_data()
        for region,region_df in self.cell_meta.groupby('structure_acronym'):
            output_file = f'brainspan.{self.rna_type}.ind.afterbirth.{region}.tsv'
            filter_cell_idx=region_df.loc[region_df['age'].str.contains('yrs', na=False)].index
            if len(filter_cell_idx)<5:
                log(f'Warning: {region} with < 5 samples, skip!')
                continue
            cols = region_df.loc[filter_cell_idx, ['donor_id', 'age', 'gender', 'structure_acronym']].astype(str).agg('.'.join, axis=1)
            col_names=[re.sub('\s+','_',c) for c in cols]
            df=self.data[filter_cell_idx-1]
            df.columns=col_names
            if self.rna_type == 'exon':
                df.index = self.gene_meta.apply(lambda row: f"{row['start']}_{row['end']}:{row['ensembl_gene_id']}", axis=1)
            else:
                df.index = self.gene_meta['ensembl_gene_id']
            df.index.name='Gene'
            df.to_csv(f'{self.output_dir}/{output_file}',sep='\t',lineterminator='\n')
            log(f'save {region} with {len(filter_cell_idx)} samples')
        pass


class PsychEncodeRNAseq(PreprocessGeneScore):
    def __init__(self,db_names:[str],diagnosis='Control',norm_methods=['ctf'],trans_methods=['log2'],remove_batch=False):
        self.remove_batch=remove_batch
        self.norm_methods=norm_methods
        self.trans_methods=trans_methods
        self.cell_meta:pd.DataFrame = None
        self.data:pd.DataFrame = None
        self.wdir=unified_path('resources/Transcriptome/Brain/PsychEncodeRNAseq/raw')
        self.output_dir=f'{REAL_RNA}/ind'
        self.db_names=db_names
        self.diagnosis=diagnosis
        make_dir(self.output_dir)

    def load_data(self,is_tpm):
        data_col_name='expected_count'
        if is_tpm:
            data_col_name='TPM'
        ind_meta=pd.read_csv(f'{self.wdir}/PECCapstoneCollection_ClinicalData.csv')
        sam_meta=pd.read_csv(f'{self.wdir}/PECCapstoneCollection_Tablesofsamples_RNAseq.csv')
        meta=pd.merge(sam_meta,ind_meta,left_on='individualID',right_on='individualID',how='left')
        raw_meta_count=meta.shape[0]
        ## filter meta
        # diagnosis
        meta = meta.loc[meta['diagnosis'] == self.diagnosis,]
        # DFC region
        meta=meta.loc[meta['tissue']=='dorsolateral prefrontal cortex',]
        # must contain sex
        meta=meta[meta['sex'].notna() & (meta['sex'] != '')]
        # must have age
        meta = meta[meta['ageDeath'].notna() & (meta['ageDeath'] != '')]
        # RIN > 7
        meta = meta[meta['RIN']>=6]
        # covert age to unified format (year)
        meta['ageDeath'] = meta['ageDeath'].astype(str).str.replace(r'\+$', '', regex=True)
        meta['ageDeath'] = pd.to_numeric(meta['ageDeath'], errors='coerce')
        meta = meta[meta['ageDeath'].notna()]
        meta = meta[meta['ageDeath'] >= 0]
        meta['ageDeath'] = meta['ageDeath'].astype(int)
        meta = meta.reset_index(drop=True)
        # summary info
        log(f'raw sample: {raw_meta_count}; after filter: {meta.shape[0]}')
        # load expr
        dfs = []
        sam_ids = []
        for db in self.db_names:
            log(f'start to load {db}')
            data_dir=f'{self.wdir}/{db}_genecounts'
            for f in os.listdir(data_dir):
                if not f.endswith('.RSEM_Quant.genes.results'):
                    continue
                if f not in meta['name'].values:
                    continue
                ## remain unique samples
                if f in sam_ids:
                    continue
                sdf=pd.read_table(f'{data_dir}/{f}',index_col=0)
                sdf=sdf[[data_col_name]]
                sam_ids.append(f)
                dfs.append(sdf)
        if len(dfs)<1:
            log(f'warning: {db} no suitable sample')
            return None
        else:
            df=pd.concat(dfs,axis=1)
            df.columns=sam_ids
            return df,meta

    def after_birth_oneregion_all(self):
        non_tpms=[]
        tpms=[]
        for m in self.norm_methods:
            if m=='tpm':
                tpms.append(m)
            else:
                non_tpms.append(m)
        if len(tpms)>0:
            self.after_birth_oneregion(True,tpms,self.trans_methods)
        if len(non_tpms)>0:
            self.after_birth_oneregion(False, non_tpms, self.trans_methods)

    def remove_batch_effect(self,df,meta):
        expr = df.copy()
        metadata = meta.copy()
        # metadata.index=metadata['name']
        metadata = metadata.loc[expr.columns]
        batch_cols=['batch']
        # for c in ['batch','platform','libraryPrep','readLength']:
        #     metadata[c] = metadata[c].fillna('unknown').astype('category').cat.codes
        #     if len(metadata[c].unique())>1:
        #         batch_cols.append(c)
        # for c in ['RIN']:
        #     metadata[c] = metadata[c].fillna(metadata[c].median())
        for c in ['sex']:
            metadata[c] = metadata[c].fillna('unknown').astype('category').cat.codes
        log(f'batch var: {batch_cols}')
        batch = metadata[batch_cols].values
        covars = metadata[['ageDeath','sex']].values #'RIN',
        combat = Combat()
        expr_corrected = combat.fit_transform(expr.T.values, batch, covars)
        correct_df = pd.DataFrame(expr_corrected.T,index=expr.index,columns=expr.columns)
        return correct_df

    def qc_two_tail_sample(self,expr_df, meta_df, name_col='name', age_col='ageDeath',
                               window_size=5, min_samples=5):

        sample_intersection = np.intersect1d(expr_df.columns, meta_df[name_col].values)
        meta_sub = meta_df[meta_df[name_col].isin(sample_intersection)].copy()
        meta_sub = meta_sub.sort_values(age_col).reset_index(drop=True)
        ages = meta_sub[age_col].values
        left_boundary = ages[0]
        for start in np.arange(ages[0], ages[-1] - window_size + 1, 1):
            count = np.sum((ages >= start) & (ages < start + window_size))
            if count >= min_samples:
                left_boundary = start
                break
        right_boundary = ages[-1]
        for end in np.arange(ages[-1] - window_size, ages[0], -1):
            count = np.sum((ages > end) & (ages <= end + window_size))
            if count >= min_samples:
                right_boundary = end + window_size
                break
        filtered_meta = meta_sub[(meta_sub[age_col] >= left_boundary) & (meta_sub[age_col] <= right_boundary)].copy()
        filtered_expr = expr_df.loc[:, filtered_meta[name_col]]
        print(f"raw age range: {ages[0]} ~ {ages[-1]}; sample size: {expr_df.shape[1]}")
        print(f"after qc: {left_boundary} ~ {right_boundary}; sample size: {filtered_expr.shape[1]}")
        return filtered_expr

    def after_birth_oneregion(self,is_tpm,norm_ms,trans_ms):
        df,meta=self.load_data(is_tpm)
        if df is None:
            log(f'no data')
            return
        ## QC sample: remove two-tail samples with low sample size.
        df=self.qc_two_tail_sample(df,meta)
        sam_ids=df.columns
        filter_cell_idx=[]
        for si in sam_ids:
            idx=meta.loc[meta['name']==si,].index[0]
            filter_cell_idx.append(idx)
        meta['age_code']=meta['ageDeath'].map(lambda x:f'{x}')
        meta['sex_code'] = meta['sex'].map({'F': 0, 'M': 1}).astype('Int64')
        meta['batch_code'], uniques = pd.factorize(meta['contributingStudy_x'])
        meta['plat_code'], uniques = pd.factorize(meta['platform'])
        meta=meta.loc[filter_cell_idx]
        cols = meta.loc[filter_cell_idx, ['individualID', 'age_code', 'sex_code', 'batch_code',
                                          'plat_code','RIN']].astype(str).agg(
            ';'.join, axis=1)
        col_names = [re.sub('\s+', '_', c) for c in cols]
        meta.index=col_names
        meta = meta.loc[~meta.index.duplicated(keep='first')]
        df.columns = col_names
        df.index=df.index.map(lambda x:str(x).split('.')[0])
        # remain one sample for replicated samples.
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        df.index.name = 'Gene'
        sorted_cols = sorted(df.columns, key=lambda col: int(col.split('.')[1].split('_')[0]))
        df=df[sorted_cols]
        ages=np.array([int(x.split('.')[1].split('_')[0]) for x in df.columns])
        df=self.qc_gs_sample(df,ages)
        df=self.qc_gs_gene(df)
        log(f'get {df.shape[1]} samples')
        for norm_method in norm_ms:
            for trans_method in trans_ms:
                copy_df=df.copy()
                if norm_method!='tpm':
                    copy_df=self.norm_sample(copy_df,norm_method)
                if trans_method=='log2':
                    copy_df = copy_df.map(lambda x: np.log2(x + 1))
                if self.remove_batch:
                    log(f'start to remove batch.')
                    copy_df = self.remove_batch_effect(copy_df, meta)
                remove_batch_str='batch'
                if self.remove_batch:
                    remove_batch_str='nobatch'
                ind_name=f'{remove_batch_str}-{norm_method}-{trans_method}'
                output_file=f'PsyEn-{"-".join(self.db_names)}.{ind_name}.tsv'
                copy_df = copy_df[sorted(copy_df.columns, key=lambda x: int(x.split(';')[1]))]
                copy_df.to_csv(f'{self.output_dir}/{output_file}', sep='\t', lineterminator='\n', float_format='%.6f')
                log(f'save {output_file}')


class CommonMindRNAseq(PreprocessGeneScore):
    def __init__(self,diagnosis='Control',norm_method='ctf',trans_method='log2'):
        self.norm_method=norm_method
        self.trans_method=trans_method
        self.cell_meta:pd.DataFrame = None
        self.data:pd.DataFrame = None
        self.wdir=unified_path('resources/Transcriptome/CommonMind/raw')
        self.output_dir=f'{REAL_RNA}/ind'
        self.diagnosis=diagnosis
        make_dir(self.output_dir)
        self.sex_col='Gender'
        self.age_col='Age_of_Death'
        self.ind_col='Individual_ID'
        self.rna_col='DLPFC_RNA_Sequencing_Sample_ID'
        self.dis_col='Dx'

    def load_data(self):
        ind_meta=pd.read_csv(f'{self.wdir}/meta/cmc_mssm-penn-pitt_clinical.csv')
        sam_meta=pd.read_csv(f'{self.wdir}/meta/cmc_mssm-penn-pitt_dlpfc_mrna-metadata.csv')
        meta=pd.merge(sam_meta,ind_meta,left_on='Individual_ID',right_on='Individual_ID',how='left',suffixes=('', '_drop'))
        raw_meta_count=meta.shape[0]
        ## filter meta
        # diagnosis
        if self.diagnosis is not None:
            meta = meta.loc[meta[self.dis_col] == self.diagnosis,]
        # DFC region
        # meta=meta.loc[meta['tissue']=='dorsolateral prefrontal cortex',]
        # must contain sex
        meta=meta[meta[self.sex_col].notna() & (meta[self.sex_col] != '')]
        # must have age
        meta = meta[meta[self.age_col].notna() & (meta[self.age_col] != '')]
        # covert age to unified format (year)
        meta[self.age_col] = pd.to_numeric(meta[self.age_col], errors='coerce')
        meta = meta[meta[self.age_col].notna()]
        meta = meta[meta[self.age_col] >= 0]
        meta[self.age_col] = meta[self.age_col].astype(int)
        meta = meta.reset_index(drop=True)
        self.cell_meta = meta
        # summary info
        log(f'raw sample: {raw_meta_count}; after filter: {meta.shape[0]}')
        # load expr
        count_path = f'{self.wdir}/RNA_seq/cmc_mssm-penn-pitt_dlpfc_mrna_illuminahiseq2500_geneexpressionraw.tsv.gz'
        df = pd.read_table(count_path,index_col=0)
        df = df[meta[self.rna_col]]
        self.data=df

    def after_birth_oneregion(self):
        ind_name=f'{self.norm_method}-{self.trans_method}'
        output_file=f'CMC.{ind_name}.tsv'
        self.load_data()
        if self.data is None:
            log(f'no data')
            return
        sam_ids=self.data.columns
        meta=self.cell_meta
        filter_cell_idx=[]
        for si in sam_ids:
            idx=meta.loc[meta[self.rna_col]==si,].index[0]
            filter_cell_idx.append(idx)
        meta['region']='DFC'
        meta['age']=meta[self.age_col].map(lambda x:f'{x}_yrs')
        sex_abbr={'Male':'M','Female':'F'}
        meta['sex']=meta[self.sex_col].map(lambda x:sex_abbr[x])
        cols = meta.loc[filter_cell_idx, [self.ind_col, 'age', 'sex', 'region']].astype(str).agg(
            '.'.join, axis=1)
        col_names = [re.sub('\s+', '_', c) for c in cols]
        df=self.data
        df.columns = col_names
        df.index=df.index.map(lambda x:str(x).split('.')[0])
        # remain one sample for replicated samples.
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        df.index.name = 'Gene'
        sorted_cols = sorted(df.columns, key=lambda col: int(col.split('.')[1].split('_')[0]))
        df=df[sorted_cols]
        ages=np.array([int(x.split('.')[1].split('_')[0]) for x in df.columns])
        df=self.qc_gs_sample(df,ages)
        df=self.qc_gs_gene(df)
        df=self.norm_sample(df,self.norm_method)
        if self.trans_method=='log2':
            df = df.map(lambda x: np.log2(x + 1))
        df.to_csv(f'{self.output_dir}/{output_file}', sep='\t', lineterminator='\n', float_format='%.6f')
        log(f'save with {df.shape[1]} samples')



if __name__ == '__main__':
    dbs=[['CMC'],['LIBD_szControl']]
    norms=['tmm'] #['tpm','tmm','ctf']
    trans=['raw'] #['log2','raw']
    for db in dbs:
        PsychEncodeRNAseq([db],norm_methods=norms,trans_methods=trans).after_birth_oneregion_all()
