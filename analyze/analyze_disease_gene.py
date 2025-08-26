# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/05/26 10:42
# @Function: Analysis of genes highlighted specific age-specific effects.
import os
import re
import time

import numpy as np
import pandas as pd
import requests
import xml.dom.minidom as xmldom

import seaborn
from matplotlib import pyplot as plt

from para import REAL_ASSOC, REAL_RNA, AGE_ONSET
from util import log, get_gene_alias, LOCAL_DIR, read_line, make_dir, unified_path, kggsee_rez, kggsee_rez_last, \
    extract_col, plot_high_risk_age_range


class NCBI:
    def __init__(self):
        self.sess=requests.session()
        pass
    def request_ncbi(self,url):
        res=self.sess.post(url)
        xobj=xmldom.parseString(res.text)
        count=xobj.documentElement.getElementsByTagName("Count")[0].firstChild.data
        pmids=[]
        for pid in xobj.documentElement.getElementsByTagName("Id"):
            pmids.append(str(pid.firstChild.data).strip())
        return count,pmids

    def single_trait_gene(self,traits,genes):
        geneTerm='+OR+'.join([f'({gene}[tiab])' for gene in genes])
        traitTerm='+OR+'.join([f'({t}[tiab])' for t in traits])
        base_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
        query=f'db=pubmed&term=(({traitTerm})+AND+({geneTerm})+AND+(gene[tiab]+OR+genes[tiab]+OR+mRNA[tiab]+OR+protein[tiab]+OR+proteins[tiab]+OR+transcription[tiab]+OR+transcript[tiab]+OR+transcripts[tiab]+OR+expressed[tiab]+OR+expression[tiab]+OR+expressions[tiab]+OR+locus[tiab]+OR+loci[tiab]+OR+SNP[tiab]))&datetype=edat&retmax=100'
        url=f'{base_url}{query}'
        count=0
        pmids=[]
        while True:
            try:
                count,pmids=self.request_ncbi(url)
                break
            except:
                log(f'except! waiting for retrying ...')
                time.sleep(30)
                continue
        return str(count).strip(),pmids

    def batch_trait_gene(self,traits,genes,out_path):
        alias_genes=get_gene_alias(genes)
        if alias_genes is None:
            return None
        gene_alias_map={}
        for i in range(len(alias_genes)):
            gene_alias_map[genes[i]]=alias_genes[i]
        first = False
        egenes=[]
        if os.path.isfile(out_path):
            try:
                df=pd.read_table(out_path,header=0)
                egenes = list(df['gene'])
            except:
                egenes=[]
        else:
            first=True
        search_genes=[]
        for g in genes:
            if g not in egenes:
                search_genes.append(g)
        log(f'all: {len(genes)}, exist: {len(egenes)}, remain: {len(search_genes)}')
        with open(out_path,'a') as bw:
            if first:
                bw.write('\t'.join(['gene','count','pmids'])+'\n')
            i=0
            for g in search_genes:
                i+=1
                c,pids=self.single_trait_gene(traits,gene_alias_map[g])
                bw.write('\t'.join([g,c,','.join(pids)])+'\n')
                log(f'({i}/{len(search_genes)}) {g}: {c}')
                time.sleep(2)

class KGGSUM:
    def __init__(self, prefix):
        self.prefix = prefix

    def gene_based_p_cutoff(self):
        p_value = None
        for line in read_line(f'{self.prefix}/log/info.log'):
            if 'genes/regions are collected according to the p-value threshold' in line:
                match = re.search(
                    r"genes/regions are collected according to the p-value threshold ([\d\.]+[eE][-+]?\d+) for conditional gene based test",
                    line
                )
                if match:
                    p_value = float(match.group(1))
                else:
                    print("No p-value found.")
        return p_value

    def ecs_assoc_genes(self,gwas_file_name,gene_score_name='',cond=True):
        Pcut = 2
        if cond:
            Pcut = self.gene_based_p_cutoff()
        gp = f'{self.prefix}/{gwas_file_name}/GeneBasedConditionalAssociationTask/{gene_score_name}.genes.hg38.condi.assoc.tsv'
        genes = []
        if os.path.isfile(gp):
            df = pd.read_table(gp)
            df = df.loc[df['Condi.ECS.P'] < Pcut, :]
            df = df.sort_values(by='Condi.ECS.P', ascending=True)
            genes = list(df.loc[:, 'RegionID'].values)
        return genes

    def ecs_assoc_genes_df(self,gwas_file_name,gene_score_name='',cond=True):
        Pcut = 2
        if cond:
            Pcut = self.gene_based_p_cutoff()
        gp = f'{self.prefix}/{gwas_file_name}/GeneBasedConditionalAssociationTask/{gene_score_name}.genes.hg38.condi.assoc.tsv'
        df = None
        if os.path.isfile(gp):
            df = pd.read_table(gp)
            df = df.loc[df['Condi.ECS.P'] < Pcut, :]
            df = df.sort_values(by='Condi.ECS.P', ascending=True)
        return df


def calc_auc(xs, ys):
    auc = 0.0
    for i in range(1, len(xs)):
        width = xs[i] - xs[i - 1]
        height = (ys[i] + ys[i - 1]) / 2
        auc += width * height
    return auc


class GoldPhenoGene:
    def __init__(self,gene_based_dir=None):
        # db_name: pubmed, open_targets.
        if gene_based_dir is None:
            gene_based_dir=unified_path(f'resources/GWAS/gold_case/gene_based')
        self.gene_based_dir=gene_based_dir

    def pubmed_gold_genes(self, phenos:[], lib_cutoff=5, db_name='pubmed'):
        gold_arrs=[]
        for pheno in phenos:
            df=pd.read_table(f'{self.gene_based_dir}/{db_name}/{pheno}.tsv')
            gene_list = [f'{pheno}:' + gene for gene in df.loc[df['count'] >= lib_cutoff, 'gene']]
            gold_arrs+=gene_list
        return sorted(set(gold_arrs))

    def opentargets_gold_genes(self, phenos:[], filter_map={'chembl':0,'globalScore':0.4}, db_name='open_targets'):
        # {'chembl':0,'eva':0,'globalScore':0.4} {'globalScore':0.3}
        gold_arrs=[]
        for pheno in phenos:
            df = pd.read_csv(f'{self.gene_based_dir}/{db_name}/{pheno}.tsv', sep='\t', index_col=0, na_values='No data')
            keep = pd.Series(False, index=df.index)
            for col, threshold in filter_map.items():
                keep |= (df[col] > threshold)
            gene_list = [f'{pheno}:{x}' for x in df.index[keep].tolist()]
            gold_arrs+=gene_list
        return sorted(set(gold_arrs))


    def getTPRandFPR(self,trueGenes, falseGenes, predTrueGenes, predFalseGenes):
        TP = len(set(predTrueGenes).intersection(set(trueGenes)))
        TN = len(set(predFalseGenes).intersection(set(falseGenes)))
        FP = len(set(predTrueGenes).intersection(set(falseGenes)))
        FN = len(set(predFalseGenes).intersection(set(trueGenes)))
        TPR, FPR = 0, 0
        if TP + FN != 0:
            TPR = TP / (TP + FN)
        if FP + TN != 0:
            FPR = FP / (FP + TN)
        return TPR, FPR, TP, len(predTrueGenes)

    def access_db(self,phenotypes:[],out_tsv,genes):
        '''
        :return:
        '''
        support_dbs=['pubmed']
        if self.db_name not in support_dbs:
            raise Exception(f'Database: {self.db_name} is not supported! Supported list: {", ".join(support_dbs)}')
        make_dir(os.path.dirname(out_tsv))
        if self.db_name=='pubmed':
            NCBI().batch_trait_gene(phenotypes, genes, out_tsv)

    def eval_ROC(self, sort_genes_arr, tags, gold_genes, plot_step=1, print_info=False,plot_fig=True):
        tag_auc = {}
        roc_vals = {}
        pp_val = {}
        k=-1
        for sgs in sort_genes_arr:
            k+=1
            true_genes=set(sgs).intersection(gold_genes)
            false_genes=set([g for g in sgs if g not in true_genes])
            tag = tags[k]
            roc_vals[tag] = []
            for i in np.arange(0,len(sgs),plot_step):
                pred_true_genes = sgs[:i]
                pred_false_genes = sgs[i:]
                TPR, FPR, TP, PP = self.getTPRandFPR(true_genes, false_genes, pred_true_genes, pred_false_genes)
                roc_vals[tag].append([TPR, FPR])
            PP = len(sgs)
            TP = len(true_genes)
            pp_val[tag] = [PP, TP]
        kc=-1
        xcor= {}
        ycor={}
        for k in tags:
            kc+=1
            xs = [z[1] for z in roc_vals[k]]
            ys = [z[0] for z in roc_vals[k]]
            auc = calc_auc(xs,ys)
            if print_info:
                print(f'{k}: predict: {pp_val[k][0]}; TP: {pp_val[k][1]}；TPR: {pp_val[k][1] / pp_val[k][0]:.3f}; AUC: {auc:.3f}')
            tag_auc[k] = auc
            xcor[k]=xs
            ycor[k]=ys
        fig=None
        if plot_fig:
            fig, axe = plt.subplots(figsize=(4, 4))
            for kc in range(len(tags)):
                k=tags[kc]
                auc=tag_auc[k]
                show_k=k
                if k=='P-value':
                    show_k=r'$\mathit{p}$-value'
                axe.plot(xcor[k], ycor[k], label=f'{show_k}, AUC={auc:.3f}')#,c=colors[kc]
            axe.set_xlabel('FPR')
            axe.set_ylabel('TPR')
            axe.set_title('')
            axe.spines['top'].set_visible(False)
            axe.spines['right'].set_visible(False)
            plt.legend(loc='lower right')
            # plt.show()
            plt.tight_layout()
        return fig, tag_auc
        # return tag_auc

def access_pubmed():
    # get disease-genes in PubMed.



    pass


import pandas as pd


def re_rank_genes_by_expression_and_disease(gene_list, expr_df):
    disease_rank = pd.Series(range(len(gene_list)), index=gene_list)
    common_genes = expr_df.index.intersection(disease_rank.index)
    disease_rank = disease_rank.loc[common_genes]
    log(f'raw gene: {len(gene_list)}; remain: {len(disease_rank)}')
    expr_df = expr_df.loc[common_genes]
    age_gene_rank_dict = {}
    # expr_df['mean']=expr_df.mean(axis=1)
    for age in expr_df.columns:
        expr_series = expr_df[age]
        expr_rank = expr_series.rank(ascending=False, method='average')
        # combined_rank = (disease_rank + expr_rank) / 2
        combined_rank = expr_rank
        sorted_genes = combined_rank.sort_values().index.tolist()
        age_gene_rank_dict[age] = sorted_genes
    return age_gene_rank_dict


def run_rez(key='batch-tmm-raw-cov3-se'):
    expr_dir=f'{REAL_RNA}/gam'
    kggsee_dir = '/home/xc/local/program/python/pDESE/paper/lib/kggsee'
    kggsee_jar = f'{kggsee_dir}/kggsee.jar'
    resource_dir = f'{kggsee_dir}/resources'
    for f in os.listdir(expr_dir):
        if key not in f:
            continue
        cmd=kggsee_rez(f'{expr_dir}/{f}',f'{expr_dir}_rez/{f}',3,kggsee_jar,resource_dir)
        log(cmd)
        os.system(cmd)


def eval_age_prior_genes(gene_score_name,gwas_names, tool_name = 'DESE',
    tag = 'bhfdr_1eN2',out_tag='1'):
    ## read tissue specific genes
    rez_score_path=f'{REAL_RNA}/gam_rez/{gene_score_name}.REZ.webapp.genes.txt.gz'
    expr_df = pd.read_table(rez_score_path,index_col=0)
    exdf=extract_col(expr_df,'.z')
    result_dir = f'{REAL_ASSOC}/{tool_name}.run.{tag}'
    plot_dir=f'{REAL_ASSOC}/{tool_name}.analyze.{tag}/evaluate_assoc_gene'
    make_dir(plot_dir)
    n_cols=1
    n_rows=len(gwas_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 1.8 * n_rows), constrained_layout=True,sharex=True)
    axes = axes.flatten()
    db_name=gene_score_name.split('.')[0].split('-')[1]
    auc_tab=f'{plot_dir}/{db_name}.{out_tag}.xlsx'
    # read age at onset.
    age_df = pd.read_excel(AGE_ONSET)
    age_ranges = {'AD': [65, 100]}
    for i in age_df.index:
        age_ranges[age_df.loc[i, 'Abbr']] = [age_df.loc[i, 'p25'], age_df.loc[i, 'p75']]
    i=-1
    with pd.ExcelWriter(auc_tab) as writer:
        for pheno in gwas_names:
            i+=1
            ax=axes[i]
            gwas_file=f'{pheno}.gwas.sum.tsv.gz'
            kgs=KGGSUM(f'{result_dir}/{pheno}')
            gs=kgs.ecs_assoc_genes(gwas_file,gene_score_name,cond=False)
            age_gs=re_rank_genes_by_expression_and_disease(gs,exdf)
            ages=sorted(age_gs.keys(),key=lambda x:int(x))
            pgs=[[f'{pheno}:{x}' for x in age_gs[a]] for a in ages]
            gpg=GoldPhenoGene()
            gold_genes=gpg.opentargets_gold_genes([pheno])
            log(f'{pheno}:{len(gold_genes)} gold genes')
            # gold_genes=gpg.pubmed_gold_genes([pheno])
            _,tag_auc=gpg.eval_ROC(pgs,ages,gold_genes,print_info=False,plot_fig=False)
            ## plot
            df=pd.DataFrame({'age':[int(a) for a in ages],'auc':[tag_auc[a] for a in ages]})
            df.to_excel(writer,pheno,index=False)
            age_col='age'
            score_col='auc'
            ag = None
            if pheno in age_ranges:
                ag = age_ranges[pheno]
            plot_high_risk_age_range(pheno,df,ag,ax,age_col,score_col)
        for j in range(len(gwas_names), len(axes)):
            axes[j].axis('off')
        fig.suptitle(db_name,fontsize=18)
        plt.savefig(f'{plot_dir}/{db_name}.{out_tag}.jpg',dpi=300)
        plt.show()
        plt.close()
        pass


if __name__ == '__main__':
    # run_rez()
    gwas_names = ['ADHD','SCZ','MDD','BIP','AD']
    expr_dir=f'{REAL_RNA}/gam'
    for gsn in os.listdir(expr_dir):
        if 'batch-tmm-raw-cov3-se.tsv' in gsn:
            # eval_age_prior_genes(gsn, gwas_names,out_tag='2')
            eval_age_prior_genes(gsn, gwas_names,out_tag='2')