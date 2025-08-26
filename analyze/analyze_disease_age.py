# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/19 23:25
# @Function:
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from tqdm import tqdm

from analyze.analyze_disease_gene import KGGSUM
from para import REAL_ASSOC, AGE_ONSET, REAL_RNA, ENSG_annot
from preprocess.predict_age_expr_by_gam import extract_age
from util import unified_path, make_dir, log, go_enrich_plot, heatmap_color_rstyle

import pandas as pd


def get_range_by_median(x):
    right_min, right_max = np.min(x), np.max(x)
    right_mid = np.nanmedian(x)
    right_range = max([right_max - right_mid, right_mid - right_min])*1.1
    return (right_mid - right_range, right_mid + right_range)

def get_range_by_ci_median(x,low,high):
    right_min, right_max = np.min(low), np.max(high)
    right_mid = np.nanmedian(x)
    right_range = max([right_max - right_mid, right_mid - right_min])*1.1
    return (right_mid - right_range, right_mid + right_range)

def replace_gene_symbol(df:pd.DataFrame):
    meta_df=pd.read_table(ENSG_annot)
    mapping_dict={meta_df.loc[i,'Gene stable ID']:meta_df.loc[i,'Gene name'] for i in meta_df.index}
    new_index = df.index.map(mapping_dict)
    df_new = df.copy()
    df_new.index = new_index
    df_new = df_new[df_new.index.notna()]
    return df_new

def extract_mean_expression(df):
    cols = df.columns
    if any(col.endswith('.mean') for col in cols):
        mean_cols = [col for col in cols if col.endswith('.mean')]
        new_cols = [col.replace('.mean', '') for col in mean_cols]
        df_mean = df[mean_cols].copy()
        df_mean.columns = new_cols
        return df_mean
    else:
        return df.copy()


def extract_mean_and_ci_expression(df):
    mean_df = df.filter(regex=r'\.mean$')
    se_df = df.filter(regex=r'\.SE$')
    upper_df = mean_df + 1.96 * se_df.values
    lower_df = mean_df - 1.96 * se_df.values
    mean_df.columns = [col.replace('.mean', '') for col in mean_df.columns]
    upper_df.columns = mean_df.columns
    lower_df.columns = mean_df.columns
    return mean_df, upper_df, lower_df


def gene_disease_correlation(gam_df, cell_df):
    gam_df.columns=gam_df.columns.astype(str)
    cell_df['age']=cell_df['age'].astype(str)
    gam_ages = gam_df.columns
    cell_ages = cell_df['age']
    common_ages = gam_ages.intersection(cell_ages)
    if len(common_ages) == 0:
        raise ValueError("no common age")
    expr = gam_df[common_ages]
    score = cell_df.set_index('age').loc[common_ages, 'score'].astype(float)
    results = []
    for gene, row in tqdm(expr.iterrows(), total=expr.shape[0], desc="Calculating Spearman"):
        corr, p = spearmanr(row.values, score.values)
        results.append({'gene': gene, 'spearman_corr': corr,'spearman_p': p})
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('spearman_corr', ascending=False).reset_index(drop=True)
    return result_df


class AnalyzeDiseaseCell():
    def __init__(self,tool_name,tag,include_gwass,output_dir,key_word,gene_score_keys):
        self.gene_score_name = None
        self.tool_name=tool_name
        self.analyze_dir = f'{output_dir}/{tool_name}.analyze.{tag}'
        self.result_dir = f'{output_dir}/{tool_name}.run.{tag}'
        if include_gwass is None:
            include_gwass = [f for f in os.listdir(self.result_dir) if os.path.isdir(f'{self.result_dir}/{f}')]
        self.gwas_name = include_gwass
        make_dir(self.analyze_dir)
        self.__load_gs_names(key_word,gene_score_keys)

    def __phase_age(self,col:str) -> int:
        col=str(col)
        arr=col.split('.')
        if len(arr)==1:
            # age=int(arr[0].split('_')[0])
            numbers = re.findall(r'\d+', arr[0])
            age=int(numbers[0])
        if len(arr)>1:
            age=int(arr[1].split('_')[0])
        return age

    def __load_gs_names(self,key_word,gene_score_keys):
        if self.tool_name=='DESE':
            gn = self.gwas_name[0]
            gss = []
            for f in os.listdir(f'{self.result_dir}/{gn}/{gn}.gwas.sum.tsv.gz/GeneBasedConditionalAssociationTask'):
                if f.endswith('.enrichment.tsv'):
                    if key_word not in f:
                        continue
                    if any(f.split('.')[0]==sub for sub in gene_score_keys):
                        gs_tag=f.split('.enrichment.tsv')[0]
                        gss.append(gs_tag)
            self.gene_score_name = sorted(gss)
        if self.tool_name=='LDSC':
            gss = set()
            for f in os.listdir(f'{self.result_dir}'):
                if f.endswith('.cell_type_results.txt'):
                    if key_word not in f:
                        continue
            self.gene_score_name = sorted(gss)

    def __get_disease_score(self,gsn,gn):
        if self.tool_name=='DESE':
            disease_cell_path = f'{self.result_dir}/{gn}/{gn}.gwas.sum.tsv.gz/GeneBasedConditionalAssociationTask/{gsn}.enrichment.tsv'
            df = pd.read_table(disease_cell_path, skipfooter=1)
            df['age'] = df['Condition'].map(lambda x: self.__phase_age(x))
            df['score'] = df['EnrichmentScore']
            df['significant'] = df['Adjusted(p)'].apply(lambda x: 'Sig.' if x < 0.05 else 'No. Sig.')
            return df
        if self.tool_name=='LDSC':
            disease_cell_path = f'{self.result_dir}/{gn}.{gsn}.gene.cell_type_results.txt'
            df = pd.read_table(disease_cell_path)
            df['age'] = df['Name'].map(lambda x: self.__phase_age(x))
            df['score'] = -np.log10(df['Coefficient_P_value'].values)
            df['bonf_p'] = df['Coefficient_P_value'].values*df.shape[0]
            df['significant'] = df['bonf_p'].apply(lambda x: 'Sig.' if x < 0.05 else 'No. Sig.')
            return df

    def __get_gene_score(self,gsn,gn):
        if self.tool_name=='DESE':
            kgs = KGGSUM(f'{self.result_dir}/{gn}')
            gs = kgs.ecs_assoc_genes_df(f'{gn}.gwas.sum.tsv.gz', gsn, cond=True)
            gs=gs[['RegionID','Condi.ECS.P']]
            gs.columns=['gene','gene_p']
            return gs

    def merge_intervals(self,intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            if current[0] <= prev[1]:
                prev[1] = max(prev[1], current[1])
            else:
                merged.append(current)
        return merged
    def plot_high_risk_age_range(self,title,df,age_range, ax, age_col='age', score_col='score',sig_col='significant'):
        line_color="#BDBDBD"
        sig_point_color="#D62728"
        age_range_color="#FFA500"
        df = df.dropna().copy()
        df = df.sort_values(by=age_col)
        sns.set_context("paper", font_scale=1.4)
        sns.set_style("white")
        show_legend=True
        if age_range is None:
            show_legend=False
        sns.lineplot(x=age_col, y=score_col, data=df, ax=ax, color=line_color, linewidth=3, zorder=1)
        sns.scatterplot(data=df, x=age_col, y=score_col, hue=sig_col,
                        palette={'Sig.': sig_point_color, 'No. Sig.': line_color},
                        ax=ax, s=30, edgecolor='black', alpha=0.8, legend=False, zorder=2)

        spans=[]
        for i in df.index:
            if df.loc[i,sig_col]=='Sig.':
                age=df.loc[i,age_col]
                spans.append([age-0.5,age+0.5])
        if age_range is not None:
            st = age_range[0]
            en = age_range[1]
            ax.axvspan(
                st,
                en,
                color=age_range_color,
                alpha=0.2,
                label=f'Diagnostic: {st}–{en}',
            )
        ax.set_xlabel("Age (years)", fontsize=12)
        ax.set_ylabel("Enrichment score", fontsize=12)
        ax.set_xlim(-5, 100)
        ax.legend(frameon=False, fontsize=11,)
        ax.grid(False)
        ax.tick_params(length=4, width=1)
        ax.set_title(f"{title}", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

    def plot_age_risk_multi_phenotypes(self,age_at_onset_path):
        age_risk_plot_dir=f'{self.analyze_dir}/age_risk_multi_pheno'
        make_dir(age_risk_plot_dir)
        # read age at onset.
        age_df=pd.read_excel(age_at_onset_path)
        age_ranges={'AD':[65,100]}
        for i in age_df.index:
            age_ranges[age_df.loc[i,'Abbr']]=[age_df.loc[i,'p25'],age_df.loc[i,'p75']]
        for gsn in self.gene_score_name:
            assoc_age_table=f"{age_risk_plot_dir}/age_risk_pheno.scatter.{gsn}.xlsx"
            with pd.ExcelWriter(assoc_age_table) as writer:
                n_cols=1
                n_rows=len(self.gwas_name)
                sns.set(style="whitegrid", font="Arial", rc={"axes.edgecolor": "0.2", "grid.color": "#E0E0E0"})
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.1 * n_rows), constrained_layout=True,sharex=True)
                axes = axes.flatten()
                for i,gn in enumerate(self.gwas_name):
                    df=self.__get_disease_score(gsn,gn)
                    ax = axes[i]
                    ag=None
                    if gn in age_ranges:
                        ag=age_ranges[gn]
                    self.plot_high_risk_age_range(f'{gn}', df, ag, ax)
                    cp_df=df.copy()
                    cp_df=cp_df[['Condition','EnrichmentScore','Adjusted(p)']]
                    cp_df.columns=['Age (years)','Enrichment score','Adjusted P']
                    cp_df.to_excel(writer, sheet_name=gn, index=False)
                for j in range(len(self.gwas_name), len(axes)):
                    axes[j].axis('off')
                db_name = gsn.split(".")[0].split('-')[-1]
                plt.suptitle(f"{db_name}", fontsize=20)
                plt.tight_layout(rect=[0, 0, 1, 1])
                plt.show()
                fig.savefig(f"{age_risk_plot_dir}/age_risk_pheno.scatter.{gsn}.jpg", dpi=300, bbox_inches='tight')
                plt.close()


    def plot_gene_age_tend(self,title,df_raw,df_fit,age_range,ax):
        age_range_color="#FFA500"
        ax.plot(df_fit['age'], df_fit['expr'],
                color='#1f77b4',
                linewidth=1.5,
                label='Fitted Mean')
        ax.fill_between(df_fit['age'], df_fit['low'], df_fit['high'],
                        color='#1f77b4', alpha=0.2, label='95% CI')
        ax2 = ax.twinx()
        ax2.scatter(df_raw['age'], df_raw['expr'],
                   color='#333333',
                   s=10, alpha=0.2,
                   label='Raw Data')
        if age_range is not None:
            st = age_range[0]
            en = age_range[1]
            ax.axvspan(
                st,
                en,
                color=age_range_color,
                alpha=0.2,
                label=f'Diagnostic: {st}–{en}',
            )
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel('Expression level', fontsize=12)
        ax.set_title(f"{title}", fontsize=14)

        st,en=get_range_by_median(df_raw['expr'])
        ax2.set_ylim(st,en)
        st,en=get_range_by_ci_median(df_fit['expr'],df_fit['low'],df_fit['high'])
        ax.set_ylim(st,en)

        ax.set_xlim(-5,100)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        # ax.spines['top'].set_visible(False)


    def plot_legend(self,out_path):
        sns.set(style="white", font="Arial")
        legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#333333',alpha=0.2,
                   markersize=6, label='Raw data'),
            Line2D([0], [0], color='#1f77b4', linewidth=2.5, label='Fitted mean'),
            Patch(facecolor='#1f77b4', edgecolor='none', alpha=0.2, label='95% CI'),
            Patch(facecolor='#FFA500', edgecolor='none', alpha=0.2, label='Diagnostic')
        ]
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.legend(handles=legend_elements, loc='center', frameon=False, ncol=1)
        ax.axis('off')
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.show()

    def cal_coexpr_gene_with_dis_age(self,top_sig=False,my_gene=None):
        top_n_gene=2
        coexpr_gene_table_dir=f'{self.analyze_dir}/coexpr_gene'
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_plot'
        # read age at onset.
        age_df=pd.read_excel(age_at_onset_path)
        age_ranges={'AD':[65,100]}
        for i in age_df.index:
            age_ranges[age_df.loc[i,'Abbr']]=[age_df.loc[i,'p25'],age_df.loc[i,'p75']]
        make_dir(coexpr_gene_table_dir,coexpr_gene_plot_dir)
        for gsn in self.gene_score_name:
            gam_path = f'{REAL_RNA}/gam/{gsn}'
            arr=gsn.split('.')
            ind_path = f'{REAL_RNA}/ind/{arr[0]}.{"-".join(arr[1].split("-")[:3])}.tsv'
            gam_df=replace_gene_symbol(extract_mean_expression(pd.read_table(gam_path,index_col=0)))
            ind_df=replace_gene_symbol(extract_mean_expression(pd.read_table(ind_path,index_col=0)))
            gam_mean_df,gam_up_df,gam_low_df=extract_mean_and_ci_expression(replace_gene_symbol(pd.read_table(gam_path,index_col=0)))
            n_cols=1
            n_rows=int(len(self.gwas_name)/n_cols)
            sns.set(style="white", font="Arial")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), constrained_layout=True,sharex=True)
            axes = axes.flatten()
            phenos=[]
            for i,gn in enumerate(self.gwas_name):
                coexpr_path=f'{coexpr_gene_table_dir}/{gsn}.{gn}.tsv'
                if not os.path.isfile(coexpr_path):
                    cell_df=self.__get_disease_score(gsn,gn)
                    dis_genes_df=self.__get_gene_score(gsn,gn)
                    corr_df=gene_disease_correlation(gam_df,cell_df)
                    merged_df = corr_df.merge(dis_genes_df, on='gene', how='left')
                    merged_df = merged_df.sort_values('spearman_corr', ascending=False).reset_index(drop=True)
                    merged_df.to_csv(coexpr_path,sep='\t',index=False,lineterminator='\n')
                    log(f'save corr genes')
                else:
                    merged_df=pd.read_table(coexpr_path)
                # top 5 genes
                top_n_gene_tag=top_n_gene
                if top_sig:
                    df_filtered = merged_df.dropna(subset=['gene_p'])
                    top_gene= df_filtered.nlargest(top_n_gene, 'spearman_corr')['gene'].values[top_n_gene-1]
                else:
                    top_gene = merged_df.nlargest(top_n_gene, 'spearman_corr')['gene'].values[top_n_gene-1]
                if my_gene is not None:
                    top_gene=my_gene
                    top_n_gene_tag=my_gene
                # start to plot
                ax = axes[i]
                pheno=gn.split('.')[0]
                phenos.append(pheno)
                title=fr'{top_gene} ({pheno})'
                expr_v=ind_df.loc[top_gene,:].values.flatten().tolist()
                df_raw=pd.DataFrame({'expr':expr_v,'age':ind_df.columns.map(lambda x:int(x.split(';')[1]))})
                df_fit=pd.DataFrame({'expr':gam_mean_df.loc[top_gene,:].values.flatten().tolist(),'age':gam_mean_df.columns.map(lambda x:int(x)),
                                     'low':gam_low_df.loc[top_gene,:].values.flatten().tolist(),
                                     'high':gam_up_df.loc[top_gene,:].values.flatten().tolist()})
                print(f"{gsn}:{pheno}: highest GAM expr at {df_fit.loc[df_fit['expr'].idxmax(), 'age']} yrs")
                age_range=None
                if pheno in age_ranges:
                    age_range=age_ranges[pheno]
                self.plot_gene_age_tend(title,df_raw,df_fit,age_range,ax)
            for j in range(len(self.gwas_name), len(axes)):
                axes[j].axis('off')
            db_name = gsn.split(".")[0].split('-')[-1]
            plt.tight_layout(h_pad=1.0)
            # plt.suptitle(f"{db_name}", fontsize=20)
            plt.show()
            pref='top_raw'
            if top_sig:
                pref='top_sig'
            fig.savefig(f"{coexpr_gene_plot_dir}/gene_expr_fit.top_{top_n_gene_tag}.{pref}.{gsn}.{'-'.join(phenos)}.jpg", dpi=300, bbox_inches='tight')
        # self.plot_legend(f"{coexpr_gene_plot_dir}/gene_expr_fit.legend.jpg")
    def plot_age_legend(self):
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_plot'
        self.plot_legend(f"{coexpr_gene_plot_dir}/gene_expr_fit.legend.jpg")
    def coexpr_gene_functional_enrich(self,top_n=200):
        coexpr_gene_table_dir=f'{self.analyze_dir}/coexpr_gene'
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_go'
        make_dir(coexpr_gene_table_dir,coexpr_gene_plot_dir)
        for gsn in self.gene_score_name:
            n_cols=1
            n_rows=int(len(self.gwas_name)/n_cols)
            # sns.set(style="white", font="Arial")
            # fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3.5 * n_rows), constrained_layout=True,sharex=True)
            # axes = axes.flatten()
            phenos=[]
            db_name = gsn.split(".")[0].split('-')[-1]
            for i,gn in enumerate(self.gwas_name):
                coexpr_path=f'{coexpr_gene_table_dir}/{gsn}.{gn}.tsv'
                merged_df=pd.read_table(coexpr_path)
                top_genes = merged_df.nlargest(top_n, 'spearman_corr')['gene'].values.tolist()
                # start to plot
                # ax = axes[i]
                pheno=gn.split('.')[0]
                phenos.append(pheno)
                title=f'{pheno}'
                go_enrich_plot(title,top_genes,None,out_path=f'{coexpr_gene_plot_dir}/enrich_table/{db_name}.{pheno}')
            # for j in range(len(self.gwas_name), len(axes)):
            #     axes[j].axis('off')
            # plt.suptitle(f"{db_name}", fontsize=20)
            # plt.subplots_adjust(left=0.67, right=0.8, bottom=0.15, top=0.9)
            # plt.show()
            pref=f'top_{top_n}'
            # fig.savefig(f"{coexpr_gene_plot_dir}/go_enrich.{pref}.{gsn}.{'-'.join(phenos)}.jpg", dpi=300, bbox_inches='tight')

    def plot_pheno_corr(self,phenos):
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_plot'
        sns.set(style="whitegrid", font="Arial", rc={"axes.edgecolor": "0.2", "grid.color": "#E0E0E0"})
        for gsn in self.gene_score_name:
            dfs=[]
            for gn in phenos:
                df = self.__get_disease_score(gsn, gn)
                df.index=df['age']
                df=df[['score']]
                df.columns=[gn]
                dfs.append(df)
            fdf=pd.concat(dfs, axis=1)
            corr_matrix = fdf.corr(method='spearman')
            plt.figure(figsize=(4,4))
            g=sns.clustermap(corr_matrix, center=0, vmin=-1, vmax=1,
                        annot=True, fmt=".2f", cmap='coolwarm',
                        linewidths=0,figsize=(6,6))
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
            db_name=gsn.split('.')[0].split('-')[1]
            plt.title(f'Spearman\'s R', fontsize=12)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(f'{coexpr_gene_plot_dir}/{db_name}.phenos.corr.png', dpi=300)
            plt.show()



def ad_ages():
    arr=[0.0127,0.0273,0.0552,0.1044,0.1854,0.3086,0.4819]
    arr = np.array(arr)
    cumsum = np.cumsum(arr)
    total = np.sum(arr)
    quantiles = cumsum / total
    print(quantiles)


if __name__ == '__main__':
    gwas_names=['ADHD','SCZ','MDD','BIP','AD','NEU','SMK','SD','IQ']
    gene_score_keys=['PsyEn-LIBD_szControl','PsyEn-CMC']
    output_dir = f'{REAL_ASSOC}'
    age_at_onset_path=f'{AGE_ONSET}'
    coexpr_gene_table_dir = f'{output_dir}/'
    adc=AnalyzeDiseaseCell('DESE','bhfdr_1eN2',gwas_names,output_dir,'tmm-raw-cov3-se',gene_score_keys)
    adc.plot_age_risk_multi_phenotypes(age_at_onset_path)
    adc.cal_coexpr_gene_with_dis_age(True)
    adc.plot_age_legend()
    adc.coexpr_gene_functional_enrich()
    # adc.cal_coexpr_gene_with_dis_age(True,my_gene='ZSCAN31')