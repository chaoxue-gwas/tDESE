# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/19 23:25
# @Function:
import os
import re
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze.analyze_disease_assoc_gene import KGGSUM
from analyze.analyze_disease_cell import get_range_by_median, get_range_by_ci_median, extract_mean_expression, \
    replace_gene_symbol, extract_mean_and_ci_expression, gene_disease_correlation
from para import REAL_ASSOC, AGE_ONSET, REAL_RNA
from util import unified_path, make_dir, log


class AnalyzeDiseaseCell():
    def __init__(self,tool_name,tag,include_gwass,output_dir,key_word):
        self.tool_name=tool_name
        self.analyze_dir = f'{output_dir}/{tool_name}.analyze.{tag}'
        self.result_dir = f'{output_dir}/{tool_name}.run.{tag}'
        if include_gwass is None:
            include_gwass = [f for f in os.listdir(self.result_dir) if os.path.isdir(f'{self.result_dir}/{f}')]
        self.gwas_name = include_gwass
        make_dir(self.analyze_dir)

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
    def plot_high_risk_age_range(self,title,df_map,age_range, ax, age_col='age', score_col='score',sig_col='significant'):
        age_range_color = "#FFA500"
        sex_colors={
            'Male':("#aec7e8","#0d3b66"),
            'Female':("#f7b6d2","#800026")
        }
        sns.set_context("paper", font_scale=1.4)
        sns.set_style("white")
        show_legend=True
        if age_range is None:
            show_legend=False
        for sex in df_map.keys():
            df=df_map[sex]
            line_color,sig_point_color=sex_colors[sex]
            df = df.dropna().copy()
            df = df.sort_values(by=age_col)
            sns.lineplot(x=age_col, y=score_col, data=df, ax=ax, color=line_color, linewidth=3, zorder=1)
            sns.scatterplot(data=df, x=age_col, y=score_col, hue=sig_col,
                            palette={'Sig.': sig_point_color, 'No. Sig.': line_color},
                            ax=ax, s=30, edgecolor='black', alpha=0.8, legend=False, zorder=2,label=sex)

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

    def plot_age_risk_multi_phenotypes(self,age_at_onset_path):
        db_names=['PsyEn-CMC','PsyEn-LIBD_szControl']
        age_risk_plot_dir=f'{self.analyze_dir}/age_risk_multi_pheno'
        make_dir(age_risk_plot_dir)
        # read age at onset.
        age_df=pd.read_excel(age_at_onset_path)
        age_ranges={'AD':[65,100]}
        for i in age_df.index:
            age_ranges[age_df.loc[i,'Abbr']]=[age_df.loc[i,'p25'],age_df.loc[i,'p75']]
        sex_code={'Female':'0','Male':'1'}
        for db in db_names:
            assoc_age_table=f"{age_risk_plot_dir}/age_risk_pheno.scatter.{db}.xlsx"
            with pd.ExcelWriter(assoc_age_table) as writer:
                n_cols=1
                n_rows=len(self.gwas_name)
                sns.set(style="whitegrid", font="Arial", rc={"axes.edgecolor": "0.2", "grid.color": "#E0E0E0"})
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3 * n_rows), constrained_layout=True,sharex=True)
                axes = axes.flatten()
                for i,gn in enumerate(self.gwas_name):
                    df_map={}
                    for k in sex_code.keys():
                        gsn=f'{db}.batch-tmm-raw-cov2-{sex_code[k]}.0-se.tsv'
                        df_map[k]=self.__get_disease_score(f'{gsn}',gn)
                    ax = axes[i]
                    ag=None
                    if gn in age_ranges:
                        ag=age_ranges[gn]
                    self.plot_high_risk_age_range(f'{gn}', df_map, ag, ax)
                    for k in sex_code.keys():
                        cp_df=df_map[k].copy()
                        cp_df=cp_df[['Condition','EnrichmentScore','Adjusted(p)']]
                        cp_df.columns=['Age (years)','Enrichment score','Adjusted P']
                        cp_df.to_excel(writer, sheet_name=f'{gn}_{k}', index=False)
                for j in range(len(self.gwas_name), len(axes)):
                    axes[j].axis('off')
                db_name = db.split('-')[1]
                plt.suptitle(f"{db_name}", fontsize=20)
                plt.tight_layout(rect=[0, 0, 1, 1])
                plt.show()
                fig.savefig(f"{age_risk_plot_dir}/age_risk_pheno.scatter.{gsn}.jpg", dpi=300, bbox_inches='tight')
                plt.close()

    def plot_gene_age_tend(self,title,plot_dfs,age_range,ax):
        age_range_color = "#FFA500"
        # 'Male': ("#aec7e8", "#0d3b66"),
        # 'Female': ("#f7b6d2", "#800026")
        color_map={'Female':('#E66A6A','#E66A6A','#E66A6A'),'Male':('#4A90E2','#4A90E2','#4A90E2')}
        # color_map={'Female':('#DD8452','#DD8452','#DD8452'),'Male':('#4C72B0','#4C72B0','#4C72B0')}
        ax2 = ax.twinx()
        for k in plot_dfs.keys():
            fit_color,fit_range_color,scatter_color=color_map[k]
            df_raw, df_fit=plot_dfs[k]
            ax.plot(df_fit['age'], df_fit['expr'],
                    color=fit_color,
                    linewidth=2,
                    label='Fitted Mean')
            ax.fill_between(df_fit['age'], df_fit['low'], df_fit['high'],
                            color=fit_range_color, alpha=0.2, label='95% CI')
            ax2.scatter(df_raw['age'], df_raw['expr'],
                       color=scatter_color,
                       s=10, alpha=0.4,
                       label='Raw Data')
            st,en=get_range_by_median(df_raw['expr'])
            ax2.set_ylim(st,en)
            st,en=get_range_by_ci_median(df_fit['expr'],df_fit['low'],df_fit['high'])
            ax.set_ylim(st,en)
        ax.set_xlim(-5,100)
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
        ax.set_ylabel('Expression level', fontsize=12)
        ax.set_title(f"{title}", fontsize=14)
        # lines1, labels1 = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        # ax.spines['top'].set_visible(False)


    def plot_legend(self,out_path):
        sns.set(style="white", font="Arial")
        legend_elements = [
            Patch(facecolor='#E66A6A', edgecolor='#E66A6A', alpha=1, linewidth=2.5, label='Female'),
            Patch(facecolor='#4A90E2', edgecolor='#4A90E2', alpha=1, linewidth=2.5, label='Male'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='#333333',alpha=0.2,
                   markersize=6, label='Raw data'),
            Line2D([0], [0], color='#333333', linewidth=2.5, label='Fitted mean'),
            Patch(facecolor='#333333', edgecolor='none', alpha=0.2, label='95% CI'),
            Patch(facecolor='#FFA500', edgecolor='none', alpha=0.2, label='Diagnostic')
        ]
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.legend(handles=legend_elements, loc='center', frameon=False, ncol=1)
        ax.axis('off')
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.show()

    def cal_coexpr_gene_with_dis_age(self,top_sig=False):
        top_n_gene=1
        coexpr_gene_table_dir=f'{self.analyze_dir}/coexpr_gene'
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_plot'
        make_dir(coexpr_gene_table_dir,coexpr_gene_plot_dir)
        db_names=['PsyEn-CMC','PsyEn-LIBD_szControl']
        age_risk_plot_dir=f'{self.analyze_dir}/age_risk_multi_pheno'
        make_dir(age_risk_plot_dir)
        # read age at onset.
        age_df=pd.read_excel(age_at_onset_path)
        age_ranges={'AD':[65,100]}
        for i in age_df.index:
            age_ranges[age_df.loc[i,'Abbr']]=[age_df.loc[i,'p25'],age_df.loc[i,'p75']]
        sex_code={'Female':'0','Male':'1'}
        for db in db_names:
            n_cols = 1
            n_rows = int(len(self.gwas_name) / n_cols)
            sns.set(style="white", font="Arial")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), constrained_layout=True,
                                     sharex=True)
            axes = axes.flatten()
            phenos = []
            df_map = {}
            for k in sex_code.keys():
                gsn = f'{db}.batch-tmm-raw-cov2-{sex_code[k]}.0-se.tsv'
                gam_path = f'{REAL_RNA}/gam/{gsn}'
                arr = gsn.split('.')
                ind_path = f'{REAL_RNA}/ind/{arr[0]}.{"-".join(arr[1].split("-")[:3])}.tsv'
                gam_df = replace_gene_symbol(extract_mean_expression(pd.read_table(gam_path, index_col=0)))
                ind_df = replace_gene_symbol(extract_mean_expression(pd.read_table(ind_path, index_col=0)))
                ind_df = ind_df[[c for c in ind_df.columns if int(c.split(';')[2])==int(sex_code[k])]]
                gam_mean_df, gam_up_df, gam_low_df = extract_mean_and_ci_expression(
                    replace_gene_symbol(pd.read_table(gam_path, index_col=0)))
                df_map[k]=(gam_df,ind_df,gam_mean_df,gam_up_df,gam_low_df)
            for i, gn in enumerate(self.gwas_name):
                coexpr_path = f'{coexpr_gene_table_dir}/{gsn}.{gn}.tsv'
                if not os.path.isfile(coexpr_path):
                    dfs=[]
                    for k in sex_code.keys():
                        gsn = f'{db}.batch-tmm-raw-cov2-{sex_code[k]}.0-se.tsv'
                        cell_df = self.__get_disease_score(gsn, gn)
                        dis_genes_df = self.__get_gene_score(gsn, gn)
                        gam_df=df_map[k][0]
                        corr_df = gene_disease_correlation(gam_df, cell_df)
                        out_df = corr_df.merge(dis_genes_df, on='gene', how='left')
                        out_df = out_df.sort_values('spearman_corr', ascending=False).reset_index(drop=True)
                        out_df.columns=out_df.columns.map(lambda x:x if x=='gene' else f'{x}.{k}')
                        dfs.append(out_df)
                    merged_df = reduce(lambda left, right: pd.merge(left, right, on='gene', how='outer'), dfs)
                    merged_df.to_csv(coexpr_path, sep='\t', index=False, lineterminator='\n')
                    log(f'save corr genes')
                else:
                    merged_df = pd.read_table(coexpr_path)
                # top 5 genes
                df=merged_df
                # df['spearman_corr_mean'] = (df['spearman_corr.Female'] + df['spearman_corr.Male']) / 2
                df['spearman_corr_mean'] = df[['spearman_corr.Female', 'spearman_corr.Male']].min(axis=1)
                if top_sig:
                    mask = df[['gene_p.Female', 'gene_p.Male', 'gene']].notna().all(axis=1)
                    df_filtered = df[mask]
                    # idx = df_filtered['spearman_corr_mean'].idxmax()
                    largest = df_filtered['spearman_corr_mean'].drop_duplicates().nlargest(top_n_gene).iloc[-1]
                    idx = df_filtered[df_filtered['spearman_corr_mean'] == largest].index[0]
                    top_gene = df_filtered.loc[idx, 'gene']
                    log(df_filtered.loc[idx,:])
                else:
                    # idx = df['spearman_corr_mean'].idxmax()
                    largest = df['spearman_corr_mean'].drop_duplicates().nlargest(top_n_gene).iloc[-1]
                    idx = df[df['spearman_corr_mean'] == largest].index[0]
                    top_gene = df.loc[idx, 'gene']
                    log(df.loc[idx, :])
                # start to plot
                ax = axes[i]
                pheno = gn.split('.')[0]
                phenos.append(pheno)
                title = fr'{top_gene} ({pheno})'
                plot_dfs={}
                for k in sex_code.keys():
                    gam_df, ind_df, gam_mean_df, gam_up_df, gam_low_df = df_map[k]
                    expr_v = ind_df.loc[top_gene, :].values.flatten().tolist()
                    df_raw = pd.DataFrame({'expr': expr_v, 'age': ind_df.columns.map(lambda x: int(x.split(';')[1]))})
                    df_fit = pd.DataFrame({'expr': gam_mean_df.loc[top_gene, :].values.flatten().tolist(),
                                           'age': gam_mean_df.columns.map(lambda x: int(x)),
                                           'low': gam_low_df.loc[top_gene, :].values.flatten().tolist(),
                                           'high': gam_up_df.loc[top_gene, :].values.flatten().tolist()})
                    plot_dfs[k]=(df_raw, df_fit)
                age_range = None
                if pheno in age_ranges:
                    age_range = age_ranges[pheno]
                self.plot_gene_age_tend(title, plot_dfs, age_range, ax)
            ax.set_xlabel('Age (years)', fontsize=12)
            for j in range(len(self.gwas_name), len(axes)):
                axes[j].axis('off')
            db_name = gsn.split(".")[0].split('-')[-1]
            plt.suptitle(f"{db_name}", fontsize=20)
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.show()
            pref = 'top_raw'
            if top_sig:
                pref = 'top_sig'
            fig.savefig(f"{coexpr_gene_plot_dir}/gene_expr_fit.top_{top_n_gene}.{pref}.{gsn}.{'-'.join(phenos)}.jpg", dpi=300,
                        bbox_inches='tight')

    def plot_age_legend(self):
        coexpr_gene_plot_dir = f'{self.analyze_dir}/coexpr_gene_plot'
        self.plot_legend(f"{coexpr_gene_plot_dir}/gene_expr_fit.legend.jpg")



if __name__ == '__main__':
    gwas_names = ['SCZ', 'MDD', 'BIP']
    gwas_names=['ADHD','SCZ','MDD','BIP','AD','NEU','SMK','SD','IQ']
    output_dir = f'{REAL_ASSOC}'
    age_at_onset_path=f'{AGE_ONSET}'
    adc=AnalyzeDiseaseCell('DESE','bhfdr_1eN2',gwas_names,output_dir,'tmm-raw-cov2-1.0-se') #tmm-raw-mean
    adc.plot_age_risk_multi_phenotypes(age_at_onset_path)
    adc.cal_coexpr_gene_with_dis_age(True)
    adc.plot_age_legend()
