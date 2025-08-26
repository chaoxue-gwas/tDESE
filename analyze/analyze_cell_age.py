# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/22 20:18
# @Function: Analyze age of sample
import os.path
import re
from abc import ABC

import pandas as pd
import seaborn
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from para import REAL_RNA
from util import unified_path, make_dir

sns.set(style="white", font="Arial")
class TimeSpecificGeneScoreModel(ABC):
    def __init__(self):
        self.ages = None
        self.expr_df = None

    def phase_age(self,col:str) -> int:
        arr=col.split(';')
        # return arr[-1]
        if len(arr)==1:
            # age=int(arr[0].split('_')[0])
            numbers = re.findall(r'\d+', arr[0])
            age=int(numbers[0])
        if len(arr)==2:
            age=int(arr[0])
        if len(arr)>2:
            age=int(arr[1].split('_')[0])
        return age
    def load_gene_score(self,gene_score_path) -> (dict[str,int],pd.DataFrame):
        df=pd.read_table(gene_score_path,index_col=0)
        df=df[[x for x in df.columns if not str(x).endswith('.SE')]]
        def remove_mean_suffix(s):
            if s.endswith('.mean'):
                return s[:-5]
            return s
        df.columns=df.columns.map(lambda x:remove_mean_suffix(str(x)))
        col_age={}
        for c in df.columns:
            col_age[c]= self.phase_age(c)#int(str(c).split('.')[1].split('_')[0])
        sorted_cols=sorted(df.columns, key=lambda x: col_age[x])
        df = df[sorted_cols]
        # if len(df.columns[0].split(';'))>1:
        #     df.columns=df.columns.map(lambda x:x.split(';')[1])
        return col_age,df
class AnalyzeCellAge(TimeSpecificGeneScoreModel):
    def __init__(self,gene_score_path,plot_dir):
        self.gs_tag=os.path.basename(gene_score_path).split('.')[0].split('-')[-1]
        self.gs_age,self.gs_df=self.load_gene_score(gene_score_path)
        self.plot_dir=plot_dir
        make_dir(plot_dir)

    def plot_sample_age(self,key=''):
        df_t = self.gs_df.T
        ages = df_t.index.to_series().map(self.gs_age)
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(df_t.values)
        plt.figure(figsize=(4, 3.6))
        df=pd.DataFrame({'UMAP_1':embedding[:, 0],'UMAP_2':embedding[:, 1],'cate':ages})
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=ages, cmap='RdBu_r', s=20, edgecolor='k')
        plt.xticks([])
        plt.yticks([])
        # sc = seaborn.scatterplot(df,x='UMAP_1',y='UMAP_2',hue='cate')
        plt.colorbar(sc, label='Age')
        plt.title(f'{self.gs_tag} ({key})',fontsize=16)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.grid(False)
        # plt.gca().set_facecolor('#f5f5f5')
        # sns.despine()
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/{self.gs_tag}.{key}.umap.png',dpi=300)
        plt.show()

    def plot_sample_corr(self,key=''):
        df_t=self.gs_df
        df_t = df_t.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        corr_matrix = df_t.corr(method='pearson')
        plt.figure(figsize=(8, 7))
        sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                    linewidths=0, cbar_kws={'label': 'Pearson Correlation'})
        plt.title(f'{self.gs_tag} ({key})',fontsize=16)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'{self.plot_dir}/{self.gs_tag}.{key}.corr.png',dpi=300)
        plt.show()


    def plot_age_dist(self):
        df_t = self.gs_df.T
        print(df_t.shape)
        ages = df_t.index.to_series().map(self.gs_age)
        show_sex={'0':'Female','1':'Male'}
        sex = df_t.index.to_series().map(lambda x:show_sex[str(x).split(';')[2]])
        df=pd.DataFrame({'Age':ages,'Sex':sex})
        plt.figure(figsize=(6, 4))
        sns.set(style="white")
        sex_colors = {
            'Male': '#4C72B0',
            'Female': '#DD8452'
        }
        sns.histplot(
            data=df,
            x='Age',
            hue='Sex',
            multiple='stack',
            bins=list(range(0, 92, 5)),
            palette=sex_colors,
            # palette='Set2',
            edgecolor='white',
            hue_order=['Male','Female'],
        )
        plt.title(self.gs_tag.split('.')[0].split('-')[-1], fontsize=16, weight='bold')
        plt.xlabel("Age", fontsize=13)
        plt.ylabel("Sample size", fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        sns.despine()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    file_key={'ind':'Sample','gam':'GAM'}
    dir_name='ind'
    gene_score_dir=f'{REAL_RNA}/{dir_name}'
    gene_score_names = os.listdir(gene_score_dir)
    plot_dir=f'{REAL_RNA}/analysis/plot'
    for f in gene_score_names:
        if not f.startswith('PsyEn'):
            continue
        print(f)
        gene_score_path=f'{gene_score_dir}/{f}'
        aca=AnalyzeCellAge(gene_score_path,plot_dir)
        aca.plot_age_dist()
        aca.plot_sample_age(file_key[dir_name])
        aca.plot_sample_corr(file_key[dir_name])

