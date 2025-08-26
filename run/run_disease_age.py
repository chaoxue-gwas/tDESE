# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/18 21:19
# @Function:
import os
from abc import ABC

from disease_cell_models import DESE, LDSC
from para import REAL_ASSOC, REAL_RNA
from util import log, batch_shell, unified_path


class RunDiseaseCell():
    def __init__(self,tool_name:str,gwas_names:[],gene_score_dir:str,output_dir):
        gwas_dir_keys={'DESE':'pmglab','LDSC':'ldsc'}
        gwas_suffix={'DESE':'.gwas.sum.tsv.gz','LDSC':'.ldsc.sumstats.gz'}
        self.tool_name=tool_name
        self.gwas_dir=unified_path(f'resources/GWAS/gold_case/{gwas_dir_keys[tool_name]}_formatted')
        self.gene_score_dir = gene_score_dir
        self.output_dir = output_dir
        self.select_gwas_name=gwas_names
        self.gwas_files:dict[str,str] = None
        self.gene_score_files = None
        self.__init_param(gwas_suffix[tool_name])

    def __init_param(self,gwas_file_suffix):
        gene_score_dir=self.gene_score_dir
        gene_score_files={}
        for f in os.listdir(gene_score_dir):
            name='.'.join(f.split('.')[:2])
            gene_score_files[name]=f'{gene_score_dir}/{f}'
        self.gene_score_files=gene_score_files
        self.gwas_files={p:f'{self.gwas_dir}/{p}{gwas_file_suffix}' for p in self.select_gwas_name}
        ## check file
        pass_check=True
        for gf in [self.gwas_files,self.gene_score_files]:
            for n,f in gf.items():
                if not os.path.isfile(f):
                    log(f'Error: no such file: {f}')
                    pass_check=False
        if not pass_check:
            raise Exception(f'Do not pass check files')


    def run(self,tag,tool_params:{}) -> object:
        output_tag=f'{self.tool_name}.run.{tag}'
        if self.tool_name=='DESE':
            #
            ntasks=10
            out_dir=f'{self.output_dir}/{output_tag}'
            cmds=[]
            for gwas_name,gwas_file in self.gwas_files.items():
                output_prefix=f'{out_dir}/{gwas_name}'
                dese=DESE(gwas_file,self.gene_score_files,output_prefix,tool_params)
                cmd=dese.run_cmd()
                cmds.append(cmd)
            batch_shell(cmds, ntasks, f'{out_dir}/batch_run.log')
        if self.tool_name=='LDSC':
            top_gene_dir = f'{self.gene_score_dir}_top_{tool_params["top_n"]}'
            out_dir=f'{self.output_dir}/{output_tag}'
            include_exprs = []
            for f in os.listdir(top_gene_dir):
                arr = f.split('.')
                if arr[2].split('timemodel_')[-1] in self.gs_spec_methods and arr[3] in self.gs_norm_methods:
                    include_exprs.append(f)
            ldsc = LDSC(self.gwas_files, top_gene_dir, out_dir, include_exprs)
            ldsc.cal_tissue_ldscore()
            ldsc.make_cts_file()
            ldsc.run_regression()


if __name__ == '__main__':
    gwas_names = ['ADHD','BIP','MDD','SCZ','AD'] + ['SD','SMK','IQ','NEU']
    # gwas_names = ['SD','SMK','IQ','RT','NEU']
    gene_score_dir = f'{REAL_RNA}/gam'
    output_dir = f'{REAL_ASSOC}'
    # run DESE
    rd=RunDiseaseCell('DESE',gwas_names,gene_score_dir,output_dir)
    rd.run('bhfdr_1eN2',{'gene_p_cut':0.01,'gene_p_method':'bhfdr','remove_mhc':True})
    # rd.run('bhfdr_5eN2',{'gene_p_cut':0.05,'gene_p_method':'bhfdr','remove_mhc':True})
    # run LDSC
    # gs_spec_methods = ['gam']
    # rd=RunDiseaseCell('LDSC',gwas_names,gs_tag,gs_norm_methods,gs_spec_methods)
    # rd.run('top_10perc',{'top_n':'10perc'})
