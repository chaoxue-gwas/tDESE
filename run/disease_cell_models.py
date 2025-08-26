# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2025/04/18 12:23
# @Function:
import os.path
import re
from abc import ABC, abstractmethod

from util import make_dir, unified_path, batch_shell_plus, log, LOCAL_DIR


class DiseaseCellModel(ABC):
    @abstractmethod
    def __init__(self, gwas_file: str, gene_score_files: dict[str, str], output_prefix: str, params: dict):
        self.gwas_file = gwas_file
        self.gene_score_files = gene_score_files
        self.output_prefix = output_prefix
        self.params = params
        pass

class DESE(DiseaseCellModel):
    def __init__(self, gwas_file: str, gene_score_files: dict[str, str], output_prefix: str, params: dict={}):
        self.gwas_file = gwas_file
        self.gene_score_files = gene_score_files
        self.output_prefix = output_prefix
        self.params = {
            'java': 'java',
            'jar_path': '/home/xc/local/program/software/kggsum/20250418/kggsum0617.jar',
            'ref_genotype': '/home/xc/local/data/resources/RefGenotype/1KG_super_pop_combine/EUR.hg19.vcf.bgz',
            'ref_genotype_ver': 'hg19',
            'remove_mhc': True,
            'calc_specific': True,
            'gwas_ver': 'hg19',
            'gwas_chr': 'chr',
            'gwas_bp': 'bp',
            'gwas_a1':'a1',
            'gwas_a2':'a2',
            'gwas_p': 'p',
            'gene_model_db': 'refgene',
            'gene_p_method': 'bhfdr',
            'gene_p_cut': 1,
            'max_gene': 1000,
            'nt': 16
        }
        for k in params.keys():
            self.params[k] = params[k]
        pass

    def run_cmd(self) -> str:
        keys, values = zip(*self.gene_score_files.items())
        gs_names = list(keys)
        gs_files = list(values)
        gene_score_files = ' '.join(
            [f'--gene-score-file file={sf} calcSpecificity={"y" if self.params["calc_specific"] else "n"}' for sf in
             gs_files])
        mhc_size = 'chr6:28477797~33448354'
        # mhc_size = 'chr6:27477797~34448354'
        if self.params['gwas_ver'] == 'hg38':
            mhc_size = 'chr6:28510120~33480577'
        mhc_para = ''
        if self.params['remove_mhc']:
            mhc_para = f'exclude={mhc_size}'
        custom_para = f'''
            assoc
            --ref-gty-file {self.params['ref_genotype']}
             refG={self.params['ref_genotype_ver']}
            --sum-file {self.gwas_file}
             cp12Cols={','.join([self.params['gwas_'+x] for x in ['chr','bp','a1','a2']])}
             pbsCols={self.params['gwas_p']}
             refG={self.params['gwas_ver']} {mhc_para}
             {gene_score_files}
            --gene-p-cut {self.params['gene_p_cut']}
            --gene-multiple-testing {self.params['gene_p_method']}
            --max-condi-gene {self.params['max_gene']}
            --gene-model-database {self.params['gene_model_db']}
            --threads {self.params['nt']}
            --output {self.output_prefix}
        '''
        ## -upstream-distance 5000
        ## --downstream-distance 5000
        ### run DESE cmd
        clean_para = re.sub('\s+', ' ', custom_para).strip()
        jar_path = self.params["jar_path"]
        cmd = f'cd {os.path.dirname(jar_path)} && {self.params["java"]} -Xmx64G -jar {jar_path} {clean_para}'
        return cmd

    def norm_cell_result(self) -> None:
        pass

    def norm_gene_result(self) -> None:
        pass

class LDSC(DiseaseCellModel):
    def __init__(self, gwas_files: dict, top_gene_dir: str, ldsc_seg_dir, include_expr_dir_names=None):
        self.include_expr_dir_names=include_expr_dir_names
        self.ldsc_seg_dir = ldsc_seg_dir
        self.gwas_files = gwas_files
        self.top_gene_dir = top_gene_dir
        self.tissue_ldsc_dir = f'{top_gene_dir}_ldsc'
        self.ldsc_cts_dir = f'{top_gene_dir}_ldsc_cts'
        self.nt=10
        LDSC_bin_dir = f'/home/xc/local/program/python/pDESE/paper/lib/ldsc'
        self.start_ldsc_env=f'source /app/conda/bin/activate ldsc && cd {LDSC_bin_dir}'
        LDSC_resource_dir=unified_path(f'resources/ToolResource/LDSC')
        ## ENSG annotation for LDSC-SEG
        self.gene_annot_ldsc = f'{LDSC_resource_dir}/GRCh37.ENSG_coord.txt'
        self.raw_snp_list_ldsc = f'{LDSC_resource_dir}/w_hm3.snplist.gz'
        self.fine_hm3_snplist = f'{LDSC_resource_dir}/fine.w_hm3.snplist.gz'
        self.eur_ref_plink = f'{LDSC_resource_dir}/1000G_Phase3_plinkfiles'
        self.LDSC_1kg_baseline = f'{LDSC_resource_dir}/1000G_Phase3_baseline_v1.2/baseline.'
        self.LDSC_weights_hm3 = f'{LDSC_resource_dir}/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.'

    def cal_tissue_ldscore(self):
        cmds = []
        for expr_name in os.listdir(self.top_gene_dir):
            if self.include_expr_dir_names is not None and expr_name not in self.include_expr_dir_names:
                continue
            edir = f'{self.top_gene_dir}/{expr_name}'
            for f in os.listdir(edir):
                # if re.search(r'AllGene|BrainCortex|ArteryAorta',f) is None:
                #     continue
                gene_set_path = f'{edir}/{f}'
                cell_ldsc_prefix = f'{self.tissue_ldsc_dir}/{expr_name}/{re.sub(".txt", "", f)}'
                make_dir(os.path.dirname(cell_ldsc_prefix))
                for chr_i in range(1, 23):
                    cmd = f'''
                            {self.start_ldsc_env} &&
                            python make_annot.py
                               --gene-set-file {gene_set_path}
                               --gene-coord-file {self.gene_annot_ldsc}
                               --windowsize 100000
                               --bimfile {self.eur_ref_plink}/1000G.EUR.QC.{chr_i}.bim
                               --annot-file {cell_ldsc_prefix}.{chr_i}.annot.gz
                           &&
                           python ldsc.py
                               --l2
                               --bfile {self.eur_ref_plink}/1000G.EUR.QC.{chr_i}
                               --ld-wind-cm 1
                               --annot {cell_ldsc_prefix}.{chr_i}.annot.gz
                               --thin-annot
                               --out {cell_ldsc_prefix}.{chr_i}
                               --print-snps {self.fine_hm3_snplist}
                       '''
                    cmd = re.sub('\s+', ' ', cmd).strip()
                    cmds.append(cmd)
        batch_shell_plus(cmds, self.nt)

    def make_cts_file(self):
        make_dir(self.ldsc_cts_dir)
        for expr_name in os.listdir(self.tissue_ldsc_dir):
            edir = f'{self.tissue_ldsc_dir}/{expr_name}'
            expr_tag = f'{".".join(expr_name.split(".")[:2])}'
            cts_path = f'{self.ldsc_cts_dir}/{expr_tag}.ldcts'
            tissues = sorted(set(['.'.join(f.split('.annot.gz')[0].split('.')[:-1]) for f in os.listdir(edir) if
                                  f.endswith('.annot.gz')]))
            if 'AllGene' not in tissues:
                log(f'({expr_tag}) No background group, exit')
                continue
            with open(cts_path, 'w') as bw:
                for t in tissues:
                    if t == 'AllGene':
                        continue
                    bw.write('\t'.join([t, f'{edir}/{t}.,{edir}/AllGene.']) + '\n')
            log(f'({expr_tag}) save {len(tissues) - 1} tissues ldcts to {cts_path}')
        pass


    def run_regression(self):
        cmds = []
        make_dir(self.ldsc_seg_dir)
        for gwas_name,gwas_file in self.gwas_files.items():
            if not gwas_file.endswith('.sumstats.gz'):
                continue
            pheno = gwas_name
            for cts in os.listdir(self.ldsc_cts_dir):
                expr_tag = cts.split('.ldcts')[0]
                cmd = f'''
                      {self.start_ldsc_env} &&
                      python ldsc.py
                         --h2-cts {gwas_file} 
                         --ref-ld-chr {self.LDSC_1kg_baseline} 
                         --out {self.ldsc_seg_dir}/{pheno}.{expr_tag} 
                         --ref-ld-chr-cts {self.ldsc_cts_dir}/{cts} 
                         --w-ld-chr {self.LDSC_weights_hm3} 
                      '''
                cmd = re.sub('\s+', ' ', cmd).strip()
                cmds.append(cmd)
        batch_shell_plus(cmds, self.nt)

