# -*- coding: utf-8 -*-
# @Author  : Xue Chao
# @Time    : 2023/07/03 17:22
# @Function:
import concurrent
import concurrent.futures
import gzip
import inspect
import multiprocessing
import platform
import re
import subprocess
import sys
import tempfile
import time
import os

import filetype
import gseapy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn
from gprofiler import GProfiler
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr,pearsonr
import pandas as pd
import matplotlib.colors as mcolors
from seaborn import cm
from sympy.physics.units import ft
import seaborn as sns

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR=f'{LOCAL_DIR}/resources'
FONTS_DIR=f'{RESOURCE_DIR}/fonts'

def unified_path(path:str) -> str:
    DATA_DIR = '/home/xc/local/data'
    if platform.system() == 'Windows':
        DATA_DIR = r'E:\WorkData\syncHPC\home\data'
    return f'{DATA_DIR}/{path}'


def log(content):
    """
    Log function
    :param content:
    """
    content = time.strftime("%Y-%m-%d %H:%M:%S [INFO] ", time.localtime(time.time())) + "%s" % content
    print(content, flush=True)


def log_exception(e):
    content = time.strftime("%Y-%m-%d %H:%M:%S [Exception] ", time.localtime(time.time())) + f"{type(e)}: {e}"
    print(content, flush=True)


def make_dir(*dirs):
    """
    Create the directory if it does not exist.
    :param dirs:
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def kggseq_ref_genes():
    ref_gene_path = f'{LOCAL_DIR}/resources/kggseqv1.1_hg19_refGene..txt'
    df=pd.read_table(ref_gene_path,header=None)
    genes=set(df[0].unique())
    gene_s=get_gene_symbol(sorted(genes))
    gene_syms=set()
    for g in gene_s:
        g=g.strip()
        if g!='' and g!='NA':
            gene_syms.add(g)
    return gene_syms

def batch_run_function(func,args,nt):
    if len(args)<nt:
        nt=len(args)
    if nt<=1:
        for arg in args:
            func(*arg)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nt) as executor:
            futures = [executor.submit(func, *item) for item in args]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_exception(e)
    # with multiprocessing.Pool(processes=nt) as pool:
    #     futures = [pool.apply_async(func, args=item) for item in args]
    #     pool.close()
    #     pool.join()
    #     for future in futures:
    #         try:
    #             future.get()
    #         except Exception as e:
    #             log_exception(e)

def remove_files(*args):
    for a in args:
        if os.path.isfile(a):
            os.remove(a)

def batch_run_function_linux(func,args,nt):
    if len(args)<nt:
        nt=len(args)
    if nt<=1:
        for arg in args:
            func(*arg)
    else:
        func_name=func.__name__
        module = inspect.getmodule(func)
        module_file = inspect.getfile(module)
        module_name = os.path.basename(module_file)[:-3]
        unique_name=f'{module_name}_{func_name}.{int(time.time())}'
        tmp_py=f'{LOCAL_DIR}/bat.{unique_name}.tmp'
        with open(tmp_py,'w') as bw:
            bw.write(f"import sys \
                \nfrom {module_name} import {func_name} \
                \nif __name__=='__main__':  \
                \n {func_name}(*sys.argv[1:]) ")
        cmds=[]
        for arg in args:
            cmd=f'python {tmp_py} '+' '.join([f'{a}' for a in arg])
            cmds.append(cmd)
        batch_shell_plus(cmds,nt)
        remove_files(tmp_py)

def clear_str_arr(genes:[]):
    ngenes = []
    for g in genes:
        g = str(g).strip()
        if g != '':
            ngenes.append(g)
    return ngenes

class GeneID():
    def __init__(self):
        self.type_id_symbol = None
        hgnc_path = f'{LOCAL_DIR}/resources/HGNC_20210810.txt'
        df = pd.read_table(hgnc_path, header=0, index_col=0, dtype='str')
        ## 过滤舍弃的Gene Symbol记录
        self.df = df.loc[df['Status'] == 'Approved',]
        self.__get_gene_symbol_map()
        self.__get_gene_symbol_alias()

    def __gene_id_type(self,gene_id:str):
        gene_type = 'symbol'
        if len(gene_id) > 14 and gene_id.startswith('ENSG'):
            gene_type = 'ensg_id'
        if re.match('^[\d]+$', gene_id):
            gene_type = 'ncbi_id'
        return gene_type

    def __get_gene_symbol_map(self):
        df=self.df
        gt_cols={
            'symbol':['Approved symbol', 'Previous symbols', 'Alias symbols'],
            'ensg_id':['Ensembl gene ID'],
            'ncbi_id':['NCBI Gene ID']
        }
        type_id_symbol = {gt:{} for gt in gt_cols.keys()}
        for i in df.index:
            if pd.isna(df.loc[i, 'Approved symbol']):
                continue
            v = str(df.loc[i, 'Approved symbol']).strip()
            for gt in gt_cols.keys():
                gene_type = gt_cols[gt]
                for col in gene_type:
                    if df.loc[i, col] and df.loc[i, 'Approved symbol']:
                        for x in str(df.loc[i, col]).split(','):
                            k = x.strip()
                            if k=='' or v=='':
                                continue
                            type_id_symbol[gt][k] = v
        self.type_id_symbol=type_id_symbol

    def __get_gene_symbol_alias(self):
        df=self.df
        symbol_arr={}
        gene_col=['Approved symbol', 'Previous symbols', 'Alias symbols']
        for i in df.index:
            if pd.isna(df.loc[i, 'Approved symbol']):
                continue
            v = str(df.loc[i, 'Approved symbol']).strip()
            if v=='':
                continue
            symbol_arr[v]=set()
            for col in gene_col:
                if pd.isna(df.loc[i, col]):
                    continue
                for x in str(df.loc[i, col]).split(','):
                    k = x.strip()
                    if k=='':
                        continue
                    symbol_arr[v].add(k)
        self.symbol_alias=symbol_arr

    def get_symbol_list(self):
        return sorted(self.symbol_alias.keys())

    def get_gene_symbol(self,genes,na_raw=False):
        genes=clear_str_arr(genes)
        if len(genes)<1:
            return None
        gene_symbols = []
        gt=self.__gene_id_type(genes[0])
        id_symbol=self.type_id_symbol[gt]
        for g in genes:
            if gt=='ensg_id':
                g = g.strip().split('.')[0]
            gs = 'NA'
            if na_raw:
                gs = g
            if g in id_symbol:
                gs = id_symbol[g]
            gene_symbols.append(gs)
        return gene_symbols

    def get_gene_alias(self,genes,include_self=False):
        genes_sym=self.get_gene_symbol(genes,True)
        if not genes_sym:
            return None
        gene_alias=[]
        for g in genes_sym:
            alias=set()
            if include_self:
                alias.add(g)
            if g in self.symbol_alias:
                alias.union(self.symbol_alias[g])
            gene_alias.append(alias)
        return gene_alias

def get_gene_symbol(genes:[]):
    gid=GeneID()
    return gid.get_gene_symbol(genes)

def get_gene_alias(genes:[]):
    gid=GeneID()
    return gid.get_gene_alias(genes,include_self=True)

def need_trans_gene_name(gene_id:str):
    gene_id=str(gene_id).strip()
    need_trans=False
    if len(gene_id) > 14 and gene_id.startswith('ENSG'):
        need_trans=True
    if re.match('^[\d]+$', gene_id):
        need_trans=True
    return need_trans

def cluster_df(cor,method='average', metric= 'euclidean'):
    Z = hierarchy.linkage(cor, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    x_order=x['leaves']
    Z = hierarchy.linkage(cor.T, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    y_order=x['leaves']
    cor=cor.iloc[x_order,y_order]
    return cor


def corr_p(df:pd.DataFrame,method='pearson',sort_by_cluster=True):
    """
    calculate correlation coefficients and p-values.
    :param df:
    :param method:
    :return:
    """
    cols=df.columns.tolist()
    col_num=len(cols)
    cor=np.zeros(np.repeat(col_num,2))
    p=np.zeros(np.repeat(col_num,2))
    for i in range(col_num):
        for j in range(i,col_num):
            if method=='pearson':
                cc,pv=pearsonr(df.iloc[:,i],df.iloc[:,j])
            if method=='spearman':
                cc,pv=spearmanr(df.iloc[:,i],df.iloc[:,j])
            cor[i,j]=cc
            cor[j,i]=cc
            p[i,j]=pv
            p[j,i]=pv
    cor_df=pd.DataFrame(cor,index=cols,columns=cols)
    p_df=pd.DataFrame(p,index=cols,columns=cols)
    if sort_by_cluster:
        cor_df=cluster_df(cor_df)
        p_df=p_df.loc[cor_df.index,cor_df.columns]
    return cor_df,p_df

def read_line(file_path):
    isGzip = True
    try:
        if str(filetype.guess(file_path).extension) == "gz":
            isGzip = True
    except:
        isGzip = False
    if isGzip:
        reader = gzip.open(file_path, "r")
    else:
        reader = open(file_path, "r")
    while True:
        line = reader.readline()
        if not line:
            reader.close()
            break
        if isGzip:
            lineArr = line.decode().strip('\n')
        else:
            lineArr = line.strip('\n')
        yield lineArr

def batch_shell(all_task,limit_task,log_file,time_sleep=0.1):
    make_dir(os.path.dirname(log_file))
    log(f'stdin/stdou/stderr info in: {log_file}')
    log_bw = open(log_file, "w")
    task_pool=[]
    task_remain=len(all_task)
    for task in all_task:
        task_remain+=-1
        break_out = True
        p = subprocess.Popen(task, shell=True, stdin=log_bw, stdout=log_bw, stderr=log_bw)
        task_pool.append(p)
        log(f' ({len(all_task)-task_remain}/{len(all_task)}) '+str(p.pid)+': '+task+' start ...')
        if len(task_pool)==limit_task or task_remain==0:
            while break_out:
                for intask_Popen in task_pool:
                    if intask_Popen.poll()!=None:
                        log(f'{str(intask_Popen.pid)} finish')
                        task_pool.remove(intask_Popen)
                        break_out = False
                        if task_remain==0:
                            break_out=True
                        if len(task_pool)==0:
                            break_out=False
                        break
                time.sleep(time_sleep)
    log_bw.close()

def print_std(l,close=True):
    l.seek(0)
    output = l.read().decode().strip()
    if output!='':
        print(output, flush=True)
    if close:
        l.close()

class FlushLog:
    def __init__(self,io):
        self.last_seek=0
        self.io=io
    def print_std(self):
        self.io.seek(self.last_seek)
        output = self.io.read().decode().strip('\n')
        print(output, flush=True)
        self.io.last_seek=self.io.tell()
        print(self.io.last_seek)

def batch_shell_plus(all_task,limit_task,time_sleep=0.1):
    task_pool=[]
    task_remain=len(all_task)
    for task in all_task:
        task_remain+=-1
        break_out = True
        fileno = tempfile.NamedTemporaryFile(delete=True)
        p = subprocess.Popen(task, shell=True, stdout=fileno, stderr=fileno)
        log(f'({len(all_task)-task_remain}/{len(all_task)}) {str(p.pid)}: {task}')
        task_pool.append([p,fileno])
        if len(task_pool)>=limit_task or task_remain==0:
            while break_out:
                for p,fileno in task_pool:
                    if p.poll()!=None:
                        print_std(fileno)
                        log(f'complete {str(p.pid)}')
                        task_pool.remove([p,fileno])
                        break_out = False
                        if task_remain==0:
                            break_out=True
                        if len(task_pool)==0:
                            break_out=False
                        break
                time.sleep(time_sleep)

def replace_with_gene_symbol_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    expr_df = expr_df.iloc[gene_idxs, :]
    # expr_df = expr_df[sorted(expr_df.columns)]
    expr_df.index=np.array(gene_syms)[gene_idxs]
    return expr_df

def replace_with_gene_symbol_in_refgene_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    expr_df = expr_df.iloc[gene_idxs, :]
    expr_df = expr_df[sorted(expr_df.columns)]
    expr_df.index=np.array(gene_syms)[gene_idxs]
    return expr_df

def index_in_gene_symbol_and_refgene_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    return gene_idxs

def index_in_gene_symbol_and_refgene_of_gene_list(genes:[]):
    gene_syms = get_gene_symbol(genes)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes_arr = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes_arr or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes_arr.add(gene_syms[i])
    return gene_idxs,gene_syms

def index_in_gene_symbol_of_gene_list(genes:[]):
    gene_syms = get_gene_symbol(genes)
    gene_idxs = []
    genes_arr = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes_arr:
            continue
        gene_idxs.append(i)
        genes_arr.add(gene_syms[i])
    return gene_idxs,gene_syms

def pvalue_adjust(pvalue:[],method='FDR'):
    # print(pvalue)
    if method not in ['FDR','Bonf']:
        log('Not support other Method, try to adjust p value by FDR or Bonf!')
    if method=='FDR':
        leng=len(pvalue)
        if leng<3:
            return pvalue
        pvalue_idx=[(i,pvalue[i]) for i in range(leng)]
        sortpvalue=sorted(pvalue_idx,key=lambda x:x[1])
        bh_fdr=[]
        if sortpvalue[-1][1]>1:
            bh_fdr.append((sortpvalue[-1][0],1))
        else:
            bh_fdr.append(sortpvalue[-1])
        for i in range(2,leng+1):
            rank=leng - i+1
            pval_idx= sortpvalue[leng - i]
            fdr=pval_idx[1]* leng /rank
            fdr_front=bh_fdr[-1][1]
            if fdr>fdr_front:
                bh_fdr.append((pval_idx[0],fdr_front))
            else:
                bh_fdr.append((pval_idx[0],fdr))
        return [x[1] for x in sorted(bh_fdr,key=lambda x:x[0])]
    if method=='Bonf':
        return np.array(pvalue)*len(pvalue)

def cpm_df(counts_df,row_is_gene=True):
    """
    Return CPM (counts per million) modified xc
    """
    if not row_is_gene:
        counts_df=counts_df.T
    lib_size = counts_df.sum()
    fdf=counts_df / lib_size * 1e6
    if not row_is_gene:
        fdf=fdf.T
    return fdf

def cpm(counts_mat,row_is_gene=True):
    """
    Return CPM (counts per million) modified xc
    """
    cp_mat = counts_mat
    if not row_is_gene:
        cp_mat=np.copy(counts_mat).T
    lib_size=np.nansum(cp_mat,axis=0)
    fmat=cp_mat/lib_size*1e6
    if not row_is_gene:
        fmat=fmat.T
    return fmat

def dgn_heatmap(df:pd.DataFrame):
    '''
    heatmap by matplotlib
    :return:
    '''

    pass


def test_visual_corr(x1:np.ndarray,x2:np.ndarray,corr_method='pearson',tag=''):
    fig,ax=plt.subplots()
    ax.plot(x1,x2,'.')
    if corr_method=='pearson':
        cc,p=scipy.stats.pearsonr(x1,x2)
    if corr_method=='spearman':
        cc,p=scipy.stats.spearmanr(x1,x2)
    ax.set_title(f'{tag}{corr_method} cc={cc:.3f}; p={p:.3g}')
    plt.tight_layout()
    plt.show()
    return cc,p

def test_visual_diff(x1:np.ndarray,x2:np.ndarray,tag=''):
    fig,ax=plt.subplots(figsize=(3,4))
    stat,p = scipy.stats.ranksums(x1, x2, alternative='greater')
    seaborn.set(style="white")
    seaborn.boxplot(data=[x1,x2], palette=['#62B298','#EF8A66'],ax=ax)
    ax.set_xticklabels(['High','Low'], fontsize=12)
    # ax.boxplot([x1,x2], vert=True, patch_artist=True)
    ax.set_title(f'{tag}\n\n$P$ = {p:.2g}')
    plt.tight_layout()
    plt.show()
    return ax


def get_index(arr_des,arr_src):
    arr_src_idx={arr_src[i]:i for i in range(len(arr_src))}
    des_idx_src={}
    for i in range(len(arr_des)):
        if arr_des[i] in arr_src:
            des_idx_src[i]=arr_des[i]
    idxs=sorted(des_idx_src.keys(),key=lambda x:arr_src_idx[des_idx_src[x]])
    return idxs

def generate_sub_axes():
    pass

def cal_hist_density(arr):
    arr=arr[~np.isnan(arr) & ~np.isinf(arr)]
    return np.histogram(arr, bins=100, density=True)

def kggsee_rez(expr_path, out_prefix, min_tissue:int, kggsee_jar,resource_dir)->str:
    cmd = f'java -Xmx30G -jar {kggsee_jar} --calcu-selectivity-rez-webapp --min-tissue {min_tissue} --gene-expression {expr_path} --out {out_prefix}'
    cmd += f' --resource {resource_dir}'
    return cmd

def kggsee_rez_last(expr_path, out_prefix, min_tissue:int, kggsee_jar)->str:
    kggsee_dir=os.path.dirname(kggsee_jar)
    cmd = f'cd {kggsee_dir} && java -Xmx30G -jar {kggsee_jar} --calcu-selectivity-rez-webapp --min-tissue {min_tissue} --gene-expression {expr_path} --out {out_prefix}'
    return cmd

def kggsee_dese(gwas_file,gene_score,out_prefix,kggsee_jar,resource_dir,multi_correct_method,fwer,top_n_gene,nt,chr_col,bp_col,p_col,
                           ref_genome='hg19',remove_hla=False,java_path='java',jvm_gb='80',
                vcf_ref=None,keep_ref=None,saved_ref=None):
    '''
    run DESE function in KGGSEE. see docs in https://kggsee.readthedocs.io/en/latest/quick_tutorials.html#dese-driver-tissue-inference.
    :return:
    '''
    para = f'''
        --no-plot
        --resource {resource_dir}
        --sum-file {gwas_file}
        --expression-file {gene_score}
        --out {out_prefix}
        --multiple-testing {multi_correct_method}
        --p-value-cutoff {fwer}
        --buildver {ref_genome}
        --nt {nt}
        --chrom-col {chr_col}
        --pos-col {bp_col}
        --p-col {p_col}
        --filter-maf-le 0.05
        --gene-finemapping
        --dese-permu-num 100
        --db-gene refgene
        --no-gz
        --min-tissue 3
        --top-gene {top_n_gene}
    '''
    if saved_ref is None:
        para += f' --vcf-ref {vcf_ref}'
        if keep_ref is not None:
            para += f' --keep-ref {keep_ref}'
    else:
        para += f' --saved-ref {saved_ref}'
    hla_range={'hg19':'27477797-34448354','hg38':'28510120-33480577'}
    if remove_hla:
        para += f' --regions-out chr6:{hla_range[ref_genome]}'
    para=re.sub('\s+',' ',para)
    cmd = f'{java_path} -Xmx{jvm_gb}G -jar {kggsee_jar} {para}'
    return cmd

def run_command(command,log_prefix=''):
    popen = subprocess.Popen(command,shell=True, stdout = subprocess.PIPE,text=True)
    while True:
        line = popen.stdout.readline()
        if not line:
            break
        line=line.strip()
        if line!='':
            print(f'{log_prefix}{line}')
    return popen.returncode

class FunctionEnrichment:
    def __init__(self,db='KEGG'):
        self.db=db
        self.__load_term()

    def __load_term(self):
        db_paths={
            'KEGG':f'{LOCAL_DIR}/resources/KEGG_2021_Human.txt',
            'GO:BP':f'{LOCAL_DIR}/resources/GO_Biological_Process_2023.txt',
            'GO:MF': f'{LOCAL_DIR}/resources/GO_Molecular_Function_2023.txt',
            'GO:CC': f'{LOCAL_DIR}/resources/GO_Cellular_Component_2023.txt'
        }
        db_path=db_paths[self.db]
        bg_genes=[]
        gene_lib = {}
        gene_anno = {}
        gid=GeneID()
        with open(db_path, 'r') as br:
            for line in br:
                arr = line.strip().split('\t')
                term=arr[0].strip()
                gene_syms = gid.get_gene_symbol([x.strip() for x in arr[2:]],True)
                gene_lib[term]=gene_syms
                bg_genes+=gene_syms
                for g in gene_syms:
                    if g not in gene_lib:
                        gene_anno[g]=[]
                    gene_anno[g].append(term)

        self.gene_lib=gene_lib
        self.gene_anno=gene_anno
        self.bg_gene_num=len(set(bg_genes))
    def enrich(self,genes:[],enrich_adj_p_cutoff=0.05,bg_genes_num=None,no_plot=True):
        if bg_genes_num is None:
            bg_genes_num=self.bg_gene_num
        try:
            enr = gseapy.enrichr(genes, self.gene_lib, background=bg_genes_num, no_plot=no_plot)
        except:
            return None
        rdf = enr.results
        rdf = rdf.loc[rdf['Adjusted P-value'] < enrich_adj_p_cutoff,]
        return rdf

    def get_gene_term(self,gene):
        term=[]
        if gene in self.gene_anno:
            term=self.gene_anno[gene]
        return term

    def get_genes_term(self,genes:[],none_rep='None'):
        '''
        返回一组基因名的注释术语。
        :param genes:
        :param none_rep:
        :return:
        '''
        terms=[]
        for g in genes:
            term=self.get_gene_term(g)
            t=none_rep
            if len(term)>0:
                t=term[0]
            terms.append(t)
        return terms

class KGGSEE:
    def __init__(self, prefix):
        self.prefix = prefix

    def gene_based_p_cutoff(self):
        pcut = None
        i = 0
        for line in read_line(f'{self.prefix}.log'):
            if 'Significance level of p value cutoffs for the overall error rate' in line:
                i += 1
                pcut = float(line.strip().split(':')[-1].strip())
        return pcut

    def cond_assoc_genes_df(self,gene_score_file=None,index_col=None):
        if gene_score_file is None:
            gene_path=f'{self.prefix}.gene.assoc.condi.txt'
        else:
            gene_path=f'{self.prefix}{gene_score_file}.gene.assoc.condi.txt'
        df = pd.read_table(gene_path,index_col=index_col)
        return df

    def assoc_cell_df(self,gene_score_file=None):
        if gene_score_file is None:
            cell_path=f'{self.prefix}.celltype.txt'
        else:
            cell_path=f'{self.prefix}{gene_score_file}.celltype.txt'
        df = pd.read_table(cell_path,skipfooter=1, engine='python')
        df = df.sort_values(by=['Adjusted(p)'])
        return df

    def cond_sig_assoc_gene_p(self,gene_score_file=None):
        gene_p={}
        p_cut=self.gene_based_p_cutoff()
        df=self.cond_assoc_genes_df(gene_score_file)
        df=df.loc[df['CondiECSP']<p_cut,:]
        for i in df.index:
            gene_p[df.loc[i, 'Gene']] = df.loc[i, 'CondiECSP']
        return gene_p

    def cond_sig_assoc_gene(self,gene_score_file=None):
        return sorted(self.cond_sig_assoc_gene_p(gene_score_file).keys())

    def assoc_cell_adj_p(self,gene_score_file=None):
        cell_p={}
        df=self.assoc_cell_df(gene_score_file)
        for i in df.index:
            cell_p[df.loc[i,'TissueName']]=float(df.loc[i,'Adjusted(p)'])
        return cell_p

    def assoc_cell_raw_p(self,gene_score_file=None):
        cell_p={}
        df=self.assoc_cell_df(gene_score_file)
        for i in df.index:
            cell_p[df.loc[i,'TissueName']]=float(df.loc[i,'Unadjusted(p)'])
        return cell_p

    def assoc_sig_cell_p(self,adj_p_cut,gene_score_file=None):
        cell_p=self.assoc_cell_adj_p(gene_score_file)
        sig_cell_p={}
        for c,p in cell_p.items():
            if p<adj_p_cut:
                sig_cell_p[c]=p
        return sig_cell_p

    def assoc_sig_cells(self,adj_p_cut,gene_score_file=None,min_top_n=1):
        cell_p = self.assoc_cell_adj_p(gene_score_file)
        cells=sorted(cell_p.keys(), key=lambda x: cell_p[x])
        k=0
        for c in cells:
            if cell_p[c]<adj_p_cut:
                k+=1
        if k<min_top_n:
            k=min_top_n
        return cells[:k]

    def genes_by_module_id(self,module_id):
        df=pd.read_table(f'{self.prefix}.assoc_gene_module.txt',index_col=0)
        genes=df.loc[module_id,'module_gene']
        return [x.strip() for x in str(genes).split(',') if x.strip()!='']



class HomoGene:
    def __init__(self,type='name'):
        self.type=type
        self.load_data()

    def load_data(self):
        type = self.type
        df=pd.read_table(f'{LOCAL_DIR}/resources/mart_hs2mm.txt.gz')
        type_col={'name':'Mouse gene name','id':'Mouse gene stable ID'}
        homo_map={}
        uniq_hs_genes=set()
        col=type_col[type]
        for i in df.index:
            mid=df.loc[i,col]
            hsg = df.loc[i, 'Gene name']
            if (not pd.isna(mid)) and (hsg not in uniq_hs_genes):
                homo_map[mid]=hsg
                uniq_hs_genes.add(hsg)
        self.homo_map=homo_map
    def mm_to_hs_genes(self,mm_gene_names:[]):
        '''
        translate mouse gene to human.
        :return:
        '''
        homo_map = self.homo_map
        homo_genes=[]
        map_idxs=[]
        for i in range(len(mm_gene_names)):
            mg=mm_gene_names[i].strip()
            if mg in homo_map:
                map_idxs.append(i)
                homo_genes.append(homo_map[mg])
        return np.array(map_idxs),np.array(homo_genes)

def remove_last_bracket(input_str):
    pattern = re.compile(r'\([^)]*\)$')
    match = pattern.search(input_str)
    if match:
        result_str = input_str[:match.start()] + input_str[match.end():]
        return result_str
    else:
        return input_str


def get_last_digits(input_str):
    match = re.search(r'\d+$', input_str)
    if match:
        return match.group()
    else:
        return ""

def __plot_bar(cmap,label='Spearman R',vmin=-1, vmax=1,save_path=''):
    fig, ax = plt.subplots(figsize=(2, 1))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, label=label, orientation='horizontal', aspect=5)
    plt.tight_layout()
    if save_path!='':
        plt.savefig(save_path)
    else:
        plt.show()

def heatmap_color_rstyle():
    return __heatmap_color_rstyle()

def heatmap_color_rstyle_single():
    colors = ['#FFFFFF', '#FE0100']
    cmap = LinearSegmentedColormap.from_list("heatmap_rstyle", list(zip([0, 1], colors)))
    return cmap

def __heatmap_color_rstyle():
    colors = ['#1B1AFD', '#FFFFFF', '#FE0100']
    cmap = LinearSegmentedColormap.from_list("heatmap_rstyle", list(zip([0, 0.5, 1], colors)))
    return cmap

def heatmap_color_rstyle():
    colors = ['#1B1AFD', '#FFFFFF', '#FE0100']
    cmap = LinearSegmentedColormap.from_list("heatmap_rstyle", list(zip([0, 0.5, 1], colors)))
    return cmap

def __plot_color_bar(cmap,label='Spearman R',vmin=-1, vmax=1,save_path='',horiz=True):
    fig, ax = plt.subplots(figsize=(1, 2))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    orient='vertical'
    if horiz:
        orient='horizontal'
        fig, ax = plt.subplots(figsize=(2, 1))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, label=label, orientation=orient, aspect=5)
    plt.tight_layout()
    if save_path!='':
        plt.savefig(save_path)
    else:
        plt.show()

def cell_corr_heatmap(cor: pd.DataFrame, out_fig):
    if cor.shape[0]<2:
        log(f'warning: empty df.')
        return
    cmap = __heatmap_color_rstyle()
    h_size = 0.2 * cor.shape[0]
    w_size = h_size
    fig, ax = plt.subplots(figsize=(w_size, h_size))
    seaborn.heatmap(cor, cmap=cmap, vmin=-1, vmax=1, linewidths=0.1, linecolor='#AEACAD', ax=ax, cbar=False)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

def cluster_df(cor,method='average', metric= 'euclidean'):
    nrow,ncol=cor.shape
    if nrow<2 or ncol<2:
        return cor
    Z = hierarchy.linkage(cor, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    x_order=x['leaves']
    Z = hierarchy.linkage(cor.T, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    y_order=x['leaves']
    cor=cor.iloc[x_order,y_order]
    return cor

def compare_cell_correlation(expr_path, degree_path):
    """
    compare tissue/cell type correlation according to expression or centrality of network.
    :return:
    """
    corr_method = 'spearman'


def jaccard(li1, li2, print_info=False):
    li1 = set(li1)
    li2 = set(li2)
    x=len(li1.intersection(li2))
    y=len(li1.union(li2))
    if y<1:
        return 0
    cc = x / float(y)
    if print_info:
        print(f'jaccard: {cc}; {len(li1)} vs {len(li2)}')
    return cc

def heatmap_custom1(df:pd.DataFrame,color_df:pd.DataFrame,annot_df:pd.DataFrame,save_fig):
    cmap=__heatmap_color_rstyle()
    base_size=100
    color_df=color_df.loc[df.index,df.columns]
    annot_df = annot_df.loc[df.index, df.columns]
    xs=[]
    ys=[]
    sizes=[]
    colors=[]
    fig, ax = plt.subplots(figsize=(5,4.4))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            x,y=j,df.shape[0]-i-1
            ys.append(y)
            xs.append(x)
            sizes.append(df.iloc[i,j])
            colors.append(color_df.iloc[i,j])
            ax.text(x,y,annot_df.iloc[i,j], ha='center', va='center',c='black',fontsize=6)
    sizes=np.abs(sizes)
    sizes=sizes*base_size/np.max(sizes)
    max_abs_r=color_df.abs().max().max()
    sc=ax.scatter(xs,ys, c=colors, s=sizes, alpha=0.5, cmap=cmap,marker='s',vmin=-max_abs_r,vmax=max_abs_r) #,vmin=-1,vmax=1
    for i in range(df.shape[0]-1):
        ax.axhline(y=0.5+i,c='gray',linewidth=0.5)
    for i in range(df.shape[1]-1):
        ax.axvline(x=0.5+i,c='gray',linewidth=0.5)

    ax.set_xlim(-0.5,df.shape[1]-0.5)
    ax.set_ylim(-0.5, df.shape[0] - 0.5)
    ax.set_xticks(np.arange(0,df.shape[1]),df.columns,rotation=90,ha='center')
    ax.set_yticks(np.arange(0, df.shape[0]),df.index[::-1])
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    cbar = plt.colorbar(sc)
    # cbar.set_label('R', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_fig)

def bio_enrich_plot(title,df:pd.DataFrame,y,x,size,color,size_log=False,ax=None,fig_path=None,
                    max_letter=120,min_letter=120,adjusted_p=True):
    matplotlib.rcParams['font.family'] = 'Arial'
    color_pale=['#E2614D','#FE9927','#4BAA52']
    ax_w=8
    ax_h=4
    def process_string(s,min_l,max_l):
        if len(s) > max_l:
            s = s[:max_l] + '...'
        elif len(s) < min_l:
            s = ' ' * (min_l - len(s)) + s
        return s
    df=df.copy()

    df[y]=df[y].apply(lambda x:process_string(x,min_letter,max_letter))
    base_size=150
    size_legend_eg=[0.1,0.5,0.9]
    df['y_idx']=np.arange(df.shape[0],0,-1)
    raw_max_size=df[size].max()
    if size_log:
        df[size]=df[size].map(lambda x:np.log2(x+1))
    max_size=df[size].max()
    df['size_norm']=df[size]*base_size/max_size
    if ax is None:
        fig, ax = plt.subplots(figsize=(ax_w,ax_h))
    k=0
    for category, group in df.groupby(color):
        ax.scatter(group[x], group['y_idx'], s=group['size_norm'], label=f"{category}", alpha=1, c=color_pale[k])
        k+=1
    ax.set_yticks(df['y_idx'])
    ax.set_yticklabels(df[y])
    legend_elements = [
        plt.scatter([], [], s=base_size*0.7, color=color_pale[k], label=category)
        for k, category in enumerate(df[color].unique())
    ]
    le1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    le1.set_title('Database')
    le1._legend_box.align = "left"
    le1.set_frame_on(False)

    p_text= r'$- \log_{10}(P)$'
    if adjusted_p:
        p_text= r'$- \log_{10}(\mathrm{Adjusted}\ P)$'
    ax.set_xlabel(p_text)
    ax2 = ax.twinx()
    for sl in size_legend_eg:
        show_eg=int(sl*max_size)
        eg=show_eg
        if size_log:
            show_eg=int(sl*raw_max_size//10*10)
            eg=np.log2(show_eg+1)
        ax2.scatter([], [], s=eg*base_size/max_size, label=f'{show_eg}', alpha=0.8, color='black')
    le2=ax2.legend(loc='upper left',bbox_to_anchor=(1, 0.64))
    le2.set_title('Count')
    le2._legend_box.align = "left"
    le2.set_frame_on(False)
    ax2.set_yticks([])
    ax.grid(axis='y', color='lightgrey', linestyle='--')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='lightgrey', linestyle='--')
    xmin, xmax = ax.get_xlim()
    new_xmin = xmin - 0.1 * (xmax - xmin)
    new_xmax = xmax + 0.1 * (xmax - xmin)
    ax.set_xlim(new_xmin, new_xmax)

    def replace_p_italic(text):
        pattern = re.compile(r'P-value', re.IGNORECASE)
        def repl(match):
            return r'$p$-value'
        new_text = pattern.sub(repl, text)
        return new_text
    ax.set_title(f'{title}',fontsize=16)
    plt.subplots_adjust(left=0.67, right=0.8, bottom=0.15, top=0.9)
    if fig_path is not None:
        make_dir(os.path.dirname(fig_path))
        plt.savefig(fig_path,dpi=300)
    # plt.show()


def go_enrich_plot(title,m_genes,ax,out_path,max_term=5,anno_dbs=['GO:BP','GO:CC','GO:MF']):
    annot_path=f'{out_path}.xlsx'
    if not os.path.isfile(annot_path):
        gp = GProfiler(return_dataframe=True)
        anno_df = gp.profile(organism='hsapiens', query=m_genes, sources=anno_dbs,
                             user_threshold=1)
        make_dir(os.path.dirname(out_path))
        anno_df.to_excel(annot_path, index=False)
    else:
        anno_df = pd.read_excel(annot_path)
    remain_idx=[]
    for cate,sdf in anno_df.groupby('source'):
        cut_term_n=max_term
        if cut_term_n > sdf.shape[0]:
            cut_term_n = sdf.shape[0]
        remain_idx+=sdf.sort_values(by=['p_value']).index[:cut_term_n].tolist()
    anno_df=anno_df.loc[remain_idx,]
    anno_df = anno_df.sort_values(by=['source','p_value'])
    anno_df['p_value']=-np.log10(anno_df['p_value'])
    bio_enrich_plot(title,anno_df,'name','p_value','intersection_size','source',ax=ax,fig_path=f'{out_path}.png')


def hex_to_rgba(hex_color, alpha):
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def extract_col(df,col_key,drop_na=False):
    m_cols = []
    for col in df.columns:
        if col.endswith(col_key):
            m_cols.append(col)
    asdf = df[m_cols]
    asdf.columns = [x.replace(f'{col_key}', '') for x in asdf.columns]
    if drop_na:
        asdf=asdf.dropna()
    return asdf

def generate_colormap(categories, cmap_name="husl"):
    unique_labels = sorted(categories.unique())
    mc=len(unique_labels)
    if mc<=20:
        color_map = seaborn.color_palette(cmap_name, n_colors=mc)
    else:
        color_map = seaborn.color_palette(cmap_name, n_colors=mc)+seaborn.color_palette("Set3", mc-20)
    row_palette = dict(zip(unique_labels,color_map))
    return categories.map(row_palette), row_palette

def cluster_df(cor,method='average', metric= 'euclidean'):
    Z = hierarchy.linkage(cor, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    x_order=x['leaves']
    Z = hierarchy.linkage(cor.T, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    y_order=x['leaves']
    cor=cor.iloc[x_order,y_order]
    return cor

def get_line_by_path(file_path,command=''):
    has_commond=False
    if command!='':
        has_commond=True
    isGzip = True
    try:
        if str(ft.guess(file_path).extension) == "gz":
            isGzip = True
    except:
        isGzip = False
    if isGzip:
        with gzip.open(file_path, "r") as reader:
            while True:
                line = reader.readline().decode().strip('\n')
                if has_commond and line.startswith(command):
                    continue
                yield line
    else:
        with open(file_path, "r") as reader:
            while True:
                line = reader.readline().strip('\n')
                if has_commond and line.startswith(command):
                    continue
                yield line

def get_path_by_prefix(path_prefix):
    wdir=os.path.dirname(path_prefix)
    for f in os.listdir(wdir):
        if f.startswith(path_prefix):
            return f'{wdir}/{f}'
    return None

def plot_high_risk_age_range_0(title,df, window_size, ax, age_col='age', score_col='score',
                             y_label="AUC",le_range_title='Highest AUC',smooth=False):
        auc_mean=df.loc[df[age_col]=='mean',score_col]
        df=df.loc[df[age_col]!='mean',:]
        df = df.dropna().copy()
        df = df.sort_values(by=age_col)
        min_age = int(df[age_col].min())
        max_age = int(df[age_col].max())
        bins = list(range(min_age, max_age - window_size+1))
        avg_risks = []
        for start_age in bins:
            end_age = start_age + window_size
            window_df = df[(df[age_col] >= start_age) & (df[age_col] <= end_age)]
            if len(window_df) > 0:
                avg_risks.append(window_df[score_col].mean())
            else:
                avg_risks.append(np.nan)

        avg_risks = np.array(avg_risks)
        max_idx = int(np.nanargmax(avg_risks))
        highlight_start = bins[max_idx]
        highlight_end = highlight_start + window_size

        sns.set_context("paper", font_scale=1.4)
        sns.set_style("white")
        if smooth:
            sns.regplot(x=age_col, y=score_col, data=df, lowess=True,
                        scatter=False, ax=ax, color='crimson', label='LOESS Smoothed')
        else:
            sns.lineplot(x=age_col, y=score_col, data=df, ax=ax, color="#4682B4")
            sns.scatterplot(data=df, x=age_col, y=score_col,
                            ax=ax, s=20, edgecolor='black', alpha=0.8, legend=False)
        ax.axvspan(
            highlight_start,
            highlight_end,
            color="#FFB347",
            alpha=0.3,
            label=f'{le_range_title}\n({highlight_start}–{highlight_end} y)'
        )
        # ax.
        ax.set_xlabel("Age (years)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title("", fontsize=13)
        ax.legend(frameon=False, fontsize=11)
        # ax.grid(False)
        # sns.despine(trim=True)
        ax.tick_params(length=4, width=1)
        ax.set_title(f"{title}", fontsize=14)
        # ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(-5,100)


def plot_high_risk_age_range(title, df, age_range, ax, age_col='age', score_col='score',
                             y_label="AUC", le_range_title='Diagnostic: ', smooth=False):
    auc_mean = df.loc[df[age_col] == 'mean', score_col]
    df = df.loc[df[age_col] != 'mean', :]
    df = df.dropna().copy()
    df = df.sort_values(by=age_col)
    min_age = int(df[age_col].min())
    max_age = int(df[age_col].max())
    # bins = list(range(min_age, max_age - window_size + 1))
    # avg_risks = []
    # for start_age in bins:
    #     end_age = start_age + window_size
    #     window_df = df[(df[age_col] >= start_age) & (df[age_col] <= end_age)]
    #     if len(window_df) > 0:
    #         avg_risks.append(window_df[score_col].mean())
    #     else:
    #         avg_risks.append(np.nan)

    # avg_risks = np.array(avg_risks)
    # max_idx = int(np.nanargmax(avg_risks))
    # highlight_start = bins[max_idx]
    # highlight_end = highlight_start + window_size
    highlight_start=age_range[0]
    highlight_end=age_range[1]

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("white")
    if smooth:
        sns.regplot(x=age_col, y=score_col, data=df, lowess=True,
                    scatter=False, ax=ax, color='crimson', label='LOESS Smoothed')
    else:
        sns.lineplot(x=age_col, y=score_col, data=df, ax=ax, color="#4682B4")
        sns.scatterplot(data=df, x=age_col, y=score_col,
                        ax=ax, s=20, edgecolor='black', alpha=0.8, legend=False)
    ax.axvspan(
        highlight_start,
        highlight_end,
        color="#FFB347",
        alpha=0.3,
        label=f'{le_range_title}: {highlight_start}–{highlight_end}'
    )
    # ax.
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    # ax.grid(False)
    # sns.despine(trim=True)
    ax.tick_params(length=4, width=1)
    ax.set_title(f"{title}", fontsize=14)
    # ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 100)


if __name__ == '__main__':
    # x=np.array([[1,2,3],[3,4,5]])
    # print(cpm(x,row_is_gene=False))
    print(remove_last_bracket('Tuft cell_Sh2d6 (d)high (Adult-Intestine)'))