# 🧠 Analysis Code for Age-Specific Genetic Mechanisms of Neuropsychiatric Disorders Using the tDESE Framework

This repository contains code for analyzing the **age-dependent genetic mechanisms** of neuropsychiatric disorders using the **tDESE framework**.  
The **core implementation of tDESE** is available on the [KGGSUM platform](https://pmglab.top/kggsum), where users can explore and apply the method directly.  
This repository provides the analysis and visualization code used in our study.

---

## 🧹 Developmental Expression Data Processing

- **`preprocess/preprocess_gene_expr.py`**  
  🧪 Preprocesses developmental dynamic expression data from PsychENCODE, including quality control and extraction of covariates.

- **`preprocess/predict_age_expr_by_gam.py`**  
  📈 Fits gene expression trajectories across age using a generalized additive model (GAM), with sex and RNA quality as covariates, and predicts the age-related component of expression at each year.

- **`preprocess/predict_age_expr_by_gam_by_sex.py`**  
  ⚧ Performs GAM fitting and prediction separately for males and females.

---

## 🔗 Integrating GWAS and Developmental Expression to Infer Risk Windows

- **`run/run_disease_age.py`**  
  🧬 Integrates GWAS summary statistics with developmental dynamic expression data to infer age windows of elevated genetic risk for each disorder.

---

## 📊 Analysis and Visualization

- **`analyze/analyze_cell_age.py`**  
  🔍 Analyzes and visualizes developmental expression data, including sample size distribution, expression-based sample correlation, and age-related expression trends.

- **`analyze/analyze_disease_age.py`**  
  🕒 Identifies and visualizes disease high-risk age windows, determines peak age-specific risk genes, and explores their biological functions.

- **`analyze/analyze_disease_age_by_sex.py`**  
  🚻 Performs sex-stratified analysis of disease high-risk age windows.

- **`analyze/analyze_disease_gene.py`**  
  🎯 Evaluates the effectiveness of using age-specific risk genes (from high-risk windows) for prioritizing disease-associated genes **compared to other developmental ages**.

---

## ⚙️ Notes
- The repository is structured into three main modules: `preprocess/`, `run/`, and `analyze/`, corresponding to data preparation, disease risk inference, and downstream analysis.  
- All scripts are written in Python and require installation of standard scientific computing libraries (e.g., `numpy`, `scipy`, `pandas`, `statsmodels`, `matplotlib`, `seaborn`).

---

## 📖 Citation
If you use this code or framework in your research, please cite our work accordingly once published.
