# aild-metabolites
Multicenter serum metabolome and interpretable machine learning analysis for autoimmune liver disease (AIH, PBC, overlap syndrome) diagnosis and PBC staging

# Project Overview
This repository contains all analytical code supporting our multicenter cohort study published in Journal of Hepatology.
We systematically characterized serum glycerophospholipid/choline metabolic axis reprogramming across three AILD subtypes (AIH, PBC, overlap syndrome [OS]) and Ludwig PBC disease stages, then constructed integrated metabolomic-clinical explainable machine learning models for non-invasive AILD differential diagnosis and PBC progression staging.
Study core design:
9-center discovery cohort (n=722) + independent 8-center external validation cohort
Untargeted metabolomics: 273 non-lipid metabolites + 2133 lipid species quantification
Multi-dimensional stratified screening for core differential metabolites
KEGG pathway enrichment analysis of stage/subtype-specific metabolic signatures
Interpretable ML models (Logistic Regression, LASSO, Random Forest) with SHAP & LIME feature interpretation
Cross-center performance validation & overfitting evaluation

# Repository Content
├── 01_AILDg_diagnosis      # Four ML algorithms, 5-fold cross-validation, train/test cohort split, visualization for aild diagnosis
├── 02_PBC_staging/         # Four ML algorithms, 5-fold cross-validation, train/test cohort split, visualization for pbc staging
├── utils/                  # Shared custom plotting & statistical functions
├── demo_data/              # De-identified simulated sample data (no real patient clinical data)
├── environment.yml         # Conda environment configuration for reproducibility
└── README.md

# Dependencies 
conda env create -f environment.yml
conda activate aild-meta

# Data Statement
All raw clinical and metabolomic data involved in this multicenter study contain sensitive patient information and cannot be fully open-sourced in compliance with hospital ethics committee requirements.
Simulated de-identified demo datasets are provided in /demo_data/ for code reproduction testing.
Qualified researchers can contact the corresponding author to obtain standardized analysis scripts and apply for original data access via formal ethical application.

# Reproducibility Notes
All statistical thresholds, screening criteria (P < 0.05, |log₂FC| > 1.5) and cohort splitting rules strictly follow the manuscript Methods section.
5-fold cross-validation and independent multicenter external validation codes are included to evaluate model overfitting and generalization.
The pipeline automatically outputs AUC, 95%CI, confusion matrix and SHAP feature ranking results consistent with the original paper.

# Citation
If you use this code or analytical pipeline in your research, please cite our work:

# License
MIT License — Free for academic non-commercial research use.
