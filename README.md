# Sediment Sound Speed Prediction – Feature Ablation Study
This repository contains the code and experimental framework for the paper:

> - **Mo, D., Hou, Z., Huang, J., Dong, F., Xiao, P., Xu, L., Zhao, B., & Li, Z.**  
>   *Sediment sound speed prediction: feature ablation and structural information analysis based on machine learning*  
>   *(Manuscript under review)*
>   
The code implements systematic feature ablation experiments (G1–G4) to quantify the contribution and redundancy of physical and grain‑size parameters in predicting sediment sound speed (100 kHz compressional wave velocity) using machine learning regression models (ElasticNet, SVR, Random Forest, XGBoost). All experiments are performed on a sediment dataset collected from the South China Sea

## 📁 Repository Structure
.<br>
├── data/                         # Put your dataset here (see Data Preparation)<br>
│   └── sediment_data.xlsx        # Example filename (actual name may differ)<br>
├── src/<br>
│   ├── data_loader.py            # Loading and preprocessing (outlier detection, split)<br>
│   ├── feature_sets.py           # Definition of all feature groups G1–G4<br>
│   ├── models.py                 # Wrappers for ElasticNet, SVR, RF, XGBoost<br>
│   ├── tuner.py                  # Optuna hyperparameter optimization with CV<br>
│   ├── evaluate.py               # Metrics (R², RMSE, MAE) and plotting<br>
│   └── run_experiments.py        # Main script to execute all experiments<br>
├── results/                      # Output directory for metrics, figures, and tables<br>
├── config.yaml                   # Global configuration (data path, random seed, etc.)<br>
├── requirements.txt              # Python dependencies with exact versions<br>
├── LICENSE<br>
└── README.md                     # This file<br>
