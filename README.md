# Sediment Sound Speed Prediction – Feature Ablation Study
This repository contains the code and experimental framework for the paper:

> - **Mo, D., Hou, Z., Huang, J., Dong, F., Xiao, P., Xu, L., Zhao, B., & Li, Z.**  
>   *Sediment sound speed prediction: feature ablation and structural information analysis based on machine learning*  
>   *(Manuscript under review)*
>   
The code implements systematic feature ablation experiments (G1–G4) to quantify the contribution and redundancy of physical and grain‑size parameters in predicting sediment sound speed (100 kHz compressional wave velocity) using machine learning regression models (ElasticNet, SVR, Random Forest, XGBoost). All experiments are performed on a sediment dataset collected from the South China Sea

## 📁 Repository Structure
```text
├── data/                         # Put your dataset here (see Data Preparation)
│   └── sediment_data.xlsx        # Example filename (actual name may differ)
├── src/
│   ├── data_loader.py            # Loading and preprocessing (outlier detection, split)
│   ├── feature_sets.py           # Definition of all feature groups G1–G4
│   ├── models.py                 # Wrappers for ElasticNet, SVR, RF, XGBoost
│   ├── tuner.py                  # Optuna hyperparameter optimization with CV
│   ├── evaluate.py               # Metrics (R², RMSE, MAE) and plotting
│   └── run_experiments.py        # Main script to execute all experiments
├── results/                      # Output directory for metrics, figures, and tables
├── config.yaml                   # Global configuration (data path, random seed, etc.)
├── requirements.txt              # Python dependencies with exact versions
├── LICENSE
└── README.md                     # This file
```

## 🔧 Environment and Dependencies
All experiments were carried out on Windows 10 with an Intel Core i7-11700KF @ 3.60 GHz and 32 GB RAM. Python 3.12.10 (64‑bit) was used inside a virtual environment.

The required packages and their exact versions are listed below. We strongly recommend using these versions to ensure reproducible results.
| Package       | Version           |
| :------------ | :---------------- |
| pandas        | 2.3.3             |
| openpyxl      | 3.1.5             |
| numpy         | 2.3.4             |
| scikit-learn  | 1.7.2             |
| xgboost       | 3.3.0             |
| optuna        | 4.5.0             |
| matplotlib    | 3.10.7            |
| seaborn       | 0.13.2            |
| scipy         | latest compatible |
| joblib        | (bundled)         |
