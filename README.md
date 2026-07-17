# Sediment Sound Speed Prediction – Feature Ablation Study
This repository contains the code and experimental framework for the paper:

> - **Mo, D., Hou, Z., Huang, J., Dong, F., Xiao, P., Xu, L., Zhao, B., & Li, Z.**  
>   *Sediment sound speed prediction: feature ablation and structural information analysis based on machine learning*  
>   *(Manuscript under review)*
>   
The code implements systematic feature ablation experiments (G1–G4) to quantify the contribution and redundancy of physical and grain‑size parameters in predicting sediment sound speed (100 kHz compressional wave velocity) using machine learning regression models (ElasticNet, SVR, Random Forest, XGBoost). All experiments are performed on a sediment dataset collected from the South China Sea

## Repository Structure

```text
- data/
  - sediment_data.xlsx
- src/
  - data_loader.py
  - feature_sets.py
  - models.py
  - tuner.py
  - evaluate.py
  - run_experiments.py
- results/
  - config.yaml
  - requirements.txt
- LICENSE
- README.md

# Put your dataset here (see Data Preparation)
# Example filename (actual name may differ)

# Loading and preprocessing (outlier detection, split)
# Definition of all feature groups G1-G4
# Wrappers for ElasticNet, SVR, RF, XGBoost
# Optuna hyperparameter optimization with CV
# Metrics (R², RMSE, MAE) and plotting
# Main script to execute all experiments
# Output directory for metrics, figures, and tables
# Global configuration (data path, random seed, etc.)
# Python dependencies with exact versions

# This file
```
