# Sediment Sound Speed Prediction – Feature Ablation Study
This repository contains the code and experimental framework for the paper:

> - **Mo, D., Hou, Z., Huang, J., Dong, F., Xiao, P., Xu, L., Zhao, B., & Li, Z.**  
>   *Sediment sound speed prediction: feature ablation and structural information analysis based on machine learning*  
>   *(Manuscript under review)*
>   
The code implements systematic feature ablation experiments (G1–G4) to quantify the contribution and redundancy of physical and grain‑size parameters in predicting sediment sound speed (100 kHz compressional wave velocity) using machine learning regression models (ElasticNet, SVR, Random Forest, XGBoost). All experiments are performed on a sediment dataset collected from the South China Sea

## Environment and Dependencies
All experiments were performed under the following environment:
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

To replicate the environment, create a new Python 3.12 virtual environment and install the required packages:
```bash
pip install pandas==2.3.3 openpyxl==3.1.5 numpy==2.3.4 scikit-learn==1.7.2 xgboost==3.3.0 optuna==4.5.0 matplotlib==3.10.7 seaborn==0.13.2 scipy
```
## Data
The dataset comprises 835 valid sediment core samples from the South China Sea, each with:

- **Target variable**: Compressional wave velocity at 100 kHz (` vs `)
- **Input features**: wet bulk density (` rho `), porosity (` n `), gravel (` g `), sand (` s `), silt (` t `), clay (` y `), mean grain size (` mz `), median grain size (` md `).

All data are provided in an Excel file ( `scs_sediment_data.xlsx` ). The sheet contains the cleaned records after outlier removal (DFFITS diagnostic).

## Outputs
- Test‑set metrics for all models and feature groups.
- Radar‑style bar plots and joint distribution plots.
- Optional CSV summary of 10×5‑fold CV statistics and paired t‑test results.

## License
This code is released for academic and non‑commercial use only. Please contact the corresponding author for further permissions.

## Contact
For questions regarding the code or data, please contact:
- Dinghao Mo: mohd5@mail.sysu.edu.cn
- Zhengyu Hou: zyhou2022@163.com
