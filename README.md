# Sediment Sound Speed Prediction via Machine Learning with Feature Ablation Analysis

[![Python](https://img.shields.io/badge/Python-3.12.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxxxx-blue)](https://doi.org/10.5281/zenodo.xxxxxxxx)

This repository contains the official implementation of the paper:

> **"Sediment Sound Speed Prediction: Feature Ablation and Structural Contrast Experiments Based on Machine Learning"**  
> *Dinghao Mo, Zhengyu Hou, Jingyi Huang, Fang Dong, Peng Xiao, Lingji Xu, Bo Zhao, Zhenglin Li*  
> Under review at *Ocean Engineering*

---

## 📖 Overview

Seafloor sediment compressional wave velocity is a critical parameter in marine acoustics and underwater engineering. This study systematically examines the functional roles and redundancy relationships among different physical features in sediment sound speed prediction using multiple machine learning algorithms.

**Key contributions of this work:**
- Quantitatively evaluates redundancy between porosity and wet bulk density
- Compares complete grain-size fraction distributions vs. statistical descriptors (Mz, Md)
- Identifies asymmetric response of grain-size information under progressive compression
- Demonstrates predictive stability under feature dimensionality reduction

Four regression algorithms are implemented and compared:
- **ElasticNet** — linear model with L1/L2 regularization
- **Support Vector Regression (SVR)** — kernel-based nonlinear regression
- **Random Forest (RF)** — bagging ensemble
- **XGBoost** — boosting ensemble

---

## 📁 Repository Structure

.
├── ElasticNet-truth.py # ElasticNet model training & evaluation
├── RF-truth.py # Random Forest model
├── SVR-truth.py # Support Vector Regression model
├── XGboost-truth.py # XGBoost model
├── requirements.txt # Python dependencies (exact versions)
├── LICENSE # MIT License
└── README.md # This file


---

## 💻 System Configuration

All experiments were conducted under the following environment:

| Category | Item | Configuration / Version |
|----------|------|-------------------------|
| **Operating System** | Windows 10 | Windows 10 |
| **Hardware** | Workstation | Intel Core i7-11700KF @ 3.60GHz, 32GB RAM |
| **Runtime Language** | Python | 3.12.10 (64-bit virtual environment) |
| **Data Processing** | pandas | v2.3.3 |
| | openpyxl | v3.1.5 |
| | numpy | v2.3.4 |
| **Machine Learning** | scikit-learn | v1.7.2 |
| | xgboost | v3.3.0 |
| | tabpfn | Stable version compatible with Python 3.12 |
| **Hyperparameter Tuning** | optuna | v4.5.0 |
| **Visualization & Statistics** | matplotlib | v3.10.7 |
| | seaborn | v0.13.2 |
| | scipy | Compatible stable version |
| | joblib | Bundled with scikit-learn v1.7.2 |

> **Note:** `tabpfn` is included for compatibility but not used in the main experiments of this study.

---

## 🔧 Requirements

To replicate the exact environment, use the following `requirements.txt`:

```txt
numpy==2.3.4
pandas==2.3.3
openpyxl==3.1.5
scikit-learn==1.7.2
xgboost==3.3.0
optuna==4.5.0
matplotlib==3.10.7
seaborn==0.13.2
scipy
joblib

Install via:
pip install -r requirements.txt

Alternatively, if you use conda:
conda create -n sed_speed python=3.12.10
conda activate sed_speed
pip install -r requirements.txt

Features
Feature	Description	Unit
ρ	Wet bulk density	g/cm³
n	Porosity	—
G	Gravel content	%
S	Sand content	%
T	Silt content	%
Y	Clay content	%
Mz	Mean grain size	φ
Md	Median grain size	φ
Target
Target	Description	Unit
Vs	Compressional wave velocity	m/s
Data Availability
The dataset supporting the findings of this study is available from the corresponding author upon reasonable request. Due to institutional data policies, the raw data are not publicly hosted in this repository.

