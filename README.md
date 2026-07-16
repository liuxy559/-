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
