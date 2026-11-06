# PySpark MLlib Pipeline for Predicting At-Risk Secondary School Students

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-orange.svg)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, reproducible source code, experiment pipelines, hyperparameters, and fairness evaluation scripts for the research paper:

> **"Comparative Analysis of Machine Learning Algorithms Using PySpark MLlib for Predicting At-Risk Senior High School Students"**  
> **Authors:** Nicholas P. Gabato Jr. (*University of Northwestern Mindanao*) & James Cloyd M. Bustillo (*Agusan del Sur State College of Agriculture and Technology*)

---

## 📌 Research Overview

Early identification of at-risk students allows educational institutions to provide targeted academic interventions before course failure occurs. This study develops a scalable, reproducible predictive analytics pipeline using **Apache Spark MLlib (PySpark)** comparing three core classifiers:
1. **Logistic Regression (LR)**
2. **Random Forest (RF)**
3. **Gradient-Boosted Trees (GBT)**

The pipeline is benchmarked on secondary school student records from the public **UCI Machine Learning Repository** (Cortez & Silva, 2008), covering both Portuguese language ($N = 649$) and Mathematics ($N = 395$) cohorts.

### Key Contributions & Features
- **Scalable Big Data Framework:** Fully native PySpark ML pipeline (StringIndexer, OneHotEncoder, VectorAssembler, MLlib Classifiers).
- **Class Imbalance Mitigation:** Inverse-frequency class weighting applied directly to the training loss objective.
- **Robust Evaluation:** 5 repeated, approximately stratified random seeds (`42, 101, 2024, 7, 99`) reporting Accuracy, Precision, Recall, F1-Score, and ROC-AUC with standard deviations.
- **Early-Warning Ablation:** Evaluation with late-term grades (G2) withheld to measure early-warning viability.
- **Fairness & Disparity Audit:** Subgroup false-negative and false-positive error audit across student sex ($F$ vs. $M$) and school ($GP$ vs. $MS$).

---

## 📁 Repository Structure

```text
pyspark-at-risk-secondary/
│
├── pyspark_pipeline.py              # Main PySpark evaluation script across 5 seeds (Portuguese & Math)
├── math_replication_and_fairness.py # Domain replication & demographic subgroup fairness audit script
├── student_at_risk.ipynb            # Google Colab / Jupyter Notebook for exploration & training
│
├── figures/                         # ROC curves, confusion matrices, and feature importance plots
│   ├── figure1_roc_curves.png
│   ├── figure2_confusion_matrices.png
│   ├── figure3_feature_importance.png
│   └── figure4_early_warning_comparison.png
│
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE                          # MIT License
└── README.md                        # Documentation & reproduction guide
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- **Python:** 3.10 or 3.11 recommended
- **Java:** OpenJDK 8, 11, or 17 (required for Apache Spark)

### 2. Install Dependencies
```bash
git clone https://github.com/nicholasgabato/pyspark-at-risk-secondary.git
cd pyspark-at-risk-secondary
pip install -r requirements.txt
```

---

## 📊 Dataset Access

The study utilizes the **UCI Student Performance Dataset**:
- **Source:** [UCI Machine Learning Repository (Dataset #320)](https://archive.ics.uci.edu/dataset/320)
- **Reference:** Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.*
- **Target Definition:** Final grade $G3 < 10$ is labeled as `At-Risk` (positive class = 1.0); $G3 \ge 10$ is labeled as `Not At-Risk` (negative class = 0.0).

To run the scripts, download `student-mat.csv` and `student-por.csv` from the UCI repository and place them in `student+performance/student/`.

---

## 🚀 Running the Experiments

### 1. Main PySpark Pipeline
Runs the 5-seed repeated stratified training and evaluation for Logistic Regression, Random Forest, and Gradient-Boosted Trees under both full ($G1 + G2$) and reduced-grade (no $G2$) settings:
```bash
python pyspark_pipeline.py
```

### 2. Replication & Fairness Audit
Runs domain validation on the Mathematics cohort and outputs sex- and school-disaggregated error rates (FNR and FPR):
```bash
python math_replication_and_fairness.py
```

### 3. Google Colab / Jupyter Notebook
Launch `student_at_risk.ipynb` on Google Colab or your local Jupyter environment for interactive step-by-step execution and visualization generation.

---

## 📈 Summary of Empirical Results

### Main Configuration ($G1, G2$ Included)
| Cohort | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Portuguese** ($N=649$) | Logistic Regression | $87.72 \pm 2.13$ | $58.50 \pm 4.39$ | $83.66 \pm 7.69$ | $68.67 \pm 3.94$ | $0.9527 \pm 0.0215$ |
| | **Random Forest** | $\mathbf{91.63 \pm 3.04}$ | $\mathbf{70.41 \pm 8.80}$ | $83.98 \pm 10.34$ | $76.41 \pm 8.23$ | $\mathbf{0.9682 \pm 0.0168}$ |
| | **Gradient-Boosted Trees** | $91.42 \pm 3.13$ | $69.25 \pm 7.74$ | $\mathbf{87.79 \pm 7.74}$ | $\mathbf{77.08 \pm 5.20}$ | $0.9674 \pm 0.0258$ |
| **Mathematics** ($N=395$) | Logistic Regression | $90.17 \pm 2.79$ | $80.60 \pm 5.51$ | $94.95 \pm 3.90$ | $87.10 \pm 3.87$ | $0.9697 \pm 0.0182$ |
| | **Random Forest** | $\mathbf{92.16 \pm 2.19}$ | $\mathbf{84.40 \pm 3.24}$ | $\mathbf{95.30 \pm 3.75}$ | $\mathbf{89.49 \pm 2.99}$ | $0.9647 \pm 0.0260$ |
| | **Gradient-Boosted Trees** | $90.79 \pm 2.32$ | $83.64 \pm 5.27$ | $91.92 \pm 4.91$ | $87.45 \pm 3.34$ | $\mathbf{0.9710 \pm 0.0209}$ |

---

## 📜 Citation

If you use this repository or its findings in your academic research, please cite:

```bibtex
@inproceedings{gabato2026comparative,
  author    = {Nicholas P. Gabato Jr. and James Cloyd M. Bustillo},
  title     = {Comparative Analysis of Machine Learning Algorithms Using PySpark MLlib for Predicting At-Risk Senior High School Students},
  year      = {2026},
  url       = {https://github.com/nicholasgabato/pyspark-at-risk-secondary}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

