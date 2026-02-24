# aml-idh-predict
Predicting isocitrate dehydrogenase mutation status in acute myeloid leukemia from gene expression profiles by machine learning

This repository accompanies the research article:

**“Predicting isocitrate dehydrogenase mutation status in acute myeloid leukemia from gene expression profiles by machine learning.”**

It provides harmonized transcriptomic data, mutation annotations, predicted labels, and Python scripts required to reproduce the analyses and train machine-learning models described in the study.

---

## 📌 Overview

Isocitrate dehydrogenase (IDH) mutations are key molecular events in acute myeloid leukemia (AML), but mutation annotations are often missing in public transcriptomic datasets.

This project presents a scalable computational framework that:

* Integrates and batch-corrects transcriptomic data from **19 AML cohorts (5,844 samples)**
* Trains machine-learning models (logistic regression and neural network) to predict IDH mutation status
* Reconstructs missing IDH annotations for **4,148 previously unannotated samples**
* Provides predicted labels and reproducible code for community reuse

The final logistic regression model achieves high discrimination performance and is used to generate the predicted annotations included in this repository.

---

## 📂 Repository Structure

```
├── data/
│   ├── expression_data_pooled_19_AML_datasets_pycombat_corrected.csv
│   ├── idh_mutation_status.csv
│   └── predicted_IDH_status_by_LR_for_19_AML_datasets.csv
│
├── scripts/
│   ├── grid_search_LR.py
│   ├── train_LR.py
│   ├── grid_search_NN.py
│   └── train_NN.py
│
└── README.md
```

---

## 📌 Data Description

### 1. Pooled Expression Dataset

**File:**
`expression_data_pooled_19_AML_datasets_pycombat_corrected.csv`

**Description:**
Batch-corrected gene expression matrix obtained by integrating 19 AML transcriptomic cohorts using **pyComBat**.

**Details:**

* Samples: **5,844**
* Genes: **9,870**
* Format: rows = samples, columns = genes
* Purpose: input features for machine-learning models

---

### 2. IDH Mutation Status Annotations

**File:**
`idh_mutation_status.csv`

**Description:**
Ground-truth IDH mutation labels for samples where annotations are available.

**Columns:**

* `idh_status` → categorical label

  * `IDH-WT`
  * `IDH-MUT`
  * `None` (unknown)
* `idh_mutant` → binary encoding

  * `0` = IDH-WT
  * `1` = IDH-MUT
  * `None` = unknown

**Coverage:**
Available for **1,696 out of 5,844 samples**

---

### 3. Predicted IDH Status

**File:**
`predicted_IDH_status_by_LR_for_19_AML_datasets.csv`

**Description:**
Final IDH status assignments produced by the logistic regression model.

**Labels:**

* `IDH-MUT` → confirmed mutant
* `IDH-WT` → confirmed wildtype
* `pIDH-MUT` → predicted mutant
* `pIDH-WT` → predicted wildtype

**Columns:**

* `Sample`
* `Dataset`
* `Known IDH status`
* `Predicted IDH status`

This file provides the reconstructed annotations for all 19 datasets.

---

### Table: Sample sizes of the AML datasets

| Dataset | Technology | Source | Total sample size | Samples with known IDH status |
|---------|------------|--------|-----------------|--------------------------------------------|
| AML-OHSU-2022 | RNA-seq | cBioPortal | 654 | 117 IDH-MUT (19.3%), 489 IDH-WT (80.7%) |
| BEATAML-1.0 | RNA-seq | GDC portal | 463 | – |
| GSE106291 | RNA-seq | NCBI GEO | 250 | 47 IDH-MUT (19.8%), 190 IDH-WT (80.2%) |
| GSE111678 | Microarrays | NCBI GEO | 260 | – |
| GSE1159 | Microarrays | NCBI GEO | 285 | – |
| GSE13159 | Microarrays | NCBI GEO | 542 | – |
| GSE146173 | RNA-seq | NCBI GEO | 246 | 50 IDH-MUT (20.3%), 196 IDH-WT (79.7%) |
| GSE165430 | RNA-seq | NCBI GEO | 268 | – |
| GSE17855 | Microarrays | NCBI GEO | 237 | – |
| GSE216738 | RNA-seq | NCBI GEO | 506 | – |
| GSE22845 | Microarrays | NCBI GEO | 154 | – |
| GSE232130 | RNA-seq | NCBI GEO | 362 | – |
| GSE253086 | RNA-seq | NCBI GEO | 136 | – |
| GSE297413 | RNA-seq | NCBI GEO | 264 | – |
| GSE37642 | Microarrays | NCBI GEO | 140 | – |
| GSE43176 | Microarrays | NCBI GEO | 104 | – |
| GSE61804 | Microarrays | NCBI GEO | 286 | – |
| GSE6891 | Microarrays | NCBI GEO | 536 | 71 IDH-MUT (15.5%), 386 IDH-WT (84.5%) |
| TCGA-LAML | RNA-seq | GDC portal | 151 | 29 IDH-MUT (19.3%), 121 IDH-WT (80.7%) |

---

## 📌 Machine Learning Models

Two supervised models are implemented:

### Logistic Regression (LR)

* Final selected model
* L2 regularization
* High interpretability and robustness
* Used for final predictions in this repository

### Neural Network (NN)

* Feed-forward multilayer perceptron
* Included for comparison and reproducibility

---

## 📌 Code Description

### Hyperparameter Optimization

#### `grid_search_LR.py`

Performs nested cross-validation grid search to identify optimal logistic regression hyperparameters.

#### `grid_search_NN.py`

Performs hyperparameter search for neural network architecture and training parameters.

---

### Model Training and Prediction

#### `train_LR.py`

* Trains the logistic regression model using selected hyperparameters
* Generates predictions for new samples

#### `train_NN.py`

* Trains the neural network model
* Outputs probability predictions and class labels

---

## 📌 Environment

**Python version:** 3.12

### Required packages
numpy, pandas, scikit-learn, torch (only for the NN model)

---

## 📖 Citation

If you use this repository, please cite the associated article.