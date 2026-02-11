Alzheimer's Disease Diagnosis Prediction

A machine learning project that predicts Alzheimer's disease diagnosis using patient demographic, lifestyle, medical, clinical, cognitive, and symptom data. The project compares **Logistic Regression** (via statsmodels GLM) with a **Bayesian-optimized XGBoost** classifier, achieving up to **94.9% accuracy** on the test set.

> Developed as part of **MGSC 661** coursework.

---

Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Key Libraries](#key-libraries)
- [License](#license)

---

Project Overview

Alzheimer's disease is a progressive neurodegenerative disorder, and early detection is critical for intervention and care planning. This project explores whether a combination of clinical assessments, demographic factors, lifestyle indicators, and symptom profiles can reliably predict an Alzheimer's diagnosis using machine learning.

The analysis follows three main stages:

1. **Exploratory Data Analysis** — distribution analysis of demographics, lifestyle factors, and the target variable.
2. **Hypothesis-Driven Logistic Regression** — systematic evaluation of feature groups (demographics, lifestyle, medical history, clinical measurements, cognitive assessments, and symptoms) using statsmodels GLM to understand which categories of predictors carry signal, along with interaction-term experiments.
3. **XGBoost with Bayesian Hyperparameter Optimization** — a gradient-boosted tree model tuned via `BayesSearchCV` for maximum predictive performance.

---

Dataset

The dataset (`alzheimers_disease_data.csv`) contains **2,149 patient records** with **35 features** spanning six categories:

| Category | Features |
|---|---|
| **Demographics** | Age, Gender, Ethnicity, EducationLevel |
| **Lifestyle** | BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality |
| **Medical History** | FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension |
| **Clinical Measurements** | SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides |
| **Cognitive & Functional** | MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL |
| **Symptoms** | Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness |

**Target variable:** `Diagnosis` (binary — 0: No Disease, 1: Disease)

The dataset has **no missing values** and the target distribution is approximately **65% No Disease / 35% Disease**.

---

Methodology

Exploratory Data Analysis

- Distribution plots for all feature groups (demographics, lifestyle, target).
- Correlation analysis against the `Diagnosis` target.

Logistic Regression (statsmodels GLM)

Multiple GLM models were fit to isolate the predictive power of each feature group:

- **Baseline model** — Cognitive & Functional features only (`MMSE`, `FunctionalAssessment`, `MemoryComplaints`, `BehavioralProblems`, `ADL`).
- **Demographic model** — Age, Gender, Ethnicity, EducationLevel.
- **Lifestyle model** — BMI, Smoking, Alcohol, Physical Activity, Diet, Sleep.
- **Medical History model** — Family history, cardiovascular disease, diabetes, depression, head injury, hypertension.
- **Clinical Measurements model** — Blood pressure, cholesterol panels.
- **Symptoms model** — Confusion, disorientation, personality changes, task difficulty, forgetfulness.
- **Interaction experiments** — Age × EducationLevel, SleepQuality × Age, MMSE × MemoryComplaints.

Scikit-learn Logistic Regression

A standard `LogisticRegression` model with `StandardScaler` preprocessing, evaluated with precision-recall curves and ROC-AUC analysis.

### XGBoost with Bayesian Optimization

- `BayesSearchCV` from `scikit-optimize` was used to tune 10 hyperparameters over 50 iterations with 3-fold cross-validation, optimizing for ROC-AUC.
- Tuned parameters include: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`, `reg_alpha`, `reg_lambda`, and `gamma`.

---

Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| GLM Baseline (Cognitive & Functional) | 0.845 | 0.739 | 0.805 | — |
| GLM + Age × Education Interaction | 0.846 | 0.747 | 0.803 | — |
| GLM + Interaction Experiments | 0.848 | 0.711 | 0.836 | — |
| **XGBoost (Bayesian-Optimized)** | **0.949** | **0.928** | **0.928** | **0.928** |

The XGBoost model significantly outperformed all logistic regression variants. Feature importance analysis (by gain) highlighted functional assessment scores, ADL, and MMSE as the top predictors.

---

Installation & Setup

Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost scikit-optimize bayesian-optimization ISLP
```

Run the Notebook

```bash
jupyter notebook MGSC_661_Code.ipynb
```

Make sure the dataset file `alzheimers_disease_data.csv` is in the same directory as the notebook.

---

Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies listed above.
3. Place `alzheimers_disease_data.csv` in the project root.
4. Open and run `MGSC_661_Code.ipynb` from top to bottom.

---

Repository Structure

```
├── MGSC_661_Code.ipynb        # Main analysis notebook
├── alzheimers_disease_data.csv # Dataset (not included — see Dataset section)
├── README.md                   # This file
```

---

Key Libraries

- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualization
- **scikit-learn** — Preprocessing, train-test split, evaluation metrics, logistic regression
- **statsmodels / ISLP** — GLM-based logistic regression with detailed statistical summaries
- **xgboost** — Gradient boosted tree classifier
- **scikit-optimize** — Bayesian hyperparameter optimization

---

License

This project is for academic purposes as part of the MGSC 661 course. Please refer to the dataset's original license for data usage terms.
