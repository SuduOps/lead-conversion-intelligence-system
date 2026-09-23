# 🎯 Lead Conversion Intelligence System

> A beginner-friendly Python Streamlit data analytics capstone project that analyses historical marketing leads, identifies the key factors associated with conversion, predicts which leads are most likely to convert, and provides practical recommendations for sales teams.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Data Source** | X Education Lead Scoring Dataset |
| **File Location** | `data/leads.csv` |
| **Target Variable** | `Converted` (1 = converted, 0 = not converted) |
| **Overall Conversion Rate** | ~38.5% |
| **Dataset Size** | 9,240 rows × 37 columns |
| **Tech Stack** | Python · Streamlit · Pandas · Plotly · scikit-learn |

---

## 🗂️ Project Structure

```
Lead_Conversion_Intelligence/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Files to exclude from version control
│
└── data/
    └── leads.csv           # Original dataset (DO NOT MODIFY)
```

---

## ✨ Features

### 📋 Page 1 — Data Quality & Overview
- Displays raw dataset shape, column types, missing values, duplicate count, and descriptive statistics
- Shows cleaned dataset statistics side-by-side with the raw data
- Explains every data-cleaning decision in plain English
- Original `data/leads.csv` is **never modified**

### 📊 Page 2 — Descriptive Analytics *(What happened?)*
- Overall conversion split (pie chart)
- Conversion rate by **Lead Origin**, **Lead Source**, **Last Activity**, **Occupation**, and **City**
- Distribution charts for **TotalVisits**, **Total Time Spent on Website**, and **Page Views Per Visit**

### 🔍 Page 3 — Diagnostic Analytics *(Why did it happen?)*
- Side-by-side box plots comparing converted vs non-converted leads across all numeric features
- Grouped bar charts for each categorical feature
- Auto-generated written business insights highlighting statistically meaningful differences

### 🤖 Page 4 — Predictive Analytics *(What is likely to happen?)*
- Trains **Logistic Regression** and **Random Forest** classifiers via a scikit-learn `Pipeline`
- Full preprocessing pipeline with median imputation (numeric) and OHE with unknown handling (categorical)
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix and ROC curve visualisations for each model
- Feature importance (Random Forest) and coefficient chart (Logistic Regression)
- Best model auto-selected by ROC-AUC

### 🎯 Page 5 — Prescriptive Analytics *(What should we do?)*
- Every lead is scored with the best model's predicted conversion probability
- Lead categories assigned automatically:
  - 🔥 **Hot Lead** — probability ≥ 70%
  - 🌡️ **Warm Lead** — probability 40–69%
  - 🥶 **Cold Lead** — probability < 40%
- Colour-coded actionable recommendations for each category
- Downloadable **Cleaned Dataset CSV** and **Lead Priority Table CSV**

### 🔧 Sidebar Filters
The sidebar provides interactive filters for:
- Lead Source
- Lead Origin
- Last Activity
- Occupation
- City

KPI cards (Total Leads, Converted, Non-Converted, Conversion Rate) update in real time based on selected filters.

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/lead-conversion-intelligence.git
cd lead-conversion-intelligence
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.32 | Web dashboard framework |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| plotly | ≥ 5.18 | Interactive visualisations |
| scikit-learn | ≥ 1.3 | ML pipelines and models |
| matplotlib | ≥ 3.7 | Supporting visualisations |
| seaborn | ≥ 0.12 | Statistical plots |

---

## 🗃️ Dataset Description

The dataset is the **X Education Lead Scoring** dataset, publicly available on Kaggle.

Key columns used in this project:

| Column | Description |
|--------|-------------|
| `Converted` | **Target** — 1 if lead converted, 0 otherwise |
| `Lead Source` | Marketing channel that brought the lead |
| `Lead Origin` | How the lead entered the system (API, form, etc.) |
| `Last Activity` | The most recent action taken by the lead |
| `TotalVisits` | Number of times the lead visited the website |
| `Total Time Spent on Website` | Cumulative minutes spent on site |
| `Page Views Per Visit` | Average pages viewed per session |
| `What is your current occupation` | Lead's professional background |
| `City` | Lead's city |

---

## 📸 Screenshots

> _Add screenshots here after running the app._

| Page | Screenshot |
|------|-----------|
| Data Overview | `screenshots/01_data_overview.png` |
| Descriptive Analytics | `screenshots/02_descriptive.png` |
| Diagnostic Analytics | `screenshots/03_diagnostic.png` |
| Predictive Analytics | `screenshots/04_predictive.png` |
| Prescriptive Analytics | `screenshots/05_prescriptive.png` |

---

## 🎓 Academic Notes

- All data cleaning is performed on an **in-memory copy** of the dataset. The original `data/leads.csv` is never overwritten.
- The ML pipeline uses `ColumnTransformer` + `Pipeline` to prevent data leakage during cross-validation or train/test splitting.
- The `handle_unknown="ignore"` option in `OneHotEncoder` ensures the app never crashes on unseen category values.
- Model selection is automatic — the model with the higher ROC-AUC on the held-out 20% test set is labelled as "best" and used for scoring in the Prescriptive tab.

---

## 📄 License

This project is submitted as a student capstone. Dataset credit: X Education / Kaggle Lead Scoring Dataset.
