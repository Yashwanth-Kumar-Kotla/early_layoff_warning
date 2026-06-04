# 📡 Layoff Radar
### Predict tech company layoffs before they happen

> Predict layoff risk for any tech company before it happens — powered by XGBoost, SHAP explainability, and live financial data.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://layoff-streamlit.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://layoff-fastapi.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)](https://docker.com)

---

## What Is This?

A machine learning system that answers one question:

> **"Is this tech company likely to lay off employees in the next 6 months?"**

Type a company name → get a risk score, a risk level, and an explanation of exactly which financial signals are driving the prediction.

Built as an end-to-end ML portfolio project — from raw data collection to a live deployed web app.

---

## Demo

| Single Company Prediction | Batch Prediction |
|---|---|
| Type any company name | Upload a CSV of companies |
| Get risk score + HIGH/MEDIUM/LOW | Get predictions for all at once |
| See top SHAP factors driving risk | Download results as CSV |

**Try it:** [layoff-streamlit.onrender.com](https://layoff-streamlit.onrender.com)

> ⚠️ Hosted on Render free tier — first load may take 30–60 seconds to wake up.

---

## How It Works

```
User types company name
         ↓
FastAPI looks up company in layoffs dataset
         ↓
Pulls latest quarterly financials from SimFin
         ↓
XGBoost model predicts layoff probability
         ↓
SHAP explains which features drove the score
         ↓
Returns: risk score + risk level + top 3 factors
```

---

## Architecture

```
┌─────────────────┐     HTTP POST      ┌──────────────────────┐
│  Streamlit UI   │ ────────────────▶  │   FastAPI Backend    │
│  (Port 8501)    │ ◀────────────────  │   (Port 8000)        │
└─────────────────┘     JSON Response  └──────────┬───────────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                       XGBoost Model         SimFin API          SHAP Explainer
                       (xgb_best_model.pkl)  (live financials)   (shap_explainer.pkl)
```

---

## ML Pipeline

### Data Sources
| Source | What it provides | Size |
|---|---|---|
| Layoffs.fyi (Kaggle) | Target variable — which companies laid off and when | 2,389 events (2020–2024) |
| SimFin API | Quarterly financials — Revenue, Net Income, Cash | 47,760 quarterly records |

### Feature Engineering
The key insight: **every row in the layoffs dataset is already a layoff event (label 1).** To create label 0 examples without introducing fake data, I used **time-based labeling**:

```
Company laid off in Q3 2022:
  Q1 2021 financials → label 0  (healthy, layoff not imminent)
  Q2 2021 financials → label 0  (healthy)
  Q3 2021 financials → label 0  (healthy)
  Q1 2022 financials → label 1  (within 6 months of layoff)
  Q2 2022 financials → label 1  (within 6 months of layoff)
```

This generates real negatives from the same companies — no structural mismatch between classes.

**Final dataset:** 1,301 rows across 147 public tech companies

### Features Used
| Feature | Source | Importance |
|---|---|---|
| Revenue Growth (QoQ) | SimFin | 🔴 Critical |
| Cash Reserves | SimFin | 🔴 Critical |
| Profit Margin | SimFin | 🟠 High |
| Industry | Layoffs dataset | 🟠 High |
| Funds Raised | Layoffs dataset | 🟡 Medium |
| Funding Stage | Layoffs dataset | 🟡 Medium |
| Country | Layoffs dataset | 🟢 Low |

### Model
```
Algorithm:       XGBoost Binary Classifier
Imbalance:       SMOTE oversampling
Hyperparameters: Tuned via GridSearchCV (243 combinations, 5-fold CV)
Best params:     learning_rate=0.05, max_depth=4, n_estimators=300

Evaluation (held-out test set):
  ROC-AUC:   0.70
  F1 Score:  0.54
  Recall:    0.52 (catches 52% of actual layoffs)
  Accuracy:  70%
```

### Explainability — SHAP
Every prediction comes with a SHAP breakdown:
```
Base score:          0.45 (average)
Revenue declining  → +0.18 (increases risk)
Low cash reserves  → +0.12 (increases risk)
High profit margin → -0.09 (decreases risk)
─────────────────────────────────────────
Final score:         0.66 → MEDIUM RISK
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | XGBoost + imbalanced-learn | Binary classification |
| Explainability | SHAP TreeExplainer | Per-prediction explanations |
| Experiment Tracking | MLflow | Parameter + metric logging |
| Financial Data | SimFin API | Live quarterly financials |
| API | FastAPI + Uvicorn | REST endpoints |
| Frontend | Streamlit + Plotly | Interactive web UI |
| Containerization | Docker + Docker Compose | 3-service deployment |
| Deployment | Render | Cloud hosting |
| Version Control | GitHub | CI/CD via auto-deploy |

---

## API Reference

### `POST /predict`
Predict layoff risk for a single company.

**Request:**
```json
{
  "company_name": "Google"
}
```

**Response:**
```json
{
  "company": "Google",
  "risk_score": 0.7813,
  "risk_level": "HIGH",
  "top_factors": [
    {
      "feature": "Cash_Reserves",
      "impact": 2.34,
      "direction": "increases risk"
    },
    {
      "feature": "Profit_Margin",
      "impact": 1.11,
      "direction": "decreases risk"
    },
    {
      "feature": "Revenue_Growth",
      "impact": 0.44,
      "direction": "increases risk"
    }
  ],
  "data_as_of": "2025-03-31",
  "message": "Prediction based on latest available financials (2025-03-31)"
}
```

### `POST /predict-batch`
Upload a CSV with a `company_name` column. Returns predictions for all companies.

**Full API documentation:** [layoff-fastapi.onrender.com/docs](https://layoff-fastapi.onrender.com/docs)

---

## Run Locally

### Option 1 — Docker (recommended)
```bash
# Clone the repo
git clone https://github.com/Yashwanth-Kumar-Kotla/early_layoff_warning.git
cd early_layoff_warning

# Set up environment variables
cp .env.example .env
# Add your SimFin API key to .env

# Build and run all 3 services
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

### Option 2 — Local Python
```bash
pip install -r requirements.txt

# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
early_layoff_warning/
│
├── api/
│   └── main.py                    ← FastAPI app — /predict, /predict-batch
│
├── app/
│   └── streamlit_app.py           ← Streamlit frontend
│
├── data/
│   ├── layoffs_cleaned.csv        ← Cleaned layoffs dataset
│   ├── master_dataset.csv         ← Final training dataset (1,301 rows)
│   ├── ticker_mapping.csv         ← Company name → ticker symbol
│   └── simfin/                    ← Cached SimFin quarterly data
│
├── models/
│   ├── xgb_best_model.pkl         ← Trained XGBoost pipeline
│   ├── ordinal_encoder.pkl        ← Fitted OrdinalEncoder
│   └── shap_explainer.pkl         ← SHAP TreeExplainer
│
├── notebooks/
│   ├── 01_eda.ipynb               ← Exploratory data analysis
│   ├── 02_feature_engineering.ipynb ← SimFin integration, time-based labeling
│   ├── 03_experimenting.ipynb     ← Model experiments, GridSearchCV
│   └── 04_modeling.ipynb          ← Final model, SHAP, MLflow
│
├── Dockerfile.api                 ← FastAPI container
├── Dockerfile.streamlit           ← Streamlit container
├── docker-compose.yml             ← Orchestrates all 3 services
├── requirements.txt
└── .env.example                   ← Environment variable template
```

---

## Limitations & Honest Notes

- **Dataset size:** 1,301 training rows, limited by public company SimFin coverage
- **Data freshness:** SimFin free tier lags ~1–2 quarters behind real time
- **Coverage:** Works best for US public tech companies in the layoffs dataset
- **Model performance:** ROC-AUC 0.70 trained on 1,301 rows,  
  performance scales with data coverage. Current version focuses on 
  public US tech companies; expanding to private company financials 
  is on the roadmap.
- **Cold starts:** Render free tier sleeps after 15 min inactivity — first request takes 30–60s

---

## Roadmap

- [ ] **v2.0** — Add FinBERT sentiment scoring on earnings call transcripts as a feature
- [ ] **v2.1** — Expand to private company data via Crunchbase API
- [ ] **v2.2** — Automated quarterly retraining pipeline via GitHub Actions
- [ ] **v3.0** — Real-time SimFin data refresh + alert system

---

## Author

**Yashwanth Kumar Kotla**
MS Data Science — Webster University, Austin TX 
Graduating December 2026


[GitHub](https://github.com/Yashwanth-Kumar-Kotla) · [LinkedIn](https://www.linkedin.com/in/yashwanthkumarkotla/)

---

*Built from scratch — data collection, feature engineering, model training, API development, containerization, and cloud deployment.*
