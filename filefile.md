# ⚠️ Tech Layoff Early Warning System

A machine learning web application that predicts layoff risk for tech companies using financial signals and explainable AI.

**Live Demo:** [layoff-streamlit.onrender.com](https://layoff-streamlit.onrender.com)  
**API Docs:** [layoff-fastapi.onrender.com/docs](https://layoff-fastapi.onrender.com/docs)

---

## What It Does

A user types a company name and gets:
- A **layoff risk score** (0–100%)
- A **risk level** (HIGH / MEDIUM / LOW)
- **Top factors** driving the prediction (powered by SHAP)
- Financial data pulled live from SimFin

---

## Architecture

```
User → Streamlit UI → FastAPI → XGBoost Model → SHAP Explainer
                          ↓
                      SimFin API (live financials)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Deployment | Render |
| Financial Data | SimFin (quarterly financials) |
| Data Source | Layoffs.fyi dataset (Kaggle) |

---

## How It Works

### Data Pipeline
- **Target variable:** Layoffs.fyi dataset (2020–2024) — 2389 layoff events
- **Financial features:** SimFin quarterly financials — Revenue Growth, Profit Margin, Cash Reserves
- **Time-based labeling:** Quarters within 6 months before a layoff = label 1, earlier quarters = label 0
- **Final dataset:** 1301 rows across 147 public tech companies

### Model
- **Algorithm:** XGBoost binary classifier
- **Imbalance handling:** SMOTE oversampling
- **Evaluation:** F1: 0.54, ROC-AUC: 0.70
- **Explainability:** SHAP TreeExplainer — top features per prediction

### Feature Importance (SHAP)
```
1. Revenue Growth    ← most predictive signal
2. Cash Reserves
3. Profit Margin
4. Industry
5. Funds Raised
6. Stage
7. Country           ← least predictive
```

---

## API Endpoints

### `POST /predict`
Input a company name, get risk score + SHAP factors.

```json
// Request
{"company_name": "Google"}

// Response
{
  "company": "Google",
  "risk_score": 0.7813,
  "risk_level": "HIGH",
  "top_factors": [
    {"feature": "Cash_Reserves", "impact": 2.34, "direction": "increases risk"},
    {"feature": "Profit_Margin", "impact": 1.11, "direction": "decreases risk"},
    {"feature": "Revenue_Growth", "impact": 0.44, "direction": "increases risk"}
  ],
  "data_as_of": "2025-03-31",
  "message": "Prediction based on latest available financials (2025-03-31)"
}
```

### `POST /predict-batch`
Upload a CSV with `company_name` column, get predictions for all companies.

---

## Running Locally

### Prerequisites
- Python 3.12
- Docker Desktop

### Option 1 — Docker Compose (recommended)
```bash
git clone https://github.com/yourusername/early_layoff_warning.git
cd early_layoff_warning

# Add your SimFin API key to .env
cp .env.example .env

# Build and run all services
docker-compose up --build
```

Open:
- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5001`

### Option 2 — Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 - FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 2 - Streamlit
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
early_layoff_warning/
├── api/
│   └── main.py                  ← FastAPI endpoints
├── app/
│   └── streamlit_app.py         ← Streamlit frontend
├── data/
│   ├── layoffs_cleaned.csv      ← cleaned layoffs dataset
│   ├── master_dataset.csv       ← final training dataset
│   ├── ticker_mapping.csv       ← company → ticker mapping
│   └── simfin/                  ← cached SimFin data
├── models/
│   ├── xgb_best_model.pkl       ← trained XGBoost pipeline
│   ├── ordinal_encoder.pkl      ← fitted OrdinalEncoder
│   └── shap_explainer.pkl       ← SHAP TreeExplainer
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_experimenting.ipynb
│   └── 04_modeling.ipynb
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## Limitations

- Model trained on 1301 rows from 147 public tech companies — limited dataset
- Financial data from SimFin lags by 1-2 quarters
- Only works for companies in the layoffs dataset or with SimFin coverage
- ROC-AUC of 0.70 — model learns signal but has room for improvement
- Render free tier: first load takes 30-60 seconds due to cold starts

---

## Future Improvements

- [ ] Add FinBERT sentiment scoring on earnings call transcripts
- [ ] Expand dataset with more private company data from Crunchbase
- [ ] Retrain quarterly with fresh SimFin data
- [ ] Add GitHub Actions CI/CD pipeline for automated retraining
- [ ] Upgrade to Render paid tier for faster response times

---

## Author

**Yashwanth Kumar**  
MS Data Science — Webster University, Austin TX  
Graduating December 2026  
