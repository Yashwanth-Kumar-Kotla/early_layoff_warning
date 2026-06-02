layoff-early-warning/
│
├── data/
│   ├── layoffs_raw.csv          ← downloaded from Kaggle (Week 1)
│   ├── transcripts_raw/         ← earnings call transcripts folder (Week 2)
│   └── master_dataset.csv       ← final combined dataset you'll build (Week 1 end)
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Week 1: explore all 3 data sources
│   ├── 02_nlp_sentiment.ipynb   ← Week 2: run FinBERT on transcripts
│   ├── 03_feature_engineering.ipynb  ← Week 3: build feature matrix
│   └── 04_modeling.ipynb        ← Week 3: XGBoost + SHAP
│
├── src/
│   ├── __init__.py              ← makes src a Python package
│   ├── data_pipeline.py         ← pulls yfinance data, merges sources
│   ├── nlp_pipeline.py          ← FinBERT sentiment scoring
│   ├── train.py                 ← model training script
│   └── predict.py               ← prediction logic (used by FastAPI)
│
├── api/
│   └── main.py                  ← FastAPI app (Week 4)
│
├── app/
│   └── streamlit_app.py         ← Streamlit frontend (Week 4)
│
├── models/
│   └── xgb_model.pkl            ← saved trained model (Week 3 output)
│
├── mlflow_artifacts/            ← MLflow stores runs here (Week 4)
│
├── .github/
│   └── workflows/
│       └── ci.yml               ← GitHub Actions pipeline (Week 5)
│
├── docker-compose.yml           ← 3 services: FastAPI + Streamlit + MLflow (Week 5)
├── requirements.txt             ← all pip dependencies
├── .env.example                 ← template for API keys (Kaggle, etc.)
├── .gitignore                   ← excludes data files, models, venv
└── README.md                    ← your portfolio write-up