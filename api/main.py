import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import io
import pickle
import numpy as np
import pandas as pd
import shap
import simfin as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Load models ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'models/xgb_best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models/ordinal_encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, 'models/shap_explainer.pkl'), 'rb') as f:
    explainer = pickle.load(f)

# ── Load lookup data ───────────────────────────────────
ticker_mapping = pd.read_csv(os.path.join(BASE_DIR, 'data/ticker_mapping.csv'))
layoffs_df = pd.read_csv(os.path.join(BASE_DIR, 'data/layoffs_cleaned.csv'))

# ── Load SimFin ────────────────────────────────────────
sf.set_api_key(os.getenv('SIMFIN_API_KEY'))
sf.set_data_dir(os.path.join(BASE_DIR, 'data/simfin/'))

df_income = sf.load_income(variant='quarterly', market='us').reset_index()
df_balance = sf.load_balance(variant='quarterly', market='us').reset_index()

# ── App ────────────────────────────────────────────────
app = FastAPI(title='Layoff Early Warning System')

print("All models and data loaded successfully")

class PredictRequest(BaseModel):
    company_name : str

class RiskFactor(BaseModel):
    feature : str
    impact : float
    direction : str

class PredictResponse(BaseModel):

    company : str
    risk_score : float
    risk_level : str
    top_factors : list[RiskFactor]
    data_as_of : str
    message : str


# ── Helper Functions ───────────────────────────────────

def get_ticker(company_name: str):
    """Map company name to ticker symbol"""
    match = ticker_mapping[
        ticker_mapping['Company'].str.lower() == company_name.lower()
    ]
    if match.empty:
        return None
    return match.iloc[0]['Ticker']


def get_simfin_features(ticker: str):
    """Pull latest quarter financials from SimFin"""
    try:
        # Get income data for ticker
        income = df_income[df_income['Ticker'] == ticker].sort_values(
            'Report Date', ascending=False
        )
        balance = df_balance[df_balance['Ticker'] == ticker].sort_values(
            'Report Date', ascending=False
        )

        if income.empty:
            return None

        # Latest quarter
        latest = income.iloc[0]
        prev = income.iloc[1] if len(income) > 1 else None

        # Calculate features
        revenue = latest['Revenue']
        net_income = latest['Net Income']
        profit_margin = net_income / revenue if revenue else None

        # Revenue growth vs previous quarter
        if prev is not None and prev['Revenue']:
            revenue_growth = (revenue - prev['Revenue']) / prev['Revenue']
        else:
            revenue_growth = None

        # Cash reserves from balance sheet
        cash_col = 'Cash, Cash Equivalents & Short Term Investments'
        if not balance.empty and cash_col in balance.columns:
            cash_reserves = balance.iloc[0][cash_col]
        else:
            cash_reserves = None

        data_as_of = str(latest['Report Date'])[:10]

        return {
            'Cash_Reserves': cash_reserves,
            'Profit_Margin': profit_margin,
            'Revenue_Growth': revenue_growth,
            'data_as_of': data_as_of
        }

    except Exception:
        return None


def get_company_info(company_name: str):
    """Get Industry, Stage, Country, Funds_Raised from layoffs dataset"""
    match = layoffs_df[
        layoffs_df['Company'].str.lower() == company_name.lower()
    ]
    if match.empty:
        return None

    row = match.iloc[0]
    return {
        'Industry': row['Industry'],
        'Stage': row['Stage'],
        'Country': row['Country'],
        'Funds_Raised': row['Funds_Raised'] if pd.notna(row['Funds_Raised']) else 0.0
    }


def get_risk_level(score: float):
    """Convert probability score to risk level"""
    if score >= 0.7:
        return 'HIGH'
    elif score >= 0.4:
        return 'MEDIUM'
    else:
        return 'LOW'


def get_shap_factors(features_df: pd.DataFrame):
    """Get top 3 SHAP factors for a prediction"""
    # Get the XGBoost model from pipeline
    xgb_model = model.named_steps['model']
    
    # Calculate SHAP values
    shap_vals = explainer.shap_values(features_df)
    
    # Build factor list
    factors = []
    feature_names = features_df.columns.tolist()
    
    for i, name in enumerate(feature_names):
        factors.append({
            'feature': name,
            'impact': abs(float(shap_vals[0][i])),
            'direction': 'increases risk' if shap_vals[0][i] > 0 else 'decreases risk'
        })
    
    # Sort by impact, return top 3
    factors = sorted(factors, key=lambda x: x['impact'], reverse=True)[:3]
    return factors

# ── Endpoints ──────────────────────────────────────────

@app.get('/')
def home():
    return {'message': 'Layoff Early Warning System API', 'status': 'running'}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    company_name = request.company_name.strip()

    # Step 1 - Get company info from layoffs dataset
    company_info = get_company_info(company_name)
    if company_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{company_name}' not found in dataset. Try exact name."
        )

    # Step 2 - Get ticker
    ticker = get_ticker(company_name)

    # Step 3 - Get financial features from SimFin
    fin_features = None
    data_as_of = 'N/A'

    if ticker:
        fin_features = get_simfin_features(ticker)
        if fin_features:
            data_as_of = fin_features.pop('data_as_of')

    # Step 4 - Build feature row
    # Use SimFin values if available, else use dataset medians
    features = {
        'Industry': company_info['Industry'],
        'Stage': company_info['Stage'],
        'Country': company_info['Country'],
        'Funds_Raised': company_info['Funds_Raised'],
        'Cash_Reserves': fin_features['Cash_Reserves'] if fin_features else 3523200000.0,
        'Profit_Margin': fin_features['Profit_Margin'] if fin_features else 0.101995,
        'Revenue_Growth': fin_features['Revenue_Growth'] if fin_features else 0.018416
    }
    late_stages = ['Series G', 'Series H', 'Series I', 'Series J', 'Subsidiary', 'Unknown']
    features['Stage'] = 'Late Stage' if features['Stage'] in late_stages else features['Stage']

    features_df = pd.DataFrame([features])

    # Step 5 - Encode categorical columns
    cat_cols = ['Industry', 'Stage', 'Country']
    features_df[cat_cols] = encoder.transform(features_df[cat_cols])

    # Step 6 - Predict
    risk_score = float(model.predict_proba(features_df)[0][1])
    risk_level = get_risk_level(risk_score)

    # Step 7 - SHAP factors
    top_factors = get_shap_factors(features_df)
    risk_factors = [RiskFactor(**f) for f in top_factors]

    return PredictResponse(
        company=company_name,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        top_factors=risk_factors,
        data_as_of=data_as_of,
        message=f"Prediction based on latest available financials ({data_as_of})"
    )



@app.post('/predict-batch')
def predict_batch(file: UploadFile = File(...)):
    # Read uploaded CSV
    contents = file.file.read()
    df_input = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    if 'company_name' not in df_input.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must have a 'company_name' column"
        )

    results = []
    for company_name in df_input['company_name']:
        try:
            # Reuse same logic as /predict
            company_info = get_company_info(company_name)
            if company_info is None:
                results.append({
                    'company': company_name,
                    'risk_score': None,
                    'risk_level': 'UNKNOWN',
                    'message': 'Company not found in dataset'
                })
                continue

            ticker = get_ticker(company_name)
            fin_features = None
            data_as_of = 'N/A'

            if ticker:
                fin_features = get_simfin_features(ticker)
                if fin_features:
                    data_as_of = fin_features.pop('data_as_of')

            features = {
                'Industry': company_info['Industry'],
                'Stage': company_info['Stage'],
                'Country': company_info['Country'],
                'Funds_Raised': company_info['Funds_Raised'],
                'Cash_Reserves': fin_features['Cash_Reserves'] if fin_features else 3523200000.0,
                'Profit_Margin': fin_features['Profit_Margin'] if fin_features else 0.101995,
                'Revenue_Growth': fin_features['Revenue_Growth'] if fin_features else 0.018416
            }

            # Clean Stage - replace Unknown with Late Stage to match training data
            late_stages = ['Series G', 'Series H', 'Series I', 'Series J', 'Subsidiary', 'Unknown']
            features['Stage'] = 'Late Stage' if features['Stage'] in late_stages else features['Stage']

            features_df = pd.DataFrame([features])
            cat_cols = ['Industry', 'Stage', 'Country']
            features_df[cat_cols] = encoder.transform(features_df[cat_cols])

            risk_score = float(model.predict_proba(features_df)[0][1])
            risk_level = get_risk_level(risk_score)

            results.append({
                'company': company_name,
                'risk_score': round(risk_score, 4),
                'risk_level': risk_level,
                'data_as_of': data_as_of,
                'message': 'Success'
            })

        except Exception as e:
            results.append({
                'company': company_name,
                'risk_score': None,
                'risk_level': 'ERROR',
                'message': str(e)
            })

    return {'predictions': results, 'total': len(results)}

@app.get('/debug')
def debug():
    return {
        'encoder_categories': {
            'Industry': encoder.categories_[0].tolist(),
            'Stage': encoder.categories_[1].tolist(),
            'Country': encoder.categories_[2].tolist()
        },
        'layoffs_rows': len(layoffs_df),
        'ticker_mapping_rows': len(ticker_mapping)
    }