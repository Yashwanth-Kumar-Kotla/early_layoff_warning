import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import io

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title='Layoff Early Warning System',
    page_icon='⚠️',
    layout='wide'
)

# ── API URL ────────────────────────────────────────────
import os
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# ── Styling ────────────────────────────────────────────
st.markdown("""
    <style>
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffaa00;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .risk-low {
        background-color: #00cc44;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────
st.title('⚠️ Tech Layoff Early Warning System')
st.markdown('Predict layoff risk for any tech company using financial signals and ML.')
st.divider()

# ── Tabs ───────────────────────────────────────────────
tab1, tab2 = st.tabs(['🔍 Single Company', '📊 Batch Prediction'])

with tab1:
    st.subheader('Company Risk Analysis')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name = st.text_input(
            'Enter company name',
            placeholder='e.g. Google, Microsoft, Amazon...'
        )
    
    with col2:
        st.write('')  # spacing
        st.write('')  # spacing
        predict_btn = st.button('Predict Risk', type='primary', use_container_width=True)
    
    if predict_btn and company_name:
        with st.spinner('Analyzing...'):
            try:
                response = requests.post(
                    f'{API_URL}/predict',
                    json={'company_name': company_name}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.divider()
                    
                    # ── Risk Score Display ──────────────────
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label='Risk Score',
                            value=f"{data['risk_score'] * 100:.1f}%"
                        )
                    
                    with col2:
                        risk_level = data['risk_level']
                        css_class = f"risk-{risk_level.lower()}"
                        st.markdown(
                            f'<div class="{css_class}">{risk_level} RISK</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.metric(
                            label='Data As Of',
                            value=data['data_as_of']
                        )
                    
                    st.divider()
                    
                    # ── SHAP Factors Chart ──────────────────
                    st.subheader('Top Risk Factors')
                    
                    factors = data['top_factors']
                    features = [f['feature'] for f in factors]
                    impacts = [f['impact'] for f in factors]
                    directions = [f['direction'] for f in factors]
                    colors = ['red' if d == 'increases risk' else 'green' for d in directions]
                    
                    fig = go.Figure(go.Bar(
                        x=impacts,
                        y=features,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{'+' if d == 'increases risk' else '-'} {v:.3f}" 
                              for v, d in zip(impacts, directions)],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title=f'SHAP Feature Impact for {company_name}',
                        xaxis_title='Impact on Risk Score',
                        yaxis_title='Feature',
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"ℹ️ {data['message']}")
                
                elif response.status_code == 404:
                    st.error(f"Company '{company_name}' not found. Try exact name from dataset.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")

with tab2:
    st.subheader('Batch Company Risk Analysis')
    st.markdown('Upload a CSV file with a `company_name` column to get predictions for multiple companies.')
    
    # ── Sample CSV download ─────────────────────────────
    sample_df = pd.DataFrame({
        'company_name': ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta']
    })
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        label='Download Sample CSV',
        data=sample_csv,
        file_name='sample_companies.csv',
        mime='text/csv'
    )
    
    st.divider()
    
    # ── File upload ─────────────────────────────────────
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
    
    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_preview)} companies:")
        st.dataframe(df_preview.head(), use_container_width=True)
        
        if st.button('Predict All', type='primary'):
            with st.spinner(f'Predicting for {len(df_preview)} companies...'):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    response = requests.post(
                        f'{API_URL}/predict-batch',
                        files={'file': ('companies.csv', uploaded_file, 'text/csv')}
                    )
                    
                    if response.status_code == 200:
                        results = response.json()['predictions']
                        df_results = pd.DataFrame(results)
                        
                        st.divider()
                        st.subheader(f'Results — {len(df_results)} companies')
                        
                        # ── Summary metrics ─────────────────
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            high = len(df_results[df_results['risk_level'] == 'HIGH'])
                            st.metric('HIGH Risk', high, delta=None)
                        with col2:
                            medium = len(df_results[df_results['risk_level'] == 'MEDIUM'])
                            st.metric('MEDIUM Risk', medium)
                        with col3:
                            low = len(df_results[df_results['risk_level'] == 'LOW'])
                            st.metric('LOW Risk', low)
                        
                        st.divider()
                        
                        # ── Color coded results table ────────
                        def color_risk(val):
                            if val == 'HIGH':
                                return 'background-color: #ffcccc'
                            elif val == 'MEDIUM':
                                return 'background-color: #fff3cc'
                            elif val == 'LOW':
                                return 'background-color: #ccffcc'
                            return ''
                        
                        styled_df = df_results.style.applymap(
                            color_risk, subset=['risk_level']
                        )
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # ── Download results ─────────────────
                        csv_output = df_results.to_csv(index=False)
                        st.download_button(
                            label='Download Results CSV',
                            data=csv_output,
                            file_name='layoff_predictions.csv',
                            mime='text/csv'
                        )
                        
                    else:
                        st.error(f"Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")