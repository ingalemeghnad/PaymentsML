# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
st.set_page_config(layout='wide')
st.title('Payment Anomaly Detection — Demo')
# Load models (local POC)
global_model = joblib.load('/Users/megh/IdeaProjects/global_iforest.pkl')
debtor_model = joblib.load('/Users/megh/IdeaProjects/debtor_iforest.pkl')
uploaded = st.file_uploader('Upload payments CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['timestamp'])
    st.subheader('Sample rows')
    st.dataframe(df.head())
    
    # Feature engineering inline (same transforms used for training)
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['execution_time'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 12)
    df['debtor_txn_count'] = df.groupby('debtor_id')['txn_id'].transform('count')
    creditor_freq = df['creditor_account'].value_counts().to_dict()
    df['creditor_freq'] = df['creditor_account'].map(creditor_freq).fillna(0)

    # simple one-hot for channel/currency
    df = pd.concat([df, pd.get_dummies(df['channel'], prefix='ch')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['currency'], prefix='cur')],    axis=1)
    feature_cols = ['log_amount', 'hour', 'debtor_txn_count',    'creditor_freq'] + [c for c in df.columns if c.startswith('ch_') or    c.startswith('cur_')]
    X = df[feature_cols].fillna(0)

    # Global scoring
    df['global_score'] = global_model.decision_function(X)
    df['global_flag'] = global_model.predict(X) # -1 anomaly, 1 normal
    
    # Debtor-level scoring: map debtor aggregates, score with debtor_model

    agg = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/debtor_profiles.csv')
    df = df.merge(agg, on='debtor_id', how='left')
    debtor_feats = df[['txn_count','avg_amt','std_amt','unique_payees','mobile_pct']].fillna(0)
    df['debtor_score'] = debtor_model.decision_function(debtor_feats)
    df['debtor_flag'] = debtor_model.predict(debtor_feats)

    # Combine decisions
    df['final_flag'] = df.apply(lambda r: 'ANOMALY' if (r['global_flag']==-1
    or r['debtor_flag']==-1) else 'NORMAL', axis=1)
    
    # Build explainability: z-score on amount
    df['z_amt'] = (df['amount'] - df['avg_amt']) / (df['std_amt'].replace(0, 1))
    def reason_for(r):
        reasons = []
        if r['global_flag']==-1:
            reasons.append('Global model outlier')
        if r['debtor_flag']==-1:
            reasons.append('Behavioural anomaly for debtor')
        if abs(r['z_amt'])>3:
            reasons.append(f'Amount deviation z={r["z_amt"]:.1f}')
        if pd.notnull(r.get('reason')) and r['reason']!='':
            reasons.append(f'Injected reason: {r["reason"]}')
        return '; '.join(reasons) if reasons else 'No strong reasons'
    
    df['explain'] = df.apply(reason_for, axis=1)
    # Show flagged
    anomalies =    df[df['final_flag']=='ANOMALY'].sort_values(by='global_score')
    st.subheader('Flagged transactions')
    st.dataframe(anomalies[['txn_id','debtor_id','amount','currency','channel','creditor_name','remittance_info','execution_time','final_flag','explain']].head(200))

    # Charts
    st.subheader('Dashboard')
    col1, col2 = st.columns(2)
    with col1:
        st.write('Anomaly counts by channel')
        st.bar_chart(df.groupby('channel')['final_flag'].apply(lambda s:
        (s=='ANOMALY').sum()))
    with col2:
        st.write('Top debtor anomalies')
        st.bar_chart(df[df['final_flag']=='ANOMALY']['debtor_id'].value_counts().head(10))
    
    # Allow download of flagged CSV
    st.download_button('Download flagged CSV', anomalies.to_csv(index=False),'flagged.csv')    