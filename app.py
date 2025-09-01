# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout='wide')
st.title('Payment Anomaly Detection — Demo')

# ----------------------------
# Load models (local POC)
# ----------------------------
global_model = joblib.load('/Users/megh/IdeaProjects/global_iforest.pkl')
trained_feature_cols = joblib.load('/Users/megh/IdeaProjects/feature_cols.pkl')
debtor_model = joblib.load('/Users/megh/IdeaProjects/debtor_iforest.pkl')

# ----------------------------
# Upload file
# ----------------------------
uploaded = st.file_uploader('Upload payments CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=['timestamp'])
    st.subheader('Sample rows')
    st.dataframe(df.head())

    # ----------------------------
    # Feature engineering (must match featurize.py)
    # ----------------------------
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['execution_time'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) else 12)
    df['debtor_txn_count'] = df.groupby('debtor_id')['txn_id'].transform('count')
    creditor_freq = df['creditor_account'].value_counts().to_dict()
    df['creditor_freq'] = df['creditor_account'].map(creditor_freq).fillna(0)

    # --- Global rarity features ---
    currency_freq = df['currency'].value_counts().to_dict()
    df['currency_freq'] = df['currency'].map(currency_freq)

    channel_freq = df['channel'].value_counts().to_dict()
    df['channel_freq'] = df['channel'].map(channel_freq)

    if 'country' in df.columns:
        cc_freq = df.groupby(['country','currency']).size().to_dict()
        df['country_currency_freq'] = df.apply(
            lambda r: cc_freq.get((r['country'], r['currency']), 0), axis=1)
    else:
        df['country_currency_freq'] = 0

    # --- Global and peer-group z-scores ---
    df['global_z_amt'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['chan_cur_z_amt'] = df.groupby(['channel','currency'])['amount'] \
                             .transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    # --- One-hot encoding ---
    df_encoded = pd.get_dummies(df, columns=["currency","channel"], prefix=["cur","ch"])

    # Align with training feature order
    X = df_encoded.reindex(columns=trained_feature_cols, fill_value=0)

    # ----------------------------
    # Global scoring
    # ----------------------------
    df['global_score'] = global_model.decision_function(X)
    df['global_flag'] = global_model.predict(X)  # -1 anomaly, 1 normal

    # ----------------------------
    # Debtor-level scoring
    # ----------------------------
    agg = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/debtor_profiles.csv')
    df = df.merge(agg, on='debtor_id', how='left')
    debtor_feats = df[['txn_count','avg_amt','std_amt','unique_payees','mobile_pct']].fillna(0)
    df['debtor_score'] = debtor_model.decision_function(debtor_feats)
    df['debtor_flag'] = debtor_model.predict(debtor_feats)

    # ----------------------------
    # Combine decisions
    # ----------------------------
    df['final_flag'] = df.apply(
        lambda r: 'ANOMALY' if (r['global_flag'] == -1 or r['debtor_flag'] == -1) else 'NORMAL',
        axis=1
    )

    # ----------------------------
    # Explainability
    # ----------------------------
    df['z_amt'] = (df['amount'] - df['avg_amt']) / (df['std_amt'].replace(0, 1))

    def reason_for(r):
        reasons = []
        if r['global_flag'] == -1:
            reasons.append('Global model outlier')
            if r['currency_freq'] < 5:
                reasons.append(f'Rare currency: {r["currency"]}')
            if r['channel_freq'] < 5:
                reasons.append(f'Rare channel: {r["channel"]}')
            if abs(r['global_z_amt']) > 3:
                reasons.append(f'Globally unusual amount (z={r["global_z_amt"]:.1f})')
            if abs(r['chan_cur_z_amt']) > 3:
                reasons.append(f'Unusual in {r["channel"]}/{r["currency"]} peer group')
        if r['debtor_flag'] == -1:
            reasons.append('Behavioural anomaly for debtor')
        if abs(r['z_amt']) > 3:
            reasons.append(f'Debtor amount deviation z={r["z_amt"]:.1f}')
        if pd.notnull(r.get('reason')) and r['reason'] != '':
            reasons.append(f'Injected reason: {r["reason"]}')
        return '; '.join(reasons) if reasons else 'No strong reasons'

    df['explain'] = df.apply(reason_for, axis=1)

    # ----------------------------
    # Show flagged anomalies
    # ----------------------------
    anomalies = df[df['final_flag'] == 'ANOMALY'].sort_values(by='global_score')
    st.subheader('Flagged transactions')
    st.dataframe(anomalies[['txn_id','debtor_id','amount','currency','channel',
                            'creditor_name','remittance_info','execution_time',
                            'final_flag','explain']].head(200))

    # ----------------------------
    # Dashboard
    # ----------------------------
    st.subheader('Dashboard')

    col1, col2 = st.columns(2)
    with col1:
        st.write('Final anomaly counts by channel')
        st.bar_chart(df.groupby('channel')['final_flag'].apply(lambda s: (s == 'ANOMALY').sum()))
    with col2:
        st.write('Top debtor anomalies (final)')
        st.bar_chart(df[df['final_flag'] == 'ANOMALY']['debtor_id'].value_counts().head(10))

    col3, col4 = st.columns(2)
    with col3:
        st.write('Global anomalies by channel')
        st.bar_chart(df[df['global_flag'] == -1]['channel'].value_counts())
    with col4:
        st.write('Debtor anomalies by debtor_id')
        st.bar_chart(df[df['debtor_flag'] == -1]['debtor_id'].value_counts().head(10))

    # ----------------------------
    # Download flagged results
    # ----------------------------
    st.download_button('Download flagged CSV', anomalies.to_csv(index=False), 'flagged.csv')
else:
    st.info('Upload a CSV file to evaluate anomalies.')
    st.stop()
# ----------------------------
# End of file
# ----------------------------

