import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/synthetic_payments_enriched.csv',
                 parse_dates=['timestamp'])

# ----------------------------
# Basic numeric transforms
# ----------------------------
# log amount helps with scale
df['log_amount'] = np.log1p(df['amount'])

# hour of execution
df['hour'] = df['execution_time'].apply(lambda x: int(str(x).split(':')[0]))

# debtor frequency
df['debtor_txn_count'] = df.groupby('debtor_id')['txn_id'].transform('count')

# frequency encoding for creditor (captures repeat payments)
creditor_freq = df['creditor_account'].value_counts().to_dict()
df['creditor_freq'] = df['creditor_account'].map(creditor_freq)

# ----------------------------
# Global rarity features
# ----------------------------
# Currency frequency (global)
currency_freq = df['currency'].value_counts().to_dict()
df['currency_freq'] = df['currency'].map(currency_freq)

# Channel frequency (global)
channel_freq = df['channel'].value_counts().to_dict()
df['channel_freq'] = df['channel'].map(channel_freq)

# Country+Currency frequency (joint rarity)
if 'country' in df.columns:
    cc_freq = df.groupby(['country','currency']).size().to_dict()
    df['country_currency_freq'] = df.apply(
        lambda r: cc_freq.get((r['country'], r['currency']), 0), axis=1)
else:
    df['country_currency_freq'] = 0  # fallback if no country column

# ----------------------------
# Global z-score for amount
# ----------------------------
df['global_z_amt'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()

# Channel+Currency peer-group z-score for amount
df['chan_cur_z_amt'] = df.groupby(['channel','currency'])['amount'] \
                         .transform(lambda x: (x - x.mean()) / x.std(ddof=0))

# ----------------------------
# One-hot encoding (fixed vocab recommended for production)
# ----------------------------
df = pd.concat([df, pd.get_dummies(df['channel'], prefix='ch')], axis=1)
df = pd.concat([df, pd.get_dummies(df['currency'], prefix='cur')], axis=1)

# ----------------------------
# Feature selection for global model
# ----------------------------
feature_cols = [
    'log_amount', 'hour', 'debtor_txn_count', 'creditor_freq',
    'currency_freq', 'channel_freq', 'country_currency_freq',
    'global_z_amt', 'chan_cur_z_amt'
] + [c for c in df.columns if c.startswith('ch_') or c.startswith('cur_')]

X = df[feature_cols].fillna(0)
X.to_csv('features_global.csv', index=False)
print('Wrote features for global model:', X.shape)

# ----------------------------
# Debtor profile aggregation (behavioural features)
# ----------------------------
agg = df.groupby('debtor_id').agg(
    txn_count=('txn_id','count'),
    avg_amt=('amount','mean'),
    median_amt=('amount','median'),
    std_amt=('amount','std'),
    unique_payees=('creditor_account','nunique'),
    mobile_pct=('channel', lambda s: (s=='mobile').mean())
).fillna(0).reset_index()

agg.to_csv('debtor_profiles.csv', index=False)
print('Wrote debtor profile table:', agg.shape)
