import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/synthetic_payments_enriched.csv',parse_dates=['timestamp'])

# Basic numeric transforms
# log amount helps with scale
df['log_amount'] = np.log1p(df['amount'])

# hour of execution
df['hour'] = df['execution_time'].apply(lambda x: int(x.split(':')[0]))

# debtor frequency
df['debtor_txn_count'] = df.groupby('debtor_id')['txn_id'].transform('count')

# frequency encoding for creditor (captures repeat payments)
creditor_freq = df['creditor_account'].value_counts().to_dict()
df['creditor_freq'] = df['creditor_account'].map(creditor_freq)

# encode channel & currency via one-hot (small vocab)
df = pd.concat([df, pd.get_dummies(df['channel'], prefix='ch')], axis=1)
df = pd.concat([df, pd.get_dummies(df['currency'], prefix='cur')], axis=1)

# Feature selection for global model
feature_cols = ['log_amount', 'hour', 'debtor_txn_count', 'creditor_freq'] + [c for c in df.columns if c.startswith('ch_') or c.startswith('cur_')]
X = df[feature_cols].fillna(0)
X.to_csv('features_global.csv', index=False)
print('Wrote features for global model:', X.shape)

# Debtor profile aggregation (behavioural)
agg = df.groupby('debtor_id').agg(
txn_count=('txn_id','count'),
avg_amt=('amount','mean'),
median_amt=('amount','median'),
std_amt=('amount','std'),
unique_payees=('creditor_account','nunique'),
mobile_pct=('channel', lambda s: (s=='mobile').mean())).fillna(0).reset_index()
agg.to_csv('debtor_profiles.csv', index=False)
print('Wrote debtor profile table:', agg.shape)