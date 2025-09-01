# train_debtor_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
agg = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/debtor_profiles.csv')
features = ['txn_count','avg_amt','std_amt','unique_payees','mobile_pct']
clf = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
clf.fit(agg[features])
joblib.dump(clf, 'debtor_iforest.pkl')
print('Saved debtor-level model')