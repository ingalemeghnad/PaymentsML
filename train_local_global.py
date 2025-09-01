# train_local_global.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
X = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/features_global.csv')
model = IsolationForest(n_estimators=200, contamination=0.02,
random_state=42)
model.fit(X)
joblib.dump(model, 'global_iforest.pkl')
print('Saved local global model')