# train_local_global.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load features
df = pd.read_csv('/Users/megh/IdeaProjects/PaymentsML/features_global.csv')

# ----------------------------
# Auto-detect feature columns
# ----------------------------
# Keep all numeric engineered features + one-hot encoded dummies
feature_cols = [c for c in df.columns if c not in ['txn_id','debtor_id']]  

X = df[feature_cols].fillna(0)

# ----------------------------
# Train Isolation Forest
# ----------------------------
model = IsolationForest(
    n_estimators=200,       # number of trees
    contamination=0.02,     # expected anomaly rate
    random_state=42,        # reproducibility
    n_jobs=-1               # use all CPU cores
)
model.fit(X)

# ----------------------------
# Save model + feature list
# ----------------------------
joblib.dump(model, 'global_iforest.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

print('Saved local global model')
print(f'Features used for training: {len(feature_cols)}')
