import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import shap
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("1. Generating Dataset...")
# Features mentioned in the video interface
features = ['max_speed', 'std_nonzero_speed', 'max_brake_pedal', 'avg_brake_pedal', 
            'brake_gt_2', 'max_accel', 'mean_pos_accel', 'mean_neg_accel', 
            'mean_jerk', 'std_abs_jerk', 'decel_time_pct', 'cpk_value']

# Generating 500 samples of synthetic data for demonstration
np.random.seed(42)
X_synthetic = np.random.randn(500, len(features))
# 0 = Non-Eco Driving, 1 = Eco Driving
y_synthetic = np.random.randint(0, 2, 500) 

df = pd.DataFrame(X_synthetic, columns=features)
df['target'] = y_synthetic

X = df.drop('target', axis=1)
y = df['target']

print("2. Preprocessing & MinMax Scaling...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print("3. Training Models (XGBoost & Gradient Boosting)...")
xgb = XGBClassifier(eval_metric='logloss')
gbc = GradientBoostingClassifier()

# Voting Classifier (Ensemble)
voting_clf = VotingClassifier(estimators=[('xgb', xgb), ('gbc', gbc)], voting='soft')
voting_clf.fit(X_train, y_train)

accuracy = voting_clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("4. Saving Model and Scaler...")
pickle.dump(voting_clf, open('models/best_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("5. Generating SHAP Explainable AI Graph (Novelty)...")
# SHAP requires a specific model, we'll use the trained XGBoost for explanation
xgb.fit(X_train, y_train)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
# This will save a plot you can use in your PPTs
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
import matplotlib.pyplot as plt
plt.savefig('static/shap_summary.png', bbox_inches='tight')
print("Model training complete. Files saved in /models folder.")