import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump
import os

# Load synthetic data (cycle lengths)
df = pd.read_csv('../data/sample_cycles.csv')
X = df[['last_cycle_length', 'n_cycles', 'avg_cycle_length']].values
y = df['next_cycle_length'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"MAE: {mae:.2f} days")

os.makedirs('../models', exist_ok=True)
dump(model, '../models/baseline_model.pkl')
print("Saved to ../models/baseline_model.pkl")
