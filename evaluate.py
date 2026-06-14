import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

model = joblib.load('models/rf_model.joblib')
df = pd.read_csv('data/clean_data.csv')

X = df[['voltage', 'current', 'temperature', 'charge_cycles', 'internal_resistance']]
y = df['SOH']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

print(f"MAE  : {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")