import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/clean_data.csv')

X = df[['voltage', 'current', 'temperature', 'charge_cycles', 'internal_resistance']]
y = df['SOH']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'models/rf_model.joblib')
print("Model trained and saved!")