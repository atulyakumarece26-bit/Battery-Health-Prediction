import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model = joblib.load('models/rf_model.joblib')
df = pd.read_csv('data/clean_data.csv')

X = df[['voltage', 'current', 'temperature', 'charge_cycles', 'internal_resistance']]
y = df['SOH']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Graph 1 — Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual SOH')
plt.ylabel('Predicted SOH')
plt.title('Actual vs Predicted SOH')
plt.savefig('plots/actual_vs_predicted.png')
plt.show()

# Graph 2 — Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(X.columns, importances, color='steelblue')
plt.title('Feature Importance')
plt.savefig('plots/feature_importance.png')
plt.show()