import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'voltage': np.random.uniform(3.0, 4.2, n),
    'current': np.random.uniform(0.5, 2.0, n),
    'temperature': np.random.uniform(20, 45, n),
    'charge_cycles': np.random.randint(1, 1000, n),
    'internal_resistance': np.random.uniform(0.01, 0.1, n),
})

data['SOH'] = (100
    - data['charge_cycles'] * 0.05
    - (data['temperature'] - 25) * 0.1
    + np.random.normal(0, 1, n)).clip(60, 100)

data.to_csv('data/battery_data.csv', index=False)
print("Dataset created successfully!")