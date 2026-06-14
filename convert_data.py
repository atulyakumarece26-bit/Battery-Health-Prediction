import scipy.io as sio
import pandas as pd
import numpy as np

# Load the .mat file
mat = sio.loadmat('B0005.mat.mat')

rows = []

# Get battery data
battery = mat['B0005']
cycles = battery['cycle'][0][0]

for i in range(cycles.shape[1]):
    cycle = cycles[0, i]
    cycle_type = str(cycle['type'][0])
    
    if cycle_type == 'discharge':
        data = cycle['data'][0][0]
        try:
            voltage = float(np.mean(data['Voltage_measured'][0]))
            current = float(np.mean(data['Current_measured'][0]))
            temperature = float(np.mean(data['Temperature_measured'][0]))
            capacity = float(data['Capacity'][0][0])
            
            rows.append({
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'charge_cycles': len(rows) + 1,
                'internal_resistance': 0.05,
                'SOH': (capacity / 2.0) * 100
            })
        except:
            continue

df = pd.DataFrame(rows)
df.to_csv('data/battery_data.csv', index=False)
print("NASA dataset converted successfully!")
print(f"Total rows: {len(df)}")
print(df.head())