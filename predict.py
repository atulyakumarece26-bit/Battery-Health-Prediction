import joblib

model = joblib.load('models/rf_model.joblib')

new_battery = [[
    3.8,   # voltage
    1.2,   # current
    30.0,  # temperature
    450,   # charge_cycles
    0.04   # internal_resistance
]]

result = model.predict(new_battery)
print(f"Battery SOH: {result[0]:.2f}%")

if result[0] > 80:
    print("Status: Battery is Healthy ✅")
elif result[0] > 60:
    print("Status: Battery Degrading ⚠️")
else:
    print("Status: Replace Battery ❌")