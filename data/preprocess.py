import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/battery_data.csv')

print(df.head())
print(df.isnull().sum())

df.dropna(inplace=True)

df.to_csv('data/clean_data.csv', index=False)
print("Data cleaned and saved!")

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig('plots/heatmap.png')
plt.show()