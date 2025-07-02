import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the synthetic data
df = pd.read_csv('data/synthetic_shelf_life_data.csv')

# Create a directory for EDA outputs if it doesn't exist
eda_dir = 'data/eda_outputs'
os.makedirs(eda_dir, exist_ok=True)

# 1. Basic info and summary statistics
print('Data shape:', df.shape)
print('\nData types:\n', df.dtypes)
print('\nMissing values:\n', df.isnull().sum())
print('\nSummary statistics:\n', df.describe(include='all'))

# 2. Distribution plots
for col in ['Initial_Shelf_Life', 'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit', 'Remaining_Shelf_Life']:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'{eda_dir}/{col}_distribution.png')
    plt.close()

# 3. Boxplots by Product
for col in ['Initial_Shelf_Life', 'Remaining_Shelf_Life']:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Product', y=col, data=df)
    plt.title(f'{col} by Product')
    plt.savefig(f'{eda_dir}/{col}_by_product.png')
    plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(8,6))
corr = df[['Initial_Shelf_Life', 'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit', 'Remaining_Shelf_Life']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig(f'{eda_dir}/correlation_heatmap.png')
plt.close()

print(f'EDA outputs (plots) saved to {eda_dir}') 