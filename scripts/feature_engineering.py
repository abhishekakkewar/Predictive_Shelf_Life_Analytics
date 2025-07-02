import pandas as pd
import numpy as np
import os

# Load the synthetic data
df = pd.read_csv('data/synthetic_shelf_life_data.csv')

# --- Feature Engineering Steps ---

# 1. Product Age (days since manufacturing)
df['Manufacturing_Date'] = pd.to_datetime(df['Manufacturing_Date'])
df['Product_Age'] = (pd.Timestamp('2023-07-01') - df['Manufacturing_Date']).dt.days

# 2. One-hot encoding for Product
df = pd.get_dummies(df, columns=['Product'], prefix='Product')

# 3. Interaction feature: Temp x Humidity
df['Temp_Humidity_Interaction'] = df['Storage_Temperature'] * df['Storage_Humidity']

# 4. (Optional) Scaling numeric features (not applied here, but can be added for some models)

# Save processed data
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'processed_shelf_life_data.csv')
df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}") 