import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define number of samples and products
n_samples = 1000
products = ['Yogurt', 'Milk', 'Cheese', 'Bread', 'Juice']

# Generate synthetic data
data = {
    'Product': np.random.choice(products, n_samples),
    'Manufacturing_Date': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 180, n_samples), unit='D'),
    'Initial_Shelf_Life': np.random.randint(7, 60, n_samples),  # days
    'Storage_Temperature': np.random.normal(8, 3, n_samples).round(1),  # °C
    'Storage_Humidity': np.random.normal(60, 10, n_samples).round(1),   # %
    'Days_in_Transit': np.random.randint(1, 10, n_samples)
}

df = pd.DataFrame(data)

def calc_remaining_shelf_life(row):
    temp_factor = max(0, 1 - 0.05 * (row['Storage_Temperature'] - 8))
    humidity_factor = max(0, 1 - 0.01 * (row['Storage_Humidity'] - 60))
    transit_factor = max(0, 1 - 0.07 * (row['Days_in_Transit'] - 3))
    noise = np.random.normal(0, 2)
    remaining = row['Initial_Shelf_Life'] * temp_factor * humidity_factor * transit_factor + noise
    return max(0, round(remaining, 1))

df['Remaining_Shelf_Life'] = df.apply(calc_remaining_shelf_life, axis=1)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
output_path = os.path.join('data', 'synthetic_shelf_life_data.csv')
df.to_csv(output_path, index=False)

print(f"Synthetic data saved to {output_path}") 