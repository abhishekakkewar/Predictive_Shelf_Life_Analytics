import pandas as pd
import numpy as np
import os

# Load the synthetic data
df = pd.read_csv('data/synthetic_shelf_life_data.csv')

# Convert Manufacturing_Date to datetime
df['Manufacturing_Date'] = pd.to_datetime(df['Manufacturing_Date'])

# --- Advanced Feature Engineering Steps ---

# 1. Time-Based Features
df['Product_Age'] = (pd.Timestamp('2023-07-01') - df['Manufacturing_Date']).dt.days
df['Day_of_Week'] = df['Manufacturing_Date'].dt.dayofweek
df['Month'] = df['Manufacturing_Date'].dt.month
df['Season'] = df['Manufacturing_Date'].dt.month % 12 // 3 + 1
df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

# 2. Environmental Risk Scores
df['Temp_Risk_Score'] = np.where(df['Storage_Temperature'] > 10, 
                                (df['Storage_Temperature'] - 10) * 2, 0)
df['Humidity_Risk_Score'] = np.where(df['Storage_Humidity'] > 70, 
                                   (df['Storage_Humidity'] - 70) * 0.5, 0)
df['Environmental_Risk'] = df['Temp_Risk_Score'] + df['Humidity_Risk_Score']

# 3. Product-Specific Features
df['Is_Dairy'] = df['Product'].isin(['Yogurt', 'Milk', 'Cheese']).astype(int)
df['Shelf_Life_Category'] = pd.cut(df['Initial_Shelf_Life'], 
                                  bins=[0, 14, 30, 60], 
                                  labels=['Short', 'Medium', 'Long'])
df['Age_Ratio'] = df['Product_Age'] / df['Initial_Shelf_Life']

# 4. Interaction Features
df['Temp_Humidity_Interaction'] = df['Storage_Temperature'] * df['Storage_Humidity']
df['Temp_Transit_Interaction'] = df['Storage_Temperature'] * df['Days_in_Transit']
df['Humidity_Transit_Interaction'] = df['Storage_Humidity'] * df['Days_in_Transit']
df['Age_Temp_Interaction'] = df['Product_Age'] * df['Storage_Temperature']

# 5. Statistical Features
df['Temp_Z_Score'] = (df['Storage_Temperature'] - df['Storage_Temperature'].mean()) / df['Storage_Temperature'].std()
df['Humidity_Percentile'] = df['Storage_Humidity'].rank(pct=True)

# 6. Business Logic Features
df['High_Temp_Flag'] = (df['Storage_Temperature'] > 12).astype(int)
df['High_Humidity_Flag'] = (df['Storage_Humidity'] > 75).astype(int)
df['Long_Transit_Flag'] = (df['Days_in_Transit'] > 5).astype(int)

# Risk level classification
conditions = [
    (df['Environmental_Risk'] <= 5),
    (df['Environmental_Risk'] <= 15),
    (df['Environmental_Risk'] > 15)
]
df['Risk_Level'] = np.select(conditions, ['Low', 'Medium', 'High'], default='High')

# 7. One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Product', 'Shelf_Life_Category', 'Risk_Level'], prefix=['Product', 'Shelf_Life', 'Risk'])

# Save processed data
os.makedirs('data', exist_ok=True)
output_path = os.path.join('data', 'processed_shelf_life_data.csv')
df.to_csv(output_path, index=False)

print(f"Enhanced processed data saved to {output_path}")
print(f"Total features after engineering: {len(df.columns)}")
print(f"New features added: Time-based, Environmental Risk, Product-specific, Interactions, Statistical, Business Logic") 