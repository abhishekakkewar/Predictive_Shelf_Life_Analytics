import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load the trained model and data
model = joblib.load('models/LightGBM_shelf_life_model.pkl')
df = pd.read_csv('data/processed_shelf_life_data.csv')

# Prepare features (exclude target and date columns)
feature_columns = [col for col in df.columns if col not in ['Remaining_Shelf_Life', 'Manufacturing_Date']]
X = df[feature_columns]

# Get feature importance from LightGBM model
feature_importance = model.feature_importances_
feature_names = feature_columns

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Create directory for outputs
os.makedirs('data/eda_outputs', exist_ok=True)

# Plot 1: Top 15 features by importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
sns.barplot(data=top_features, x='Importance', y='Feature')
plt.title('Top 15 Most Important Features (LightGBM)')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('data/eda_outputs/top_features_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Feature importance by category
def categorize_feature(feature_name):
    if 'Product_' in feature_name:
        return 'Product Type'
    elif 'Risk_' in feature_name:
        return 'Risk Level'
    elif 'Shelf_Life_' in feature_name:
        return 'Shelf Life Category'
    elif 'Interaction' in feature_name:
        return 'Interaction Features'
    elif 'Risk_Score' in feature_name or 'Environmental_Risk' in feature_name:
        return 'Risk Scores'
    elif 'Z_Score' in feature_name or 'Percentile' in feature_name:
        return 'Statistical Features'
    elif 'Flag' in feature_name:
        return 'Business Logic Flags'
    elif 'Age' in feature_name or 'Day' in feature_name or 'Month' in feature_name or 'Season' in feature_name:
        return 'Time-Based Features'
    elif 'Is_' in feature_name:
        return 'Binary Features'
    else:
        return 'Original Features'

importance_df['Category'] = importance_df['Feature'].apply(categorize_feature)

# Plot by category
plt.figure(figsize=(14, 8))
category_importance = importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
sns.barplot(x=category_importance.values, y=category_importance.index)
plt.title('Feature Importance by Category')
plt.xlabel('Total Importance')
plt.tight_layout()
plt.savefig('data/eda_outputs/feature_importance_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed results
importance_df.to_csv('data/eda_outputs/feature_importance_analysis.csv', index=False)

# Print summary
print("=== FEATURE IMPORTANCE ANALYSIS ===")
print(f"Total features analyzed: {len(feature_names)}")
print("\nTop 10 Most Important Features:")
print(importance_df.head(10)[['Feature', 'Importance']].to_string(index=False))

print("\nFeature Importance by Category:")
print(importance_df.groupby('Category')['Importance'].sum().sort_values(ascending=False).to_string())

print(f"\nAnalysis saved to:")
print("- data/eda_outputs/feature_importance_analysis.csv")
print("- data/eda_outputs/top_features_importance.png")
print("- data/eda_outputs/feature_importance_by_category.png") 