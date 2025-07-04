import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create models directory
os.makedirs('models', exist_ok=True)

# Load the enhanced dataset
print("Loading enhanced dataset...")
df = pd.read_csv('data/processed_shelf_life_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Features: {len(df.columns) - 2}")  # Excluding target and date

# Prepare features and target
feature_columns = [col for col in df.columns if col not in ['Remaining_Shelf_Life', 'Manufacturing_Date']]
X = df[feature_columns]
y = df['Remaining_Shelf_Life']

print(f"Feature columns: {len(feature_columns)}")
print("Feature categories:")
for col in feature_columns:
    if 'Product_' in col:
        print(f"  - Product Type: {col}")
    elif 'Risk_' in col:
        print(f"  - Risk Level: {col}")
    elif 'Shelf_Life_' in col:
        print(f"  - Shelf Life Category: {col}")
    elif 'Interaction' in col:
        print(f"  - Interaction: {col}")
    elif 'Risk_Score' in col or 'Environmental_Risk' in col:
        print(f"  - Risk Score: {col}")
    elif 'Z_Score' in col or 'Percentile' in col:
        print(f"  - Statistical: {col}")
    elif 'Flag' in col:
        print(f"  - Business Logic: {col}")
    elif 'Age' in col or 'Day' in col or 'Month' in col or 'Season' in col:
        print(f"  - Time-Based: {col}")
    elif 'Is_' in col:
        print(f"  - Binary: {col}")
    else:
        print(f"  - Original: {col}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'models/feature_scaler.pkl')

# Model configurations
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
feature_importance_data = {}

print("\n=== MODEL TRAINING WITH ENHANCED FEATURES ===")

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name == 'LightGBM':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Get feature importance
        feature_importance_data[name] = {
            'features': feature_columns,
            'importance': model.feature_importances_
        }
    elif name == 'XGBoost':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Get feature importance
        feature_importance_data[name] = {
            'features': feature_columns,
            'importance': model.feature_importances_
        }
    else:  # RandomForest
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Get feature importance
        feature_importance_data[name] = {
            'features': feature_columns,
            'importance': model.feature_importances_
        }
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[name] = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse
    }
    
    print(f"{name} Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Save model
    joblib.dump(model, f'models/{name}_shelf_life_model.pkl')

# Find best model
best_model = max(results.keys(), key=lambda x: results[x]['R2'])
print(f"\nBest Model: {best_model} (R² = {results[best_model]['R2']:.4f})")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('models/model_evaluation_metrics.csv')
print(f"\nModel evaluation metrics saved to: models/model_evaluation_metrics.csv")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
for model_name, importance_data in feature_importance_data.items():
    importance_df = pd.DataFrame({
        'Feature': importance_data['features'],
        'Importance': importance_data['importance']
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{model_name} - Top 10 Features:")
    print(importance_df.head(10)[['Feature', 'Importance']].to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv(f'models/{model_name}_feature_importance.csv', index=False)

# Cross-validation for best model
print(f"\n=== CROSS-VALIDATION FOR {best_model} ===")
best_model_instance = models[best_model]
if best_model == 'LightGBM':
    cv_scores = cross_val_score(best_model_instance, X, y, cv=5, scoring='r2')
else:
    cv_scores = cross_val_score(best_model_instance, X_train_scaled, y_train, cv=5, scoring='r2')

print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# After best model selection
if best_model == 'LightGBM':
    print("\n=== HYPERPARAMETER TUNING: LightGBM ===")
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, -1]
    }
    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
    grid = GridSearchCV(lgbm, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV R²: {grid.best_score_:.4f}")
    # Save tuned model
    joblib.dump(grid.best_estimator_, 'models/LightGBM_shelf_life_model_tuned.pkl')
    # Save tuned metrics
    tuned_metrics = pd.DataFrame([{
        'Model': 'LightGBM_Tuned',
        'R2_CV': grid.best_score_,
        **grid.best_params_
    }])
    tuned_metrics.to_csv('models/LightGBM_tuned_metrics.csv', index=False)
    print("Tuned LightGBM model and metrics saved.")

print("\n=== TRAINING COMPLETE ===")
print("Models saved:")
for name in models.keys():
    print(f"  - models/{name}_shelf_life_model.pkl")
print("  - models/feature_scaler.pkl")
print("  - models/model_evaluation_metrics.csv") 