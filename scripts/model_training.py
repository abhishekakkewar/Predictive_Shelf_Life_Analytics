import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb
import lightgbm as lgb

# Load processed data
df = pd.read_csv('data/processed_shelf_life_data.csv')

# Define features and target
X = df.drop(['Remaining_Shelf_Life', 'Manufacturing_Date'], axis=1)
y = df['Remaining_Shelf_Life']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train
eval_results = {}
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    eval_results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

# Select best model (by RMSE)
best_model_name = min(eval_results, key=lambda k: eval_results[k]['RMSE'])
best_model = models[best_model_name]

# Save best model
os.makedirs('models', exist_ok=True)
model_path = f'models/{best_model_name}_shelf_life_model.pkl'
joblib.dump(best_model, model_path)
print(f"Best model ({best_model_name}) saved to {model_path}")

# Save evaluation metrics
metrics_path = 'models/model_evaluation_metrics.csv'
pd.DataFrame(eval_results).T.to_csv(metrics_path)
print(f"Evaluation metrics saved to {metrics_path}") 