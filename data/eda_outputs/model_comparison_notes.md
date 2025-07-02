# Model Comparison & Interpretation: Predictive Shelf Life Analytics

## Models Evaluated
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**

## Evaluation Metrics
- **RMSE (Root Mean Squared Error):** Measures the average magnitude of prediction errors. Lower is better. Sensitive to large errors/outliers.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual values. Lower is better. More robust to outliers.
- **R² (R-squared):** Proportion of variance explained by the model. Closer to 1 is better.

## Results Summary
| Model              |   RMSE |   MAE |   R² |
|--------------------|-------:|------:|-----:|
| Linear Regression  |  3.80  | 3.02  | 0.94 |
| Random Forest      |  3.10  | 2.44  | 0.96 |
| XGBoost            |  3.11  | 2.43  | 0.96 |
| LightGBM           |  2.75  | 2.08  | 0.97 |

## Interpretation
- **LightGBM** achieved the best performance across all metrics, with the lowest RMSE and MAE, and the highest R². This indicates it is the most accurate and reliable model for predicting remaining shelf life in this dataset.
- **Random Forest** and **XGBoost** also performed well, with similar results, and both outperformed Linear Regression.
- **Linear Regression** had the highest error, likely due to its inability to capture complex, non-linear relationships present in the data.

## KPI Importance
- **RMSE** is the most important KPI for this use case because:
  - It penalizes large errors more than MAE, which is crucial in shelf life prediction where large underestimations or overestimations can lead to significant business losses (e.g., spoilage or missed sales).
  - It is widely used in regression problems and is easy to interpret in the same units as the target variable (days).
- **MAE** is also useful for understanding average error, but RMSE is preferred when large errors are especially costly.
- **R²** provides a sense of overall model fit but does not directly reflect error magnitude.

## Best Model Selection
- **LightGBM** is selected as the best model due to its superior performance on RMSE, MAE, and R².
- It is robust, handles feature interactions well, and is efficient for large datasets.

---
This documentation can be used directly in your PPT or project report to explain the model comparison, KPI selection, and best model justification. 