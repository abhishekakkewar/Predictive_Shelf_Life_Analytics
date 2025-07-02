# EDA Interpretation: Predictive Shelf Life Analytics

## 1. Data Overview
- **Shape:** 1,000 rows × 7 columns.
- **Columns:** Product, Manufacturing_Date, Initial_Shelf_Life, Storage_Temperature, Storage_Humidity, Days_in_Transit, Remaining_Shelf_Life.
- **Missing Values:** None detected, indicating a complete dataset.

## 2. Summary Statistics
- **Initial Shelf Life:** Ranges from 7 to 59 days, with a mean around 28 days.
- **Storage Temperature:** Centered around 8°C (mean), with a standard deviation of ~3°C, simulating typical cold storage.
- **Storage Humidity:** Mean around 60%, with some spread, reflecting variable storage conditions.
- **Days in Transit:** Ranges from 1 to 9 days, median around 5 days.
- **Remaining Shelf Life:** Ranges from about 2 to 81 days, mean around 28 days, showing the impact of environmental factors.

## 3. Distribution Plots
- **Initial Shelf Life:** Fairly uniform or slightly right-skewed, as expected from random generation.
- **Storage Temperature & Humidity:** Both show normal-like distributions, as per the synthetic data generation.
- **Days in Transit:** Discrete, with most products spending 3–7 days in transit.
- **Remaining Shelf Life:** Slightly left-skewed, indicating that some products lose shelf life more rapidly due to adverse conditions.

## 4. Boxplots by Product
- **Initial Shelf Life by Product:** Some products (e.g., Cheese, Bread) may have higher median shelf life than others (e.g., Milk, Yogurt), reflecting product differences.
- **Remaining Shelf Life by Product:** Similar trends, but with more spread, showing that environmental factors and transit can significantly reduce shelf life for some products.

## 5. Correlation Heatmap
- **Initial Shelf Life vs. Remaining Shelf Life:** Strong positive correlation, as expected.
- **Storage Temperature & Humidity vs. Remaining Shelf Life:** Negative correlations, confirming that higher temperature and humidity reduce shelf life.
- **Days in Transit vs. Remaining Shelf Life:** Negative correlation, indicating longer transit reduces shelf life.
- **Inter-feature Correlations:** Storage temperature and humidity are not strongly correlated with each other, which is good for modeling.

## Key Insights
- **Environmental factors (temperature, humidity, transit time) have a measurable negative impact on remaining shelf life.**
- **Product type matters:** Some products are more resilient, while others are more sensitive to environmental conditions.
- **No missing data or major outliers, so the dataset is ready for feature engineering and modeling.** 