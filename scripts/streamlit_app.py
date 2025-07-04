import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta, date
import warnings
import io
import shap
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Predictive Shelf Life Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Style for better visual alignment */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #222 !important;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #fff !important;
        font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_models():
    """Load data and the best trained model (tuned LightGBM)"""
    try:
        # Load data
        df = pd.read_csv('data/processed_shelf_life_data.csv')
        
        # Load only the best model (tuned LightGBM)
        model = None
        if os.path.exists('models/LightGBM_shelf_life_model_tuned.pkl'):
            model = joblib.load('models/LightGBM_shelf_life_model_tuned.pkl')
        else:
            st.error("Tuned LightGBM model not found. Please run model training first.")
            return None, None, None, None
        
        # Load scaler
        scaler = None
        if os.path.exists('models/feature_scaler.pkl'):
            scaler = joblib.load('models/feature_scaler.pkl')
        
        # Load feature importance
        feature_importance = None
        if os.path.exists('models/LightGBM_feature_importance.csv'):
            feature_importance = pd.read_csv('models/LightGBM_feature_importance.csv')
        
        return df, model, scaler, feature_importance
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def create_features(input_data):
    """Create features for prediction matching the training data exactly (33 features, correct order)"""
    features = {}
    # 1. Original features
    features['Initial_Shelf_Life'] = input_data['initial_shelf_life']
    features['Storage_Temperature'] = input_data['storage_temperature']
    features['Storage_Humidity'] = input_data['storage_humidity']
    features['Days_in_Transit'] = input_data['days_in_transit']
    
    # 2. Product_Age and time-based features
    manufacturing_date = input_data['manufacturing_date']
    if isinstance(manufacturing_date, date):
        manufacturing_date = datetime.combine(manufacturing_date, datetime.min.time())
    current_date = datetime.now()
    product_age = (current_date - manufacturing_date).days
    features['Product_Age'] = product_age
    features['Day_of_Week'] = manufacturing_date.weekday()
    features['Month'] = manufacturing_date.month
    features['Season'] = (manufacturing_date.month % 12) // 3 + 1
    features['Is_Weekend'] = 1 if manufacturing_date.weekday() >= 5 else 0
    
    # 3. Environmental risk features
    temp_risk = (input_data['storage_temperature'] - 10) * 2 if input_data['storage_temperature'] > 10 else 0
    humidity_risk = (input_data['storage_humidity'] - 70) * 0.5 if input_data['storage_humidity'] > 70 else 0
    features['Temp_Risk_Score'] = temp_risk
    features['Humidity_Risk_Score'] = humidity_risk
    features['Environmental_Risk'] = temp_risk + humidity_risk
    
    # 4. Product-specific features
    features['Is_Dairy'] = 1 if input_data['product_type'] in ['Yogurt', 'Milk', 'Cheese'] else 0
    features['Age_Ratio'] = product_age / input_data['initial_shelf_life']
    
    # 5. Interaction features
    features['Temp_Humidity_Interaction'] = input_data['storage_temperature'] * input_data['storage_humidity']
    features['Temp_Transit_Interaction'] = input_data['storage_temperature'] * input_data['days_in_transit']
    features['Humidity_Transit_Interaction'] = input_data['storage_humidity'] * input_data['days_in_transit']
    features['Age_Temp_Interaction'] = product_age * input_data['storage_temperature']
    
    # 6. Statistical features (use training means/stds for z-score, percentiles are approximated)
    # These values should match those used in training
    temp_mean = 7.5  # Example mean from training
    temp_std = 3.0   # Example std from training
    features['Temp_Z_Score'] = (input_data['storage_temperature'] - temp_mean) / temp_std
    features['Humidity_Percentile'] = input_data['storage_humidity'] / 100.0  # Approximate percentile
    
    # 7. Business logic flags
    features['High_Temp_Flag'] = 1 if input_data['storage_temperature'] > 12 else 0
    features['High_Humidity_Flag'] = 1 if input_data['storage_humidity'] > 75 else 0
    features['Long_Transit_Flag'] = 1 if input_data['days_in_transit'] > 5 else 0
    
    # 8. One-hot encoding for product type (Bread, Cheese, Juice, Milk, Yogurt)
    for prod in ['Bread', 'Cheese', 'Juice', 'Milk', 'Yogurt']:
        features[f'Product_{prod}'] = 1 if input_data['product_type'] == prod else 0
    
    # 9. One-hot encoding for shelf life category (Short, Medium, Long)
    for cat in ['Short', 'Medium', 'Long']:
        features[f'Shelf_Life_{cat}'] = 1 if input_data['shelf_life_category'] == cat else 0
    
    # 10. One-hot encoding for risk level (Low, Medium)
    for risk in ['Low', 'Medium']:
        features[f'Risk_{risk}'] = 1 if input_data['risk_level'] == risk else 0
    
    # Ensure correct order and only the 33 features used in training
    feature_order = [
        'Initial_Shelf_Life', 'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit',
        'Product_Age', 'Day_of_Week', 'Month', 'Season', 'Is_Weekend',
        'Temp_Risk_Score', 'Humidity_Risk_Score', 'Environmental_Risk', 'Is_Dairy', 'Age_Ratio',
        'Temp_Humidity_Interaction', 'Temp_Transit_Interaction', 'Humidity_Transit_Interaction', 'Age_Temp_Interaction',
        'Temp_Z_Score', 'Humidity_Percentile', 'High_Temp_Flag', 'High_Humidity_Flag', 'Long_Transit_Flag',
        'Product_Bread', 'Product_Cheese', 'Product_Juice', 'Product_Milk', 'Product_Yogurt',
        'Shelf_Life_Short', 'Shelf_Life_Medium', 'Shelf_Life_Long', 'Risk_Low', 'Risk_Medium'
    ]
    return {k: features[k] for k in feature_order}

def predict_shelf_life(input_data, model, scaler):
    """Make prediction using the tuned LightGBM model"""
    features = create_features(input_data)
    feature_df = pd.DataFrame([features])
    
    try:
        # LightGBM doesn't need scaling
        pred = model.predict(feature_df)[0]
        return max(0, pred)  # Ensure non-negative
    except Exception as e:
        st.error(f"Error with prediction: {str(e)}")
        return None

def validate_bulk_data(df):
    """Validate bulk upload data"""
    errors = []
    
    # Check required columns
    required_columns = [
        'Product_Type', 'Initial_Shelf_Life', 'Risk_Level', 'Shelf_Life_Category',
        'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit', 'Manufacturing_Date'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types
    if 'Initial_Shelf_Life' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Initial_Shelf_Life']):
            errors.append("Initial_Shelf_Life must be numeric")
    
    if 'Storage_Temperature' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Storage_Temperature']):
            errors.append("Storage_Temperature must be numeric")
    
    if 'Storage_Humidity' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Storage_Humidity']):
            errors.append("Storage_Humidity must be numeric")
    
    if 'Days_in_Transit' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Days_in_Transit']):
            errors.append("Days_in_Transit must be numeric")
    
    # Check valid values
    valid_products = ['Bread', 'Cheese', 'Juice', 'Milk', 'Yogurt']
    valid_risk_levels = ['Low', 'Medium', 'High']
    valid_categories = ['Short', 'Medium', 'Long']
    
    if 'Product_Type' in df.columns:
        invalid_products = df[~df['Product_Type'].isin(valid_products)]['Product_Type'].unique()
        if len(invalid_products) > 0:
            errors.append(f"Invalid Product_Type values: {', '.join(invalid_products)}")
    
    if 'Risk_Level' in df.columns:
        invalid_risks = df[~df['Risk_Level'].isin(valid_risk_levels)]['Risk_Level'].unique()
        if len(invalid_risks) > 0:
            errors.append(f"Invalid Risk_Level values: {', '.join(invalid_risks)}")
    
    if 'Shelf_Life_Category' in df.columns:
        invalid_cats = df[~df['Shelf_Life_Category'].isin(valid_categories)]['Shelf_Life_Category'].unique()
        if len(invalid_cats) > 0:
            errors.append(f"Invalid Shelf_Life_Category values: {', '.join(invalid_cats)}")
    
    # Check date format
    if 'Manufacturing_Date' in df.columns:
        try:
            pd.to_datetime(df['Manufacturing_Date'])
        except:
            errors.append("Manufacturing_Date must be in valid date format (YYYY-MM-DD)")
    
    return errors

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Predictive Shelf Life Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Shelf Life Prediction for Perishable Products")
    
    # Problem Statement / Use Case
    st.info("""
    **Problem Statement:**
    Food manufacturers and supply chain managers face significant challenges in predicting the remaining shelf life of perishable products due to varying storage and transit conditions. Inaccurate predictions can lead to increased waste, lost revenue, and food safety risks.
    
    **Use Case:**
    This solution enables users to forecast the remaining shelf life of products like dairy, bread, and juice by leveraging product and environmental data. It helps optimize inventory, reduce waste, and improve decision-making across the supply chain.
    """)
    
    # Load data and model
    df, model, scaler, feature_importance = load_data_and_models()
    
    if df is None or model is None:
        st.error("Failed to load data or model. Please ensure all required files are present.")
        return
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Prediction", "📈 Analytics", "🔍 Feature Importance", "💬 AI Assistant", "🧩 Explainable AI"])
    
    with tab1:
        st.header("🎯 Shelf Life Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Information")
            
            # Product details
            product_type = st.selectbox(
                "Product Type",
                ['Bread', 'Cheese', 'Juice', 'Milk', 'Yogurt']
            )
            
            initial_shelf_life = st.number_input(
                "Initial Shelf Life (days)",
                min_value=1,
                max_value=365,
                value=30
            )
            
            risk_level = st.selectbox(
                "Risk Level",
                ['Low', 'Medium', 'High']
            )
            
            shelf_life_category = st.selectbox(
                "Shelf Life Category",
                ['Short', 'Medium', 'Long']
            )
        
        with col2:
            st.subheader("Environmental Conditions")
            
            storage_temperature = st.slider(
                "Storage Temperature (°C)",
                min_value=-10.0,
                max_value=25.0,
                value=4.0,
                step=0.1
            )
            
            storage_humidity = st.slider(
                "Storage Humidity (%)",
                min_value=20.0,
                max_value=100.0,
                value=60.0,
                step=0.1
            )
            
            days_in_transit = st.number_input(
                "Days in Transit",
                min_value=1,
                max_value=30,
                value=3
            )
            
            manufacturing_date = st.date_input(
                "Manufacturing Date",
                value=date.today() - timedelta(days=7),
                max_value=date.today()
            )
        
        # Prediction button
        if st.button("🚀 Predict Remaining Shelf Life", type="primary"):
            input_data = {
                'product_type': product_type,
                'initial_shelf_life': initial_shelf_life,
                'risk_level': risk_level,
                'shelf_life_category': shelf_life_category,
                'storage_temperature': storage_temperature,
                'storage_humidity': storage_humidity,
                'days_in_transit': days_in_transit,
                'manufacturing_date': manufacturing_date
            }
            
            prediction = predict_shelf_life(input_data, model, scaler)
            
            if prediction is not None:
                st.success(f"✅ **Predicted Remaining Shelf Life: {prediction:.1f} days**")
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Shelf Life", f"{initial_shelf_life} days")
                with col2:
                    st.metric("Product Age", f"{(date.today() - manufacturing_date).days} days")
                with col3:
                    st.metric("Storage Temp", f"{storage_temperature}°C")
                
                # Recommendations
                st.info("💡 **Recommendations:**")
                if prediction < 7:
                    st.warning("⚠️ **Critical:** Product has very limited remaining shelf life. Consider immediate action.")
                elif prediction < 14:
                    st.warning("⚠️ **Warning:** Product has limited remaining shelf life. Monitor closely.")
                else:
                    st.success("✅ **Good:** Product has adequate remaining shelf life.")
                
                # Optimization tips
                st.markdown("**Optimization Tips:**")
                tips = []
                if storage_temperature > 8:
                    tips.append("Lower storage temperature to extend shelf life")
                if storage_humidity > 75:
                    tips.append("Reduce humidity levels to prevent spoilage")
                if days_in_transit > 5:
                    tips.append("Optimize transit time to preserve freshness")
                
                if tips:
                    for tip in tips:
                        st.markdown(f"• {tip}")
                else:
                    st.success("Current conditions are optimal for shelf life preservation!")
            else:
                st.error("Failed to make prediction. Please check your input data.")
        
        # Bulk Prediction Section
        st.markdown("---")
        st.subheader("📊 Bulk Prediction")
        st.info("Upload an Excel file with multiple products for batch prediction. The file should contain the required input features without the target variable.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file (.xlsx or .xls)",
            type=['xlsx', 'xls'],
            help="Upload Excel file with columns: Product_Type, Initial_Shelf_Life, Risk_Level, Shelf_Life_Category, Storage_Temperature, Storage_Humidity, Days_in_Transit, Manufacturing_Date"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                bulk_df = pd.read_excel(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(bulk_df)} records.")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(bulk_df.head(), use_container_width=True)
                
                # Validate data
                st.subheader("🔍 Data Validation")
                validation_errors = validate_bulk_data(bulk_df)
                
                if validation_errors:
                    st.error("❌ Validation errors found:")
                    for error in validation_errors:
                        st.error(f"• {error}")
                else:
                    st.success("✅ All data validation checks passed!")
                    
                    # Process predictions
                    if st.button("🚀 Process Bulk Predictions", type="primary"):
                        with st.spinner("Processing predictions..."):
                            predictions = []
                            
                            for idx, row in bulk_df.iterrows():
                                try:
                                    # Convert date string to date object if needed
                                    mfg_date = row['Manufacturing_Date']
                                    if isinstance(mfg_date, str):
                                        mfg_date = pd.to_datetime(mfg_date).date()
                                    elif hasattr(mfg_date, 'date'):
                                        mfg_date = mfg_date.date()
                                    
                                    input_data = {
                                        'product_type': row['Product_Type'],
                                        'initial_shelf_life': row['Initial_Shelf_Life'],
                                        'risk_level': row['Risk_Level'],
                                        'shelf_life_category': row['Shelf_Life_Category'],
                                        'storage_temperature': row['Storage_Temperature'],
                                        'storage_humidity': row['Storage_Humidity'],
                                        'days_in_transit': row['Days_in_Transit'],
                                        'manufacturing_date': mfg_date
                                    }
                                    
                                    prediction = predict_shelf_life(input_data, model, scaler)
                                    predictions.append(prediction if prediction is not None else 0)
                                    
                                except Exception as e:
                                    st.error(f"Error processing row {idx + 1}: {str(e)}")
                                    predictions.append(0)
                            
                            # Add predictions to dataframe
                            result_df = bulk_df.copy()
                            result_df['Predicted_Remaining_Shelf_Life'] = predictions
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Products", len(result_df))
                            with col2:
                                st.metric("Avg Remaining Shelf Life", f"{result_df['Predicted_Remaining_Shelf_Life'].mean():.1f} days")
                            with col3:
                                st.metric("Min Shelf Life", f"{result_df['Predicted_Remaining_Shelf_Life'].min():.1f} days")
                            with col4:
                                st.metric("Max Shelf Life", f"{result_df['Predicted_Remaining_Shelf_Life'].max():.1f} days")
                            
                            # Download button
                            st.subheader("💾 Download Results")
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name="shelf_life_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Excel download
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                result_df.to_excel(writer, sheet_name='Predictions', index=False)
                            output.seek(0)
                            
                            st.download_button(
                                label="📥 Download as Excel",
                                data=output.getvalue(),
                                file_name="shelf_life_predictions.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.header("📈 Data Analytics")
        
        # Create Product_Type column from one-hot encoded columns
        df['Product_Type'] = 'Unknown'
        for product in ['Bread', 'Cheese', 'Juice', 'Milk', 'Yogurt']:
            mask = df[f'Product_{product}'] == 1
            df.loc[mask, 'Product_Type'] = product
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Unique Product Types", df['Product_Type'].nunique())
        with col3:
            st.metric("Avg Remaining Shelf Life", f"{df['Remaining_Shelf_Life'].mean():.1f} days")
        with col4:
            st.metric("Avg Storage Temperature", f"{df['Storage_Temperature'].mean():.1f}°C")
        
        # Product type analysis
        st.subheader("📊 Product Type Analysis")
        product_stats = df.groupby('Product_Type').agg({
            'Remaining_Shelf_Life': ['mean', 'std', 'count'],
            'Storage_Temperature': 'mean',
            'Storage_Humidity': 'mean'
        }).round(2)
        
        product_stats.columns = ['Avg Remaining Shelf Life (days)', 'Std Remaining Shelf Life', 'Count', 'Avg Temperature (°C)', 'Avg Humidity (%)']
        product_stats = product_stats.reset_index()
        
        st.dataframe(product_stats, use_container_width=True)
        
        # Enhanced Visualizations
        st.subheader("📈 Detailed Analytics")
        
        # Row 1: Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Remaining shelf life distribution
            fig = px.histogram(
                df, 
                x='Remaining_Shelf_Life',
                nbins=30,
                title="Distribution of Remaining Shelf Life",
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Storage temperature distribution
            fig = px.histogram(
                df, 
                x='Storage_Temperature',
                nbins=20,
                title="Distribution of Storage Temperature",
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Product analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Product type comparison
            fig = px.box(
                df,
                x='Product_Type',
                y='Remaining_Shelf_Life',
                title="Remaining Shelf Life by Product Type",
                color='Product_Type'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Humidity distribution
            fig = px.histogram(
                df,
                x='Storage_Humidity',
                nbins=20,
                title="Distribution of Storage Humidity",
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Scatter plots and correlations
        st.subheader("🔗 Relationship Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature vs Shelf Life
            try:
                fig = px.scatter(
                    df,
                    x='Storage_Temperature',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Temperature vs Remaining Shelf Life",
                    trendline="ols"
                )
            except ImportError:
                # Fallback without trendline if statsmodels is not available
                fig = px.scatter(
                    df,
                    x='Storage_Temperature',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Temperature vs Remaining Shelf Life"
                )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Humidity vs Shelf Life
            try:
                fig = px.scatter(
                    df,
                    x='Storage_Humidity',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Humidity vs Remaining Shelf Life",
                    trendline="ols"
                )
            except ImportError:
                # Fallback without trendline if statsmodels is not available
                fig = px.scatter(
                    df,
                    x='Storage_Humidity',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Humidity vs Remaining Shelf Life"
                )
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 4: Advanced analytics
        st.subheader("📊 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Days in transit analysis
            fig = px.box(
                df,
                x='Days_in_Transit',
                y='Remaining_Shelf_Life',
                color='Product_Type',
                title="Impact of Transit Days on Shelf Life"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Product age analysis
            try:
                fig = px.scatter(
                    df,
                    x='Product_Age',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Product Age vs Remaining Shelf Life",
                    trendline="ols"
                )
            except ImportError:
                # Fallback without trendline if statsmodels is not available
                fig = px.scatter(
                    df,
                    x='Product_Age',
                    y='Remaining_Shelf_Life',
                    color='Product_Type',
                    title="Product Age vs Remaining Shelf Life"
                )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary insights
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Temperature Impact:**
            - Lower temperatures generally extend shelf life
            - Optimal range: 2-6°C for most products
            """)
        
        with col2:
            st.info("""
            **Humidity Effect:**
            - High humidity (>75%) reduces shelf life
            - Optimal range: 50-70% for most products
            """)
        
        with col3:
            st.info("""
            **Transit Time:**
            - Longer transit reduces remaining shelf life
            - Critical for perishable products
            """)
    
    with tab3:
        st.header("🔍 Feature Importance Analysis")
        
        if feature_importance is not None:
            # Sort by importance (highest first)
            feature_importance_sorted = feature_importance.sort_values('Importance', ascending=False)
            
            # Top features
            top_features = feature_importance_sorted.head(15)
            
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features (Highest to Lowest)"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature categories
            st.subheader("📊 Feature Categories")
            
            # Categorize features
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
                    return 'Risk Assessment'
                elif 'Temp_' in feature_name or 'Humidity_' in feature_name:
                    return 'Environmental Factors'
                elif 'Age' in feature_name or 'Day' in feature_name or 'Month' in feature_name or 'Season' in feature_name:
                    return 'Time-Based Features'
                elif 'Flag' in feature_name:
                    return 'Business Logic Flags'
                elif 'Z_Score' in feature_name or 'Percentile' in feature_name:
                    return 'Statistical Features'
                else:
                    return 'Other'
            
            feature_importance_sorted['Category'] = feature_importance_sorted['Feature'].apply(categorize_feature)
            
            # Category importance
            category_importance = feature_importance_sorted.groupby('Category')['Importance'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=category_importance.values,
                y=category_importance.index,
                orientation='h',
                title="Feature Importance by Category"
            )
            fig.update_layout(xaxis_title="Total Importance", yaxis_title="Category", yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed feature table
            st.subheader("📋 Detailed Feature Importance")
            st.dataframe(feature_importance_sorted, use_container_width=True)
            
        else:
            st.warning("Feature importance data not available. Please run feature importance analysis first.")
    
    with tab4:
        st.header("💬 AI Assistant")
        
        # Main info box
        st.info("🚀 **Future Scope Feature** - This AI Assistant capability is planned for future development to enhance your shelf life analytics experience.")
        
        # What the AI Assistant can do
        st.subheader("🤖 What Can the AI Assistant Do?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Analysis & Insights**
            - Analyze shelf life patterns across different products
            - Identify trends in storage conditions and their impact
            - Generate automated reports on product performance
            - Detect anomalies in shelf life predictions
            
            **🔍 Natural Language Queries**
            - "Which products have the shortest shelf life?"
            - "How does temperature affect yogurt shelf life?"
            - "What are the optimal storage conditions for cheese?"
            - "Show me products at risk of spoilage"
            """)
        
        with col2:
            st.markdown("""
            **💼 Business Intelligence**
            - Storage optimization recommendations
            - Inventory management insights
            - Cost analysis and waste reduction strategies
            - Supply chain optimization suggestions
            
            **🧠 Model Interpretation**
            - Explain prediction results in plain English
            - Identify key factors affecting shelf life
            - Provide actionable recommendations
            - Answer "what-if" scenarios
            """)
        
        # Importance and Benefits
        st.subheader("🎯 Importance & Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 Enhanced User Experience**
            - Intuitive natural language interface
            - No need to learn complex query languages
            - Instant insights and recommendations
            - Personalized analytics experience
            
            **📈 Improved Decision Making**
            - Data-driven insights for managers
            - Real-time problem identification
            - Proactive risk management
            - Optimized resource allocation
            """)
        
        with col2:
            st.markdown("""
            **💰 Business Value**
            - Reduced food waste and costs
            - Improved customer satisfaction
            - Better inventory management
            - Competitive advantage through AI
            
            **🔬 Advanced Analytics**
            - Pattern recognition beyond traditional analysis
            - Predictive insights and forecasting
            - Automated report generation
            - Continuous learning and improvement
            """)
        
        # Example Questions and Answers
        st.subheader("💬 Example Questions & Answers")
        
        st.markdown("""
        **Here are some example questions you could ask the AI Assistant when it's available:**
        """)
        
        # Example Q&A
        examples = [
            {
                "question": "Which products have the shortest remaining shelf life?",
                "answer": "Based on the data, yogurt products typically have the shortest shelf life (average 14-28 days), followed by milk (7-21 days). Products with high storage temperatures (>8°C) or humidity (>75%) show significantly reduced shelf life."
            },
            {
                "question": "How does temperature affect yogurt shelf life?",
                "answer": "Temperature has a strong negative correlation with yogurt shelf life. For every 1°C increase above 4°C, shelf life decreases by approximately 2-3 days. Optimal storage temperature for yogurt is 2-6°C."
            },
            {
                "question": "What are the optimal storage conditions for cheese?",
                "answer": "Cheese performs best at 2-6°C with 60-80% humidity. Lower temperatures extend shelf life significantly, while high humidity (>80%) can cause spoilage. Transit time should be minimized to under 5 days."
            },
            {
                "question": "Show me products at risk of spoilage",
                "answer": "Products with remaining shelf life <7 days, storage temperature >10°C, or humidity >80% are at high risk. Current analysis shows 15% of products fall into high-risk categories and should be prioritized for immediate action."
            },
            {
                "question": "What factors most impact shelf life prediction accuracy?",
                "answer": "The top factors are: 1) Initial shelf life (39% importance), 2) Temperature-transit interactions (38% importance), 3) Humidity-transit interactions (38% importance). Environmental conditions and product age are also critical."
            }
        ]
        
        for i, example in enumerate(examples, 1):
            with st.expander(f"Example {i}: {example['question']}"):
                st.markdown(f"**Q:** {example['question']}")
                st.markdown(f"**A:** {example['answer']}")
        
        # Technical Requirements
        st.subheader("⚙️ Technical Requirements")
        
        st.markdown("""
        **🔧 Infrastructure Needs:**
        - Large Language Model (LLM) API integration
        - Natural Language Processing (NLP) capabilities
        - Real-time data processing
        - Secure API key management
        
        **📊 Data Integration:**
        - SQL query generation from natural language
        - Real-time data access and processing
        - Context-aware responses
        - Multi-modal data handling
        """)
        
        # Call to Action
        st.subheader("🎯 Get Started Today")
        
        st.markdown("""
        While the AI Assistant is in development, you can:
        
        ✅ **Use the current prediction and analytics features**
        ✅ **Upload bulk data for batch processing**
        ✅ **Explore feature importance analysis**
        ✅ **Download detailed reports and insights**
        
        **Stay tuned for AI Assistant updates!** 🤖✨
        """)

    with tab5:
        st.header("🧩 Explainable AI: Model Interpretability")
        st.markdown("""
        This section provides interpretability for the tuned LightGBM model using SHAP (SHapley Additive exPlanations). SHAP helps you understand how each feature impacts the model's predictions, both globally and for individual samples.
        """)

        if model is not None and df is not None:
            # Prepare feature data (exclude target and non-feature columns)
            feature_cols = [col for col in df.columns if col not in ["Remaining_Shelf_Life", "Manufacturing_Date"]]
            X = df[feature_cols]
            # Ensure all columns are numeric or boolean for SHAP/LightGBM
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='raise')
                    except Exception:
                        if set(X[col].dropna().unique()) <= {0, 1, True, False}:
                            X[col] = X[col].astype(bool)
            X = X.select_dtypes(include=[np.number, bool])

            st.subheader("🔎 Model Overview")
            st.markdown("""
            - **Model:** Tuned LightGBM Regressor
            - **Best for:** Predicting remaining shelf life of perishable products
            - **Why LightGBM?**: Outperformed other models (RandomForest, XGBoost) in accuracy and speed after hyperparameter tuning.
            """)

            # SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(X)

            st.subheader("🌍 Global Feature Importance (SHAP)")
            st.markdown("The plot below shows which features have the biggest impact on model predictions across the whole dataset.")
            # SHAP summary plot (bar)
            import matplotlib.pyplot as plt
            fig_summary, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig_summary)

            st.subheader("🔬 Local Explanation (Single Prediction)")
            st.markdown("Select a sample to see how each feature contributed to its prediction.")
            sample_idx = st.number_input("Select sample index", min_value=0, max_value=len(X)-1, value=0)
            sample = X.iloc[[sample_idx]]
            sample_shap = shap_values[sample_idx]
            st.write("**Feature values for this sample:**")
            st.dataframe(sample.T, use_container_width=True)
            # SHAP force plot (as image)
            force_fig = shap.plots.force(explainer.expected_value, sample_shap.values, sample, matplotlib=True, show=False)
            st.pyplot(force_fig, clear_figure=True)

            st.subheader("📝 Plain English Explanation")
            st.markdown("""
            - **Positive SHAP values** (red) push the prediction higher (longer shelf life).
            - **Negative SHAP values** (blue) push the prediction lower (shorter shelf life).
            - The most important features are shown in the global plot above.
            - For the selected sample, the force plot shows which features most influenced the prediction.
            """)
        else:
            st.warning("Model or data not loaded. Please ensure the model and data are available.")

if __name__ == "__main__":
    main()