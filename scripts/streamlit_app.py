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
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Analytics", "🔍 Feature Importance", "💬 AI Assistant"])
    
    with tab1:
        st.header("🎯 Shelf Life Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Information")
            
            # Product details
            product_type = st.selectbox(
                "Product Type",
                ['Yogurt', 'Milk', 'Cheese', 'Butter', 'Cream']
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
            st.subheader("Storage Conditions")
            
            # Storage conditions
            storage_temperature = st.slider(
                "Storage Temperature (°C)",
                min_value=-5.0,
                max_value=25.0,
                value=4.0,
                step=0.5
            )
            
            storage_humidity = st.slider(
                "Storage Humidity (%)",
                min_value=30,
                max_value=95,
                value=65
            )
            
            days_in_transit = st.number_input(
                "Days in Transit",
                min_value=0,
                max_value=30,
                value=2
            )
            
            days_in_storage = st.number_input(
                "Days in Storage",
                min_value=0,
                max_value=365,
                value=5
            )
        
        # Manufacturing date
        st.subheader("Manufacturing Date")
        manufacturing_date = st.date_input(
            "Manufacturing Date",
            value=datetime.now() - timedelta(days=10)
        )
        
        # Prediction button
        if st.button("🚀 Predict Remaining Shelf Life", type="primary"):
            with st.spinner("Making prediction..."):
                input_data = {
                    'product_type': product_type,
                    'initial_shelf_life': initial_shelf_life,
                    'risk_level': risk_level,
                    'shelf_life_category': shelf_life_category,
                    'storage_temperature': storage_temperature,
                    'storage_humidity': storage_humidity,
                    'days_in_transit': days_in_transit,
                    'days_in_storage': days_in_storage,
                    'manufacturing_date': manufacturing_date
                }
                
                prediction = predict_shelf_life(input_data, model, scaler)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                if prediction is not None:
                    # Main prediction display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Remaining Shelf Life (Tuned LightGBM)",
                            f"{prediction:.1f} days",
                            delta=f"{prediction - initial_shelf_life:.1f} days"
                        )
                    
                    with col2:
                        # Calculate percentage remaining
                        percentage_remaining = (prediction / initial_shelf_life) * 100
                        st.metric(
                            "Shelf Life Remaining",
                            f"{percentage_remaining:.1f}%"
                        )
                    
                    # Additional insights
                    st.subheader("📊 Prediction Insights")
                    
                    if prediction < initial_shelf_life * 0.5:
                        st.error("⚠️ Critical: Product has less than 50% shelf life remaining!")
                    elif prediction < initial_shelf_life * 0.7:
                        st.warning("⚠️ Warning: Product has less than 70% shelf life remaining")
                    else:
                        st.success("✅ Good: Product has substantial shelf life remaining")
                    
                    # Risk assessment
                    st.subheader("⚠️ Risk Assessment")
                    risk_factors = []
                    
                    if storage_temperature > 8:
                        risk_factors.append("High storage temperature")
                    if storage_humidity > 80:
                        risk_factors.append("High humidity levels")
                    if days_in_transit > 7:
                        risk_factors.append("Extended transit time")
                    if days_in_storage > 30:
                        risk_factors.append("Extended storage period")
                    
                    if risk_factors:
                        st.warning(f"Risk factors detected: {', '.join(risk_factors)}")
                    else:
                        st.success("No significant risk factors detected")
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
                bulk_data = pd.read_excel(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(bulk_data)} records.")
                
                # Display sample of uploaded data
                st.subheader("📋 Sample of Uploaded Data")
                st.dataframe(bulk_data.head(), use_container_width=True)
                
                # Check required columns
                required_columns = [
                    'Product_Type', 'Initial_Shelf_Life', 'Risk_Level', 'Shelf_Life_Category',
                    'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit', 'Manufacturing_Date'
                ]
                
                missing_columns = [col for col in required_columns if col not in bulk_data.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("""
                    **Required columns:**
                    - Product_Type (Bread, Cheese, Juice, Milk, Yogurt)
                    - Initial_Shelf_Life (numeric)
                    - Risk_Level (Low, Medium, High)
                    - Shelf_Life_Category (Short, Medium, Long)
                    - Storage_Temperature (numeric, °C)
                    - Storage_Humidity (numeric, %)
                    - Days_in_Transit (numeric)
                    - Manufacturing_Date (date format)
                    """)
                else:
                    # Data validation
                    st.subheader("🔍 Data Validation")
                    
                    # Check data types and values
                    validation_issues = []
                    
                    # Product type validation
                    valid_products = ['Bread', 'Cheese', 'Juice', 'Milk', 'Yogurt']
                    invalid_products = bulk_data[~bulk_data['Product_Type'].isin(valid_products)]['Product_Type'].unique()
                    if len(invalid_products) > 0:
                        validation_issues.append(f"Invalid Product_Type values: {', '.join(invalid_products)}")
                    
                    # Risk level validation
                    valid_risks = ['Low', 'Medium', 'High']
                    invalid_risks = bulk_data[~bulk_data['Risk_Level'].isin(valid_risks)]['Risk_Level'].unique()
                    if len(invalid_risks) > 0:
                        validation_issues.append(f"Invalid Risk_Level values: {', '.join(invalid_risks)}")
                    
                    # Shelf life category validation
                    valid_categories = ['Short', 'Medium', 'Long']
                    invalid_categories = bulk_data[~bulk_data['Shelf_Life_Category'].isin(valid_categories)]['Shelf_Life_Category'].unique()
                    if len(invalid_categories) > 0:
                        validation_issues.append(f"Invalid Shelf_Life_Category values: {', '.join(invalid_categories)}")
                    
                    # Numeric validation
                    numeric_columns = ['Initial_Shelf_Life', 'Storage_Temperature', 'Storage_Humidity', 'Days_in_Transit']
                    for col in numeric_columns:
                        if not pd.api.types.is_numeric_dtype(bulk_data[col]):
                            validation_issues.append(f"{col} must be numeric")
                    
                    # Date validation
                    try:
                        pd.to_datetime(bulk_data['Manufacturing_Date'])
                    except:
                        validation_issues.append("Manufacturing_Date must be in valid date format")
                    
                    if validation_issues:
                        st.error("❌ Data validation failed:")
                        for issue in validation_issues:
                            st.write(f"• {issue}")
                    else:
                        st.success("✅ Data validation passed!")
                        
                        # Bulk prediction button
                        if st.button("🚀 Start Bulk Prediction", type="primary"):
                            with st.spinner("Processing bulk predictions..."):
                                predictions_list = []
                                errors_list = []
                                
                                # Process each row
                                for idx, row in bulk_data.iterrows():
                                    try:
                                        # Prepare input data
                                        input_data = {
                                            'product_type': row['Product_Type'],
                                            'initial_shelf_life': float(row['Initial_Shelf_Life']),
                                            'risk_level': row['Risk_Level'],
                                            'shelf_life_category': row['Shelf_Life_Category'],
                                            'storage_temperature': float(row['Storage_Temperature']),
                                            'storage_humidity': float(row['Storage_Humidity']),
                                            'days_in_transit': int(row['Days_in_Transit']),
                                            'days_in_storage': 5,  # Default value
                                            'manufacturing_date': pd.to_datetime(row['Manufacturing_Date']).date()
                                        }
                                        
                                        # Make prediction
                                        prediction = predict_shelf_life(input_data, model, scaler)
                                        
                                        if prediction is not None:
                                            # Calculate additional metrics
                                            percentage_remaining = (prediction / input_data['initial_shelf_life']) * 100
                                            
                                            # Risk assessment
                                            risk_factors = []
                                            if input_data['storage_temperature'] > 8:
                                                risk_factors.append("High temperature")
                                            if input_data['storage_humidity'] > 80:
                                                risk_factors.append("High humidity")
                                            if input_data['days_in_transit'] > 7:
                                                risk_factors.append("Long transit")
                                            
                                            # Status classification
                                            if prediction < input_data['initial_shelf_life'] * 0.5:
                                                status = "Critical"
                                            elif prediction < input_data['initial_shelf_life'] * 0.7:
                                                status = "Warning"
                                            else:
                                                status = "Good"
                                            
                                            predictions_list.append({
                                                'Row_Index': idx + 1,
                                                'Product_Type': input_data['product_type'],
                                                'Initial_Shelf_Life': input_data['initial_shelf_life'],
                                                'Predicted_Remaining_Shelf_Life': round(prediction, 2),
                                                'Shelf_Life_Remaining_Percentage': round(percentage_remaining, 1),
                                                'Status': status,
                                                'Risk_Factors': ', '.join(risk_factors) if risk_factors else 'None',
                                                'Storage_Temperature': input_data['storage_temperature'],
                                                'Storage_Humidity': input_data['storage_humidity'],
                                                'Days_in_Transit': input_data['days_in_transit'],
                                                'Manufacturing_Date': input_data['manufacturing_date']
                                            })
                                        else:
                                            errors_list.append(f"Row {idx + 1}: Prediction failed")
                                            
                                    except Exception as e:
                                        errors_list.append(f"Row {idx + 1}: {str(e)}")
                                
                                # Create results dataframe
                                if predictions_list:
                                    results_df = pd.DataFrame(predictions_list)
                                    
                                    # Display results summary
                                    st.success(f"✅ Bulk prediction completed! Processed {len(predictions_list)} records successfully.")
                                    
                                    if errors_list:
                                        st.warning(f"⚠️ {len(errors_list)} records had errors:")
                                        for error in errors_list[:5]:  # Show first 5 errors
                                            st.write(f"• {error}")
                                        if len(errors_list) > 5:
                                            st.write(f"• ... and {len(errors_list) - 5} more errors")
                                    
                                    # Results summary
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Processed", len(predictions_list))
                                    with col2:
                                        st.metric("Average Remaining Shelf Life", f"{results_df['Predicted_Remaining_Shelf_Life'].mean():.1f} days")
                                    with col3:
                                        critical_count = len(results_df[results_df['Status'] == 'Critical'])
                                        st.metric("Critical Status", critical_count)
                                    with col4:
                                        good_count = len(results_df[results_df['Status'] == 'Good'])
                                        st.metric("Good Status", good_count)
                                    
                                    # Display results table
                                    st.subheader("📊 Prediction Results")
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Download functionality
                                    st.subheader("💾 Download Results")
                                    
                                    # Create Excel file with multiple sheets
                                    from io import BytesIO
                                    import openpyxl
                                    from openpyxl.styles import Font, PatternFill, Alignment
                                    
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                        # Main results sheet
                                        results_df.to_excel(writer, sheet_name='Predictions', index=False)
                                        
                                        # Summary sheet
                                        summary_data = {
                                            'Metric': [
                                                'Total Records Processed',
                                                'Successful Predictions',
                                                'Failed Predictions',
                                                'Average Remaining Shelf Life (days)',
                                                'Critical Status Count',
                                                'Warning Status Count',
                                                'Good Status Count',
                                                'Average Storage Temperature (°C)',
                                                'Average Storage Humidity (%)'
                                            ],
                                            'Value': [
                                                len(bulk_data),
                                                len(predictions_list),
                                                len(errors_list),
                                                round(results_df['Predicted_Remaining_Shelf_Life'].mean(), 2),
                                                len(results_df[results_df['Status'] == 'Critical']),
                                                len(results_df[results_df['Status'] == 'Warning']),
                                                len(results_df[results_df['Status'] == 'Good']),
                                                round(results_df['Storage_Temperature'].mean(), 2),
                                                round(results_df['Storage_Humidity'].mean(), 2)
                                            ]
                                        }
                                        summary_df = pd.DataFrame(summary_data)
                                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                        
                                        # Product type analysis sheet
                                        product_analysis = results_df.groupby('Product_Type').agg({
                                            'Predicted_Remaining_Shelf_Life': ['mean', 'std', 'count'],
                                            'Status': lambda x: (x == 'Critical').sum()
                                        }).round(2)
                                        product_analysis.columns = ['Avg_Remaining_Shelf_Life', 'Std_Remaining_Shelf_Life', 'Count', 'Critical_Count']
                                        product_analysis.to_excel(writer, sheet_name='Product_Analysis')
                                    
                                    # Download button
                                    st.download_button(
                                        label="📥 Download Excel Results",
                                        data=output.getvalue(),
                                        file_name=f"shelf_life_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                    
                                    st.info("📋 The Excel file contains three sheets:")
                                    st.write("• **Predictions**: Detailed results for each product")
                                    st.write("• **Summary**: Overall statistics and metrics")
                                    st.write("• **Product_Analysis**: Breakdown by product type")
                                    
                                else:
                                    st.error("❌ No predictions were successful. Please check your data format.")
                                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure the file is a valid Excel file (.xlsx or .xls) with the correct format.")
    
    with tab2:
        st.header("📈 Data Analytics")
        
        # Create Product_Type column for analysis
        df_analysis = df.copy()
        product_columns = ['Product_Bread', 'Product_Cheese', 'Product_Juice', 'Product_Milk', 'Product_Yogurt']
        df_analysis['Product_Type'] = 'Other'
        for col in product_columns:
            if col in df_analysis.columns:
                df_analysis.loc[df_analysis[col] == True, 'Product_Type'] = col.replace('Product_', '')
        
        # Get unique product types (excluding 'Other')
        unique_products = df_analysis[df_analysis['Product_Type'] != 'Other']['Product_Type'].unique()
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(df))
        with col2:
            st.metric("Unique Product Types", len(unique_products))
        with col3:
            st.metric("Avg Remaining Shelf Life", f"{df['Remaining_Shelf_Life'].mean():.1f} days")
        with col4:
            st.metric("Avg Storage Temperature", f"{df['Storage_Temperature'].mean():.1f}°C")
        
        # Product Type Analysis with detailed metrics
        st.subheader("📊 Product Type Analysis")
        
        # Calculate comprehensive statistics for each product type
        product_stats = df_analysis[df_analysis['Product_Type'] != 'Other'].groupby('Product_Type').agg({
            'Remaining_Shelf_Life': ['mean', 'std', 'count'],
            'Storage_Temperature': ['mean', 'std'],
            'Storage_Humidity': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
        product_stats = product_stats.reset_index()
        
        # Rename columns for better display
        product_stats.columns = [
            'Product Type', 'Avg Remaining Shelf Life (days)', 'Std Remaining Shelf Life', 'Count',
            'Avg Storage Temperature (°C)', 'Std Storage Temperature', 'Avg Storage Humidity (%)', 'Std Storage Humidity'
        ]
        
        # Display the comprehensive product statistics
        st.dataframe(product_stats, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Remaining shelf life distribution by product type
            fig = px.box(
                df_analysis[df_analysis['Product_Type'] != 'Other'],
                x='Product_Type',
                y='Remaining_Shelf_Life',
                title="Remaining Shelf Life by Product Type",
                color='Product_Type'
            )
            fig.update_layout(xaxis_title="Product Type", yaxis_title="Remaining Shelf Life (days)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs Remaining shelf life
            fig = px.scatter(
                df_analysis[df_analysis['Product_Type'] != 'Other'],
                x='Storage_Temperature',
                y='Remaining_Shelf_Life',
                color='Product_Type',
                title="Temperature vs Remaining Shelf Life by Product Type",
                hover_data=['Storage_Humidity']
            )
            fig.update_layout(xaxis_title="Storage Temperature (°C)", yaxis_title="Remaining Shelf Life (days)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Storage temperature distribution by product type
            fig = px.box(
                df_analysis[df_analysis['Product_Type'] != 'Other'],
                x='Product_Type',
                y='Storage_Temperature',
                title="Storage Temperature by Product Type",
                color='Product_Type'
            )
            fig.update_layout(xaxis_title="Product Type", yaxis_title="Storage Temperature (°C)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Storage humidity distribution by product type
            fig = px.box(
                df_analysis[df_analysis['Product_Type'] != 'Other'],
                x='Product_Type',
                y='Storage_Humidity',
                title="Storage Humidity by Product Type",
                color='Product_Type'
            )
            fig.update_layout(xaxis_title="Product Type", yaxis_title="Storage Humidity (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary insights
        st.subheader("🔍 Key Insights")
        
        # Find best and worst performing products
        best_product = product_stats.loc[product_stats['Avg Remaining Shelf Life (days)'].idxmax()]
        worst_product = product_stats.loc[product_stats['Avg Remaining Shelf Life (days)'].idxmin()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Best Performing Product:** {best_product['Product Type']}")
            st.write(f"• Average Remaining Shelf Life: {best_product['Avg Remaining Shelf Life (days)']:.1f} days")
            st.write(f"• Average Storage Temperature: {best_product['Avg Storage Temperature (°C)']:.1f}°C")
            st.write(f"• Average Storage Humidity: {best_product['Avg Storage Humidity (%)']:.1f}%")
        
        with col2:
            st.warning(f"**Product Needing Attention:** {worst_product['Product Type']}")
            st.write(f"• Average Remaining Shelf Life: {worst_product['Avg Remaining Shelf Life (days)']:.1f} days")
            st.write(f"• Average Storage Temperature: {worst_product['Avg Storage Temperature (°C)']:.1f}°C")
            st.write(f"• Average Storage Humidity: {worst_product['Avg Storage Humidity (%)']:.1f}%")
    
    with tab3:
        st.header("🔍 Feature Importance Analysis")
        
        if feature_importance is not None:
            # Sort features by importance in descending order (highest on top)
            feature_importance_sorted = feature_importance.sort_values('Importance', ascending=False)
            
            # Top features (already sorted, highest importance on top)
            top_features = feature_importance_sorted.head(15)
            
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Most Important Features (LightGBM) - Highest to Lowest"
            )
            fig.update_layout(
                xaxis_title="Feature Importance Score",
                yaxis_title="Feature Name",
                yaxis={'categoryorder': 'total ascending'}  # This ensures highest importance appears at the top
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature categories
            st.subheader("Feature Categories")
            
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
            
            feature_importance_sorted['Category'] = feature_importance_sorted['Feature'].apply(categorize_feature)
            category_importance = feature_importance_sorted.groupby('Category')['Importance'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=category_importance.values,
                y=category_importance.index,
                orientation='h',
                title="Feature Importance by Category - Highest to Lowest"
            )
            fig.update_layout(
                xaxis_title="Total Importance Score",
                yaxis_title="Feature Category",
                yaxis={'categoryorder': 'total ascending'}  # This ensures highest importance category appears at the top
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed feature table (sorted by importance)
            st.subheader("Detailed Feature Importance (Sorted by Importance)")
            st.dataframe(feature_importance_sorted, use_container_width=True)
            
            # Top 5 features summary
            st.subheader("🏆 Top 5 Most Important Features")
            top_5_features = feature_importance_sorted.head(5)
            for idx, row in top_5_features.iterrows():
                st.write(f"**{idx+1}.** {row['Feature']} - Importance Score: {row['Importance']:.2f}")
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
            - "What's the optimal storage temperature for yogurt?"
            - "Show me products at risk of spoilage"
            - "Compare shelf life performance between products"
            """)
        
        with col2:
            st.markdown("""
            **📈 Business Intelligence**
            - Provide recommendations for storage optimization
            - Suggest inventory management strategies
            - Analyze cost implications of shelf life losses
            - Generate predictive insights for supply chain planning
            
            **🎯 Model Interpretation**
            - Explain why a product has a specific shelf life prediction
            - Identify key factors affecting shelf life
            - Suggest feature improvements for better predictions
            - Validate model assumptions and results
            """)
        
        # Importance and Benefits
        st.subheader("💡 Why is AI Assistant Important?")
        
        st.markdown("""
        **🎯 Enhanced Decision Making**
        - Transform complex data into actionable insights
        - Enable non-technical users to extract valuable information
        - Provide real-time recommendations for operational decisions
        
        **⏱️ Time and Cost Savings**
        - Automate routine data analysis tasks
        - Reduce manual report generation time
        - Enable faster response to shelf life issues
        
        **🔬 Advanced Analytics**
        - Leverage natural language processing for intuitive data exploration
        - Combine multiple data sources for comprehensive insights
        - Provide context-aware recommendations based on business rules
        """)
        
        # How it helps users
        st.subheader("👥 How Can It Help Different Users?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏭 Operations Managers**
            - Monitor real-time shelf life status
            - Optimize storage conditions
            - Plan inventory rotations
            - Reduce waste and losses
            """)
        
        with col2:
            st.markdown("""
            **📊 Data Analysts**
            - Explore data through natural language
            - Generate automated reports
            - Identify patterns and trends
            - Validate model performance
            """)
        
        with col3:
            st.markdown("""
            **🎯 Business Stakeholders**
            - Get executive summaries
            - Understand business impact
            - Make informed decisions
            - Track KPIs and metrics
            """)
        
        # Technical Implementation
        st.subheader("🔧 Technical Implementation")
        
        st.markdown("""
        **🛠️ Planned Features:**
        - **OpenAI GPT Integration**: Leverage advanced language models for natural language understanding
        - **SQL Query Generation**: Convert natural language to database queries
        - **Real-time Data Access**: Connect to live data sources for current insights
        - **Custom Knowledge Base**: Train on domain-specific shelf life knowledge
        - **Multi-modal Support**: Handle text, charts, and data visualizations
        
        **🔐 Security & Privacy:**
        - Secure API key management
        - Data encryption and privacy protection
        - Role-based access control
        - Audit trails for all interactions
        """)
        
        # Future Roadmap
        st.subheader("🚀 Future Development Roadmap")
        
        st.markdown("""
        **Phase 1: Basic Integration**
        - Natural language query processing
        - Basic data insights and recommendations
        - Simple report generation
        
        **Phase 2: Advanced Analytics**
        - Predictive insights and forecasting
        - Automated anomaly detection
        - Integration with external data sources
        
        **Phase 3: Enterprise Features**
        - Multi-user collaboration
        - Custom knowledge base training
        - Advanced visualization capabilities
        """)
        
        # Call to Action
        st.subheader("📞 Get Involved")
        
        st.markdown("""
        **💬 We'd Love Your Input!**
        - What specific questions would you like to ask about your shelf life data?
        - Which types of insights would be most valuable for your operations?
        - How would you prefer to interact with the AI Assistant?
        
        **📧 Contact Us:**
        Share your feedback and requirements to help shape the development of this feature!
        """)
        
        # Demo placeholder
        st.subheader("🎬 Demo Preview")
        
        with st.expander("See Example AI Assistant Interactions"):
            st.markdown("""
            **Example 1: Data Analysis**
            ```
            User: "Which products are most sensitive to temperature changes?"
            AI: "Based on the data, yogurt and milk show the highest sensitivity 
                 to temperature variations. Products stored above 8°C show a 
                 23% reduction in shelf life compared to optimal conditions."
            ```
            
            **Example 2: Business Intelligence**
            ```
            User: "What's the financial impact of current storage conditions?"
            AI: "Current storage conditions are causing an estimated 15% shelf 
                 life reduction, resulting in approximately $45,000 monthly 
                 losses. Optimizing temperature control could save $12,000/month."
            ```
            
            **Example 3: Operational Recommendations**
            ```
            User: "How can I improve shelf life for dairy products?"
            AI: "For dairy products, I recommend:
                 • Maintain temperature between 2-4°C
                 • Keep humidity below 70%
                 • Reduce transit time to under 3 days
                 • Implement FIFO inventory rotation"
            ```
            """)

if __name__ == "__main__":
    main()

