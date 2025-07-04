import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_input_file():
    """Create a sample Excel file for testing bulk prediction"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of sample records
    n_records = 50
    
    # Product types and their characteristics
    product_configs = {
        'Bread': {'initial_shelf_life_range': (7, 14), 'temp_range': (2, 8), 'humidity_range': (50, 70)},
        'Cheese': {'initial_shelf_life_range': (30, 90), 'temp_range': (2, 6), 'humidity_range': (60, 80)},
        'Juice': {'initial_shelf_life_range': (14, 30), 'temp_range': (1, 5), 'humidity_range': (40, 60)},
        'Milk': {'initial_shelf_life_range': (7, 21), 'temp_range': (1, 4), 'humidity_range': (50, 70)},
        'Yogurt': {'initial_shelf_life_range': (14, 28), 'temp_range': (2, 6), 'humidity_range': (60, 80)}
    }
    
    # Generate sample data
    data = []
    
    for i in range(n_records):
        # Randomly select product type
        product_type = np.random.choice(list(product_configs.keys()))
        config = product_configs[product_type]
        
        # Generate realistic values based on product type
        initial_shelf_life = np.random.randint(config['initial_shelf_life_range'][0], 
                                             config['initial_shelf_life_range'][1] + 1)
        
        storage_temperature = np.random.uniform(config['temp_range'][0], config['temp_range'][1])
        
        storage_humidity = np.random.uniform(config['humidity_range'][0], config['humidity_range'][1])
        
        days_in_transit = np.random.randint(1, 8)
        
        # Risk level based on conditions
        if storage_temperature > 6 or storage_humidity > 75 or days_in_transit > 5:
            risk_level = np.random.choice(['Medium', 'High'], p=[0.6, 0.4])
        else:
            risk_level = np.random.choice(['Low', 'Medium'], p=[0.7, 0.3])
        
        # Shelf life category based on initial shelf life
        if initial_shelf_life <= 14:
            shelf_life_category = 'Short'
        elif initial_shelf_life <= 30:
            shelf_life_category = 'Medium'
        else:
            shelf_life_category = 'Long'
        
        # Manufacturing date (within last 30 days)
        days_ago = np.random.randint(1, 31)
        manufacturing_date = datetime.now() - timedelta(days=days_ago)
        
        data.append({
            'Product_Type': product_type,
            'Initial_Shelf_Life': initial_shelf_life,
            'Risk_Level': risk_level,
            'Shelf_Life_Category': shelf_life_category,
            'Storage_Temperature': round(storage_temperature, 1),
            'Storage_Humidity': round(storage_humidity, 1),
            'Days_in_Transit': days_in_transit,
            'Manufacturing_Date': manufacturing_date.strftime('%Y-%m-%d')
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to Excel file
    output_file = 'data/sample_bulk_prediction_input.xlsx'
    df.to_excel(output_file, index=False, sheet_name='Sample_Data')
    
    print(f"✅ Sample input file created: {output_file}")
    print(f"📊 Generated {len(df)} sample records")
    
    # Display sample statistics
    print("\n📋 Sample Data Statistics:")
    print(f"• Product Types: {df['Product_Type'].value_counts().to_dict()}")
    print(f"• Risk Levels: {df['Risk_Level'].value_counts().to_dict()}")
    print(f"• Shelf Life Categories: {df['Shelf_Life_Category'].value_counts().to_dict()}")
    print(f"• Temperature Range: {df['Storage_Temperature'].min():.1f}°C to {df['Storage_Temperature'].max():.1f}°C")
    print(f"• Humidity Range: {df['Storage_Humidity'].min():.1f}% to {df['Storage_Humidity'].max():.1f}%")
    print(f"• Transit Days Range: {df['Days_in_Transit'].min()} to {df['Days_in_Transit'].max()} days")
    
    # Display first few rows
    print("\n📄 First 5 rows of sample data:")
    print(df.head().to_string(index=False))
    
    return output_file

def create_validation_test_file():
    """Create a test file with some invalid data to test validation"""
    
    # Create data with some invalid values
    test_data = [
        # Valid records
        {'Product_Type': 'Milk', 'Initial_Shelf_Life': 14, 'Risk_Level': 'Low', 'Shelf_Life_Category': 'Short', 
         'Storage_Temperature': 3.5, 'Storage_Humidity': 65.0, 'Days_in_Transit': 2, 'Manufacturing_Date': '2024-01-15'},
        
        # Invalid product type
        {'Product_Type': 'InvalidProduct', 'Initial_Shelf_Life': 20, 'Risk_Level': 'Medium', 'Shelf_Life_Category': 'Medium', 
         'Storage_Temperature': 5.0, 'Storage_Humidity': 70.0, 'Days_in_Transit': 3, 'Manufacturing_Date': '2024-01-16'},
        
        # Invalid risk level
        {'Product_Type': 'Cheese', 'Initial_Shelf_Life': 45, 'Risk_Level': 'VeryHigh', 'Shelf_Life_Category': 'Long', 
         'Storage_Temperature': 4.0, 'Storage_Humidity': 75.0, 'Days_in_Transit': 4, 'Manufacturing_Date': '2024-01-17'},
        
        # Invalid shelf life category
        {'Product_Type': 'Bread', 'Initial_Shelf_Life': 10, 'Risk_Level': 'Low', 'Shelf_Life_Category': 'VeryShort', 
         'Storage_Temperature': 6.0, 'Storage_Humidity': 60.0, 'Days_in_Transit': 1, 'Manufacturing_Date': '2024-01-18'},
        
        # Invalid date format
        {'Product_Type': 'Yogurt', 'Initial_Shelf_Life': 21, 'Risk_Level': 'Medium', 'Shelf_Life_Category': 'Medium', 
         'Storage_Temperature': 3.0, 'Storage_Humidity': 65.0, 'Days_in_Transit': 2, 'Manufacturing_Date': 'InvalidDate'},
    ]
    
    df = pd.DataFrame(test_data)
    
    output_file = 'data/validation_test_input.xlsx'
    df.to_excel(output_file, index=False, sheet_name='Test_Data')
    
    print(f"\n✅ Validation test file created: {output_file}")
    print("This file contains both valid and invalid data to test the validation features.")
    
    return output_file

if __name__ == "__main__":
    print("🚀 Creating sample input files for bulk prediction testing...")
    
    # Create main sample file
    sample_file = create_sample_input_file()
    
    # Create validation test file
    validation_file = create_validation_test_file()
    
    print(f"\n📁 Files created in 'data' directory:")
    print(f"• {sample_file} - Main sample file with valid data")
    print(f"• {validation_file} - Test file with validation errors")
    
    print("\n💡 Usage Instructions:")
    print("1. Use 'sample_bulk_prediction_input.xlsx' for normal testing")
    print("2. Use 'validation_test_input.xlsx' to test error handling")
    print("3. Upload these files in the 'Bulk Prediction' section of the Streamlit app")
    print("4. The app will validate the data and show any errors") 