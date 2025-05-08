import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import traceback
from openai import OpenAI
from PIL import Image
import json
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Nomlab EcoWise",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* General Styling */
    .stApp {
        background-color: #f9f9f9;
    }
    
    /* Header Styling */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1rem 0.5rem 1rem;
        margin-bottom: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .logo {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .logo-icon {
        font-size: 26px;
        background-color: #4CAF50;
        color: white;
        height: 40px;
        width: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }
    
    .logo-text {
        font-weight: bold;
        font-size: 20px;
        color: #333;
    }
    
    .logo-subtitle {
        font-size: 12px;
        color: #666;
    }
    
    .nav-buttons {
        display: flex;
        gap: 10px;
    }
    
    .nav-button {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        cursor: pointer;
        border: none;
    }
    
    .login-button {
        background-color: #4CAF50;
        color: white;
    }
    
    .demo-button {
        background-color: white;
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    
    /* Hero Section */
    .hero-section {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .hero-icon {
        background-color: #e9f5e9;
        width: 60px;
        height: 60px;
        border-radius: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        color: #4CAF50;
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #666;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #333;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Prompt Items */
    .prompt-item {
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .prompt-item:hover {
        border-color: #4CAF50;
        background-color: #f9fff9;
    }
    
    .prompt-item.selected {
        border-color: #4CAF50;
        background-color: #f0f9f0;
    }
    
    .prompt-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    
    .prompt-icon {
        font-size: 18px;
    }
    
    .prompt-question {
        font-size: 0.9rem;
        color: #555;
        margin: 0;
    }
    
    /* Insights Area */
    .insights-area {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 300px;
        color: #999;
    }
    
    .clock-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .no-insights {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Progress Steps */
    .progress-step {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .progress-icon {
        width: 24px;
        height: 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .progress-icon.active {
        background-color: #4CAF50;
        color: white;
    }
    
    .progress-icon.completed {
        background-color: #4CAF50;
        color: white;
    }
    
    .progress-text {
        font-size: 0.9rem;
        color: #555;
    }
    
    /* Login Page */
    .login-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 2rem auto;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-logo {
        font-size: 32px;
        background-color: #4CAF50;
        color: white;
        height: 60px;
        width: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 30px;
        margin: 0 auto 1rem auto;
    }
    
    .login-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    
    .login-subtitle {
        font-size: 1rem;
        color: #666;
    }
    
    .login-footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #999;
    }
    
    .error-message {
        background-color: #ffe6e6;
        color: #d32f2f;
        padding: 0.75rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Results Card */
    .results-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .results-card.analysis {
        border-left: 4px solid #4CAF50;
    }
    
    .results-card.asker {
        border-left: 4px solid #4e9ed4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "gpt-4o"
# Add login status session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = False
if 'login_attempt' not in st.session_state:
    st.session_state.login_attempt = 0
# Add chatbot mode selection
if 'chatbot_mode' not in st.session_state:
    st.session_state['chatbot_mode'] = "Chatbot Analysis"
if 'results_area' not in st.session_state:
    st.session_state['results_area'] = None
# Store the mode used for the last analysis to display results correctly
if 'last_analysis_mode' not in st.session_state:
    st.session_state['last_analysis_mode'] = None

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
            return obj.isoformat()
        # Handle NumPy types
        if hasattr(obj, 'item'):
            return obj.item()  # This converts NumPy values to Python native types
        # Handle NumPy arrays and other iterables
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)

# --- Helper Functions ---
def generate_data_summary(df):
    """Generate a comprehensive summary of the dataframe for prompt enhancement"""
    buffer = io.StringIO()
    buffer.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    
    # Column names and types
    buffer.write("Column Names and Data Types:\n")
    for col, dtype in zip(df.columns, df.dtypes):
        buffer.write(f"- {col}: {dtype}\n")
    buffer.write("\n")
    
    # Basic statistics
    buffer.write("Basic Statistics:\n")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        buffer.write("Numeric Columns Statistics:\n")
        for col in numeric_cols:
            non_null = df[col].count()
            nulls = df[col].isna().sum()
            if non_null > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                buffer.write(f"- {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, nulls={nulls}\n")
            else:
                buffer.write(f"- {col}: All values are null\n")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(cat_cols) > 0:
        buffer.write("\nCategorical Columns:\n")
        for col in cat_cols:
            unique_vals = df[col].nunique()
            buffer.write(f"- {col}: {unique_vals} unique values\n")
            
            # Show value counts for columns with few unique values
            if unique_vals > 0 and unique_vals <= 10:
                top_vals = df[col].value_counts().head(5).to_dict()
                buffer.write(f"  Top values: {top_vals}\n")
    
    # Date columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    if len(date_cols) > 0:
        buffer.write("\nDate Columns:\n")
        for col in date_cols:
            if df[col].count() > 0:
                min_date = df[col].min()
                max_date = df[col].max()
                buffer.write(f"- {col}: range from {min_date} to {max_date}\n")
    
    # Sample rows
    buffer.write("\nFirst 5 rows:\n")
    buffer.write(df.head(20).to_string())
    
    # Missing values
    missing = df.isna().sum().sum()
    if missing > 0:
        buffer.write(f"\n\nMissing Values: {missing} total missing values in the dataset\n")
        
    return buffer.getvalue()

def create_sample_dataframe():
    """Create a sample retail dataset for demonstration purposes"""
    np.random.seed(42)
    
    # Create date range for the past 30 days
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
    
    # Generate product data
    products = ['Organic Apples', 'Eco-Friendly Detergent', 'Bamboo Toothbrush', 
                'Reusable Water Bottle', 'Solar Charger', 'Recycled Paper Towels',
                'Plant-Based Protein', 'Compostable Bags', 'Sustainable Coffee',
                'LED Light Bulbs']
    
    # Generate random data
    data = []
    for _ in range(300):  # 300 rows of data
        date = np.random.choice(dates)
        product = np.random.choice(products)
        category = {'Organic Apples': 'Food', 
                   'Eco-Friendly Detergent': 'Household', 
                   'Bamboo Toothbrush': 'Personal Care',
                   'Reusable Water Bottle': 'Lifestyle', 
                   'Solar Charger': 'Electronics', 
                   'Recycled Paper Towels': 'Household',
                   'Plant-Based Protein': 'Food', 
                   'Compostable Bags': 'Household', 
                   'Sustainable Coffee': 'Food',
                   'LED Light Bulbs': 'Household'}[product]
        
        price = {'Organic Apples': np.random.uniform(1.5, 3.0), 
                'Eco-Friendly Detergent': np.random.uniform(8.0, 12.0), 
                'Bamboo Toothbrush': np.random.uniform(3.0, 5.0),
                'Reusable Water Bottle': np.random.uniform(15.0, 25.0), 
                'Solar Charger': np.random.uniform(20.0, 40.0), 
                'Recycled Paper Towels': np.random.uniform(4.0, 6.0),
                'Plant-Based Protein': np.random.uniform(18.0, 30.0), 
                'Compostable Bags': np.random.uniform(5.0, 8.0), 
                'Sustainable Coffee': np.random.uniform(10.0, 15.0),
                'LED Light Bulbs': np.random.uniform(6.0, 10.0)}[product]
        
        cost = price * np.random.uniform(0.4, 0.7)  # Cost is 40-70% of price
        quantity = np.random.randint(1, 10)
        revenue = price * quantity
        
        # Add promotion flag randomly
        promotion = np.random.choice([True, False], p=[0.2, 0.8])
        
        # Store region
        region = np.random.choice(['North', 'South', 'East', 'West'])
        
        # Customer segment
        segment = np.random.choice(['New', 'Returning', 'Loyal'], p=[0.3, 0.5, 0.2])
        
        data.append({
            'date': date,
            'product': product,
            'category': category,
            'price': round(price, 2),
            'cost': round(cost, 2),
            'quantity': quantity,
            'revenue': round(revenue, 2),
            'profit': round(revenue - (cost * quantity), 2),
            'promotion': promotion,
            'region': region,
            'customer_segment': segment
        })
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    mask = np.random.random(size=df.shape) < 0.02
    df = df.mask(mask)
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

# --- OpenAI Analysis Function (Code Interpreter) ---
def run_openai_analysis(df, prompt_text):
    """Use OpenAI Assistant with Code Interpreter to analyze the data."""
    if not st.session_state.api_key:
        st.error("Please enter your OpenAI API key in the settings sidebar")
        return False
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=st.session_state.api_key)
        
        # Create a progress tracker that will be cleared when done
        progress_container = st.empty()
        
        with progress_container.container():
            st.markdown('<div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);">', unsafe_allow_html=True)
            
            # Step 1: Enhancing the prompt with data summary
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon active">1</div>'
                '<div class="progress-text">Enhancing prompt with data context...</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Generate data summary
            data_summary = generate_data_summary(df)
            
            # 1. Column-Level Semantic Descriptions
            column_descriptions = """
## Column Definitions and Business Context

### Key Metrics Explained:
- **date**: Transaction date, used for trend analysis and seasonality detection
- **product**: Product name/identifier, the main item being sold to customers
- **category**: Product categorization (e.g., Food, Household), important for segment analysis
- **price**: Retail selling price of the product, directly impacts revenue and customer purchase decisions
- **cost**: Product cost to retailer, critical for determining margins and profitability
- **quantity**: Number of units sold, a direct measure of sales volume
- **revenue**: Total monetary value of sales (price × quantity), the primary top-line metric
- **profit**: Financial gain after costs (revenue - (cost × quantity)), the ultimate measure of business success
- **promotion**: Whether the item was on promotion, a key lever for driving sales and testing price elasticity
- **region**: Geographic sales region, important for localization strategies and identifying regional preferences
- **customer_segment**: Customer grouping (New/Returning/Loyal), critical for customer lifecycle management

### Expected Value Ranges and Interpretation:
- **price**: $1.50-$40 - Higher prices may benefit from bundling strategies
- **cost**: $0.60-$28 - Should typically be 40-70% of price for healthy margins
- **profit margin**: 30-60% is healthy for sustainable retail
- **revenue**: Individual transactions vs. aggregate is important to distinguish
- **quantity**: 1-10 units per transaction is typical, higher suggests bulk buying or promotions
- **customer_segment**: New customers have higher acquisition costs but are needed for growth

### Relationships to Monitor:
- Price elasticity: How sales volume changes with price adjustments
- Promotion effectiveness: Compare revenue/profit during promotion vs. non-promotion periods
- Category performance: Market basket analysis for cross-selling opportunities
- Regional variations: Identify localized preferences and optimization opportunities
"""
            
            # 3. Data Quality Assessment Routine
            data_quality_routine = """
## Data Quality Assessment Routine

To ensure accurate analysis, run these data quality checks at the beginning:

```python
# 1. Check for missing values and handle appropriately
def check_missing_values(df):
    # Count missing values by column
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    # Display missing value stats
    print("\\n--- Missing Values Analysis ---")
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent.round(2)
    })
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Handle missing values appropriately based on column
    if df['revenue'].isnull().any():
        # For revenue, we can recalculate from price and quantity if those are available
        mask = df['revenue'].isnull() & df['price'].notnull() & df['quantity'].notnull()
        df.loc[mask, 'revenue'] = df.loc[mask, 'price'] * df.loc[mask, 'quantity']
        print("Recalculated missing revenue values where possible")
    
    if df['profit'].isnull().any():
        # For profit, we can recalculate from revenue and cost if available
        mask = df['profit'].isnull() & df['revenue'].notnull() & df['cost'].notnull() & df['quantity'].notnull()
        df.loc[mask, 'profit'] = df.loc[mask, 'revenue'] - (df.loc[mask, 'cost'] * df.loc[mask, 'quantity'])
        print("Recalculated missing profit values where possible")
    
    # For remaining missing values in critical columns, consider imputation techniques
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            # Use median for imputation as it's more robust to outliers
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in {col} with median: {median_val:.2f}")
    
    return df

# 2. Detect and handle outliers
def detect_outliers(df):
    print("\\n--- Outlier Detection ---")
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        # Calculate IQR for outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Column {col}: {len(outliers)} outliers detected ({len(outliers)/len(df):.1%} of data)")
            
            # For critical business metrics, flag but don't remove outliers
            if col in ['price', 'cost', 'revenue', 'profit']:
                print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}], Outlier thresholds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"  Note: Outliers in {col} should be examined as they may represent luxury items or special promotions")
    
    return df

# 3. Validate data consistency
def validate_data_consistency(df):
    print("\\n--- Data Consistency Checks ---")
    
    # Business rule validations
    validation_issues = 0
    
    # Check if revenue equals price * quantity
    if 'revenue' in df.columns and 'price' in df.columns and 'quantity' in df.columns:
        calculated_revenue = df['price'] * df['quantity']
        revenue_diff = abs(df['revenue'] - calculated_revenue)
        invalid_revenue = df[revenue_diff > 0.01]  # Allow for small rounding differences
        
        if not invalid_revenue.empty:
            validation_issues += 1
            print(f"Revenue calculation issue in {len(invalid_revenue)} rows")
            print("Example of inconsistent revenue calculation:")
            print(invalid_revenue[['price', 'quantity', 'revenue']].head(3))
    
    # Check if profit equals revenue - (cost * quantity)
    if 'profit' in df.columns and 'revenue' in df.columns and 'cost' in df.columns and 'quantity' in df.columns:
        calculated_profit = df['revenue'] - (df['cost'] * df['quantity'])
        profit_diff = abs(df['profit'] - calculated_profit)
        invalid_profit = df[profit_diff > 0.01]  # Allow for small rounding differences
        
        if not invalid_profit.empty:
            validation_issues += 1
            print(f"Profit calculation issue in {len(invalid_profit)} rows")
            print("Example of inconsistent profit calculation:")
            print(invalid_profit[['revenue', 'cost', 'quantity', 'profit']].head(3))
    
    # Check for negative values in metrics that should be positive
    for col in ['price', 'cost', 'quantity', 'revenue']:
        if col in df.columns:
            negatives = df[df[col] < 0]
            if not negatives.empty:
                validation_issues += 1
                print(f"Found {len(negatives)} rows with negative {col} values")
    
    if validation_issues == 0:
        print("All consistency checks passed!")
    
    return df

# Run all data quality checks
def run_data_quality_assessment(df):
    print("===== STARTING DATA QUALITY ASSESSMENT =====")
    df = check_missing_values(df)
    df = detect_outliers(df)
    df = validate_data_consistency(df)
    print("===== DATA QUALITY ASSESSMENT COMPLETE =====")
    return df
```
"""
            
            # 4. Guided Exploration Flow
            guided_exploration = """
## Guided Data Exploration Flow

Follow this step-by-step exploration path to gain comprehensive understanding of retail performance:

```python
def guided_exploration(df):
    # 1. OVERVIEW ANALYSIS
    print("\\n=== 1. DATASET OVERVIEW ===")
    # Start with overall summary statistics
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total products: {df['product'].nunique()}")
    print(f"Total categories: {df['category'].nunique()}")
    print(f"Total revenue: ${df['revenue'].sum():,.2f}")
    print(f"Total profit: ${df['profit'].sum():,.2f}")
    print(f"Overall profit margin: {(df['profit'].sum() / df['revenue'].sum() * 100):.1f}%")
    
    # 2. PRODUCT PERFORMANCE
    print("\\n=== 2. PRODUCT PERFORMANCE ===")
    # Analyze top and bottom performing products
    product_perf = df.groupby('product').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'quantity': 'sum'
    }).reset_index()
    product_perf['profit_margin'] = (product_perf['profit'] / product_perf['revenue'] * 100).round(1)
    product_perf['avg_price'] = (df.groupby('product')['price'].mean()).round(2)
    
    # Top products by revenue
    print("Top 5 Products by Revenue:")
    top_revenue = product_perf.sort_values('revenue', ascending=False).head(5)
    print(top_revenue)
    
    # Top products by profit
    print("\\nTop 5 Products by Profit:")
    top_profit = product_perf.sort_values('profit', ascending=False).head(5)
    print(top_profit)
    
    # Worst performing products
    print("\\nBottom 5 Products by Profit:")
    bottom_profit = product_perf.sort_values('profit').head(5)
    print(bottom_profit)
    
    # 3. CATEGORY ANALYSIS
    print("\\n=== 3. CATEGORY ANALYSIS ===")
    category_perf = df.groupby('category').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'quantity': 'sum',
        'product': 'nunique'
    }).rename(columns={'product': 'unique_products'}).reset_index()
    category_perf['profit_margin'] = (category_perf['profit'] / category_perf['revenue'] * 100).round(1)
    
    print("Category Performance Overview:")
    print(category_perf.sort_values('revenue', ascending=False))
    
    # 4. TIME TREND ANALYSIS
    print("\\n=== 4. TIME TREND ANALYSIS ===")
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    # Create time-based aggregations
    time_trends = df.groupby(df['date'].dt.date).agg({
        'revenue': 'sum',
        'profit': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # Calculate 7-day moving average for trend spotting
    time_trends['revenue_7day_ma'] = time_trends['revenue'].rolling(7, min_periods=1).mean()
    time_trends['profit_7day_ma'] = time_trends['profit'].rolling(7, min_periods=1).mean()
    
    print("Recent Time Trends (last 10 days):")
    print(time_trends.tail(10))
    
    # 5. PROMOTION EFFECTIVENESS
    print("\\n=== 5. PROMOTION EFFECTIVENESS ===")
    # Compare metrics with and without promotions
    promo_analysis = df.groupby('promotion').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'quantity': 'sum',
        'date': 'nunique'  # Days with/without promotions
    }).rename(columns={'date': 'days'}).reset_index()
    
    # Calculate per-day metrics to make fair comparisons
    promo_analysis['revenue_per_day'] = (promo_analysis['revenue'] / promo_analysis['days']).round(2)
    promo_analysis['profit_per_day'] = (promo_analysis['profit'] / promo_analysis['days']).round(2)
    promo_analysis['quantity_per_day'] = (promo_analysis['quantity'] / promo_analysis['days']).round(2)
    
    print("Promotion vs. No Promotion Performance:")
    print(promo_analysis)
    
    # Calculate promotion lift
    if len(promo_analysis) > 1:
        promo_row = promo_analysis[promo_analysis['promotion'] == True]
        non_promo_row = promo_analysis[promo_analysis['promotion'] == False]
        
        if not promo_row.empty and not non_promo_row.empty:
            rev_lift = (promo_row['revenue_per_day'].values[0] / non_promo_row['revenue_per_day'].values[0] - 1) * 100
            qty_lift = (promo_row['quantity_per_day'].values[0] / non_promo_row['quantity_per_day'].values[0] - 1) * 100
            profit_lift = (promo_row['profit_per_day'].values[0] / non_promo_row['profit_per_day'].values[0] - 1) * 100
            
            print(f"\\nPromotion Lift (% increase during promotions):")
            print(f"Revenue Lift: {rev_lift:.1f}%")
            print(f"Quantity Lift: {qty_lift:.1f}%")
            print(f"Profit Lift: {profit_lift:.1f}%")
    
    # 6. CUSTOMER SEGMENT ANALYSIS
    if 'customer_segment' in df.columns:
        print("\\n=== 6. CUSTOMER SEGMENT ANALYSIS ===")
        segment_analysis = df.groupby('customer_segment').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Calculate derived metrics
        all_quantity = df['quantity'].sum()
        all_count = len(df)
        
        segment_analysis['segment_share'] = (segment_analysis['revenue'] / segment_analysis['revenue'].sum() * 100).round(1)
        
        # Calculate average order value by segment
        aov_by_segment = df.groupby(['customer_segment', 'date']).agg({'revenue': 'sum'}).groupby('customer_segment').mean()
        segment_analysis = segment_analysis.merge(aov_by_segment, on='customer_segment', suffixes=('', '_avg_order'))
        
        print("Customer Segment Performance:")
        print(segment_analysis.sort_values('revenue', ascending=False))
    
    # 7. REGIONAL PERFORMANCE
    if 'region' in df.columns:
        print("\\n=== 7. REGIONAL PERFORMANCE ===")
        region_analysis = df.groupby('region').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'product': 'nunique'
        }).rename(columns={'product': 'unique_products'}).reset_index()
        
        region_analysis['profit_margin'] = (region_analysis['profit'] / region_analysis['revenue'] * 100).round(1)
        region_analysis['revenue_share'] = (region_analysis['revenue'] / region_analysis['revenue'].sum() * 100).round(1)
        
        print("Regional Performance Overview:")
        print(region_analysis.sort_values('revenue', ascending=False))
        
        # Regional product preferences
        print("\\nTop Product by Region:")
        for region in df['region'].unique():
            region_df = df[df['region'] == region]
            top_product = region_df.groupby('product')['revenue'].sum().sort_values(ascending=False).index[0]
            print(f"{region}: {top_product}")
    
    return df
```

After running through this guided exploration, you'll have a comprehensive understanding of the dataset across multiple dimensions and be ready to dive deeper into specific areas of interest.
"""
            
            # Visualization enhancement instructions
            viz_instructions = """
When creating visualizations:

1. Use a consistent, professional color palette (suggestions: 'viridis', 'plasma', or 'Blues' for continuous data; 'Set2' or 'Pastel1' for categorical data)
2. Always include clear titles, axis labels, and legends
3. Add annotations to highlight key insights directly on charts
4. Size charts appropriately (figure size minimum 10x6)
5. Format numbers on axes with appropriate decimal places and thousands separators
6. For time series, use date formatting on x-axis
7. Use appropriate visualization types:
   - Bar charts for categorical comparisons
   - Line charts for time series 
   - Scatter plots for relationships between variables
   - Box plots for distributions and outliers
   - Heatmaps for correlation matrices
8. For multiple related visualizations, use subplots with a shared title
9. Add a brief interpretation caption below each visualization in your code

Example code for properly formatted visualizations:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Create visualization
fig, ax = plt.subplots()
sns.barplot(data=df, x='category', y='value', palette='Set2', ax=ax)

# Add labels and title
ax.set_title('Revenue by Category', fontsize=16)
ax.set_xlabel('Product Category', fontsize=14)
ax.set_ylabel('Revenue ($)', fontsize=14)

# Format y-axis with commas for thousands
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Add annotations
for i, v in enumerate(df['value']):
    ax.text(i, v + 100, f"${v:,.0f}", ha='center')

# Add grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Ensure tight layout
plt.tight_layout()
```
"""
            
            # Structured output instructions
            structure_instructions = """
Your analysis should follow this structured format:

1. **Executive Summary**: 2-3 sentence overview of key findings
2. **Data Overview**: Brief description of the dataset with key statistics
3. **Analysis**: Detailed findings organized in logical sections
4. **Visualizations**: Include 2-4 professional visualizations that best illustrate key insights
5. **Recommendations**: 3-5 specific, actionable recommendations based on the data
6. **Limitations**: Brief mention of any data limitations or areas for further investigation

For each recommendation:
- Make it specific and actionable
- Explain the expected impact
- Tie it directly to findings in the data
"""
            
            # Enhance the prompt with data summary and instructions
            enhanced_prompt = f"""
{prompt_text}

IMPORTANT INSTRUCTIONS:
1. Analyze the retail data based on the question above
2. Provide specific, actionable insights with supporting data
3. Include 2-3 relevant visualizations that directly support your findings
4. Format your response with clear section headers 
5. End with 3-5 concrete, data-driven recommendations

DATA SUMMARY:
- Rows: {df.shape[0]}, Columns: {df.shape[1]}
- Column names: {', '.join(df.columns.tolist())}
{data_summary[:500]}... (summary truncated)

Please make your analysis concise and directly answer the question asked.
"""
            
            # Update progress for step 1
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon completed">✓</div>'
                '<div class="progress-text">Prompt enhanced with data context</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Step 2: Uploading data
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon active">2</div>'
                '<div class="progress-text">Uploading data to OpenAI...</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Convert DataFrame to CSV for upload
            csv_data = df.to_csv(index=False)
            
            # Upload the file
            file_response = client.files.create(
                file=io.BytesIO(csv_data.encode('utf-8')),
                purpose="assistants"
            )
            file_id = file_response.id
            
            # Update progress for step 2
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon completed">✓</div>'
                '<div class="progress-text">Data uploaded successfully</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Step 3: Creating assistant
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon active">3</div>'
                '<div class="progress-text">Initializing AI assistant...</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Create an assistant with the code interpreter tool
            assistant = client.beta.assistants.create(
                name="Retail Category Analyst",
                instructions="""You are a retail category analyst for EcoWise. Create detailed, structured retail analytics reports with visualizations.

REPORT STRUCTURE:
1. 🔍 OBJECTIVES
   - Summarize the key objectives and business questions
   - State what the analysis will address

2. 📊 CATEGORY PERFORMANCE OVERVIEW
   - Present overall performance metrics with visualizations
   - Analyze trends, growth rates, and category health
   - Include a visualization of key metrics

3. 🧱 SEGMENT/PRODUCT SCORECARD
   - Create a detailed breakdown by segment or products
   - Include a table or chart showing comparative metrics
   - Analyze the relationship between segments

4. 🔬 PRODUCT PRODUCTIVITY ANALYSIS
   - Analyze product performance metrics in depth
   - Create visualizations of top/bottom performers
   - Calculate key retail metrics (e.g., rate of sale, GMROI)

5. 📏 GAPS & OPPORTUNITIES
   - Identify underserved segments or product gaps
   - Analyze competitive positioning
   - Suggest potential new products or categories

6. 📎 RECOMMENDATIONS
   - Provide 3-5 data-driven, actionable recommendations
   - Support each with specific data points
   - Quantify potential impact where possible

7. 📈 FUTURE CONSIDERATIONS
   - Suggest forward-looking insights
   - Identify emerging trends
   - Propose "what if" scenarios

Always include high-quality visualizations that directly support your findings. Format all currency with $ and percentages with % symbols. Use emoji icons for section headers. Focus on being both analytical and actionable.""",
                model=st.session_state.model_choice,
                tools=[{"type": "code_interpreter"}]
            )
            
            # Update progress for step 3
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon completed">✓</div>'
                '<div class="progress-text">AI assistant initialized</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Step 4: Creating thread and sending message
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon active">4</div>'
                '<div class="progress-text">Setting up analysis parameters...</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Create thread and send message
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=[{
                    "type": "text",
                    "text": enhanced_prompt
                }],
                attachments=[{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}]
            )
            
            # Update progress for step 4
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon completed">✓</div>'
                '<div class="progress-text">Analysis parameters configured</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Step 5: Running the analysis
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon active">5</div>'
                '<div class="progress-text">Executing in-depth data analysis...</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                temperature=0  # Set temperature to 0 for deterministic responses
            )
            
            # Separate progress tracking for analysis
            analysis_progress = st.progress(0)
            status_message = st.empty()
            
            # Poll for completion
            stages = ["Initializing", "Analyzing data structure", "Identifying patterns", 
                     "Calculating metrics", "Creating visualizations", "Formulating recommendations",
                     "Finalizing insights"]
            current_stage = 0
            
            while run.status in ["queued", "in_progress", "requires_action"]:
                # Update progress animation
                current_stage = (current_stage + 1) % len(stages)
                progress_value = min(0.05 + (current_stage / len(stages)) * 0.7, 0.95)  # Cap at 95%
                analysis_progress.progress(progress_value)
                status_message.markdown(f"<div style='text-align: center; color: #516f90;'>{stages[current_stage]}...</div>", unsafe_allow_html=True)
                
                if run.status == "requires_action":
                    # Handle tool calls if needed
                    try:
                        run = client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=[]
                        )
                    except Exception as tool_err:
                        st.error(f"Error during analysis: {tool_err}")
                        break
                else:
                    # Check status again
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )
                time.sleep(1)
            
            # Complete the progress bar
            analysis_progress.progress(1.0)
            status_message.markdown("<div style='text-align: center; color: #00bf6f;'>Analysis complete!</div>", unsafe_allow_html=True)
            
            # Update final progress step
            st.markdown(
                '<div class="progress-step">'
                '<div class="progress-icon completed">✓</div>'
                '<div class="progress-text">Analysis completed successfully</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear the progress container
        progress_container.empty()
        
        # Retrieve and store results
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            order='asc'
        )
        
        results = []
        
        assistant_responded = False
        for msg in messages.data:
            if msg.role == "assistant":
                assistant_responded = True
                for content_part in msg.content:
                    if content_part.type == "text":
                        # Process markdown text
                        results.append({
                            "type": "text",
                            "content": content_part.text.value
                        })
                    elif content_part.type == "image_file":
                        try:
                            image_file_id = content_part.image_file.file_id
                            image_response = client.files.content(image_file_id)
                            image_bytes = image_response.read()
                            
                            # Store image data for later display
                            results.append({
                                "type": "image",
                                "content": image_bytes
                            })
                        except Exception as img_err:
                            results.append({
                                "type": "error",
                                "content": f"Unable to display a visualization: {img_err}"
                            })
        
        # Update session state for the main UI to display
        st.session_state.analysis_results = results
        st.session_state.analysis_complete = True
        st.session_state['last_analysis_mode'] = "Chatbot Analysis"
        
        # Clean up resources
        try:
            client.beta.assistants.delete(assistant.id)
            client.files.delete(file_id)
        except Exception as cleanup_err:
            pass  # Silently handle cleanup errors
        
        return True
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.code(traceback.format_exc())
        return False

# --- New function for Chatbot Asker mode ---
def run_chatbot_asker(df, prompt):
    """Simplified Chatbot Asker mode - generates a basic summary and asks the LLM."""
    try:
        # 1. Basic validation
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API key in the settings sidebar")
            return False
            
        if df is None or df.empty:
            st.error("No data available for analysis.")
            return False
            
        # 2. Store mode for display purposes
        st.session_state['last_analysis_mode'] = "Chatbot Asker"
        
        # 3. Create a simple progress indicator
        progress_container = st.empty()
        progress_container.info("⏳ Preparing analysis...")
        
        # 4. Get column information but avoid complex operations
        column_names = df.columns.tolist()
        
        # Gather basic stats about the data
        data_stats = {}
        try:
            if 'revenue' in df.columns:
                data_stats["total_revenue"] = float(df['revenue'].sum())
                data_stats["avg_revenue"] = float(df['revenue'].mean())
            if 'profit' in df.columns:
                data_stats["total_profit"] = float(df['profit'].sum())
                data_stats["avg_profit"] = float(df['profit'].mean())
            if 'category' in df.columns:
                data_stats["categories"] = df['category'].unique().tolist()
            if 'product' in df.columns:
                data_stats["product_count"] = df['product'].nunique()
        except:
            pass
        
        # Store data stats for potential display
        st.session_state['data_for_asker'] = {
            "metadata": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": column_names
            },
            "statistics": data_stats
        }
        
        # 5. Update progress
        progress_container.info("⏳ Sending request to OpenAI...")
        
        # 6. Create a structured retail analytics prompt
        system_prompt = """You are an expert retail category analyst for EcoWise. Create professional, structured retail analytics reports that follow a consistent format.

REPORT STRUCTURE:
1. 🔍 OBJECTIVES
   • Summarize the key objectives and business questions
   • State what the analysis will address

2. 📊 CATEGORY PERFORMANCE OVERVIEW
   • Present overall performance metrics with visualizations
   • Analyze trends, growth rates, and category health
   • Include a visualization of key metrics

3. 🧱 SEGMENT/PRODUCT SCORECARD
   • Create a detailed breakdown by segment or products
   • Include a table or chart showing comparative metrics
   • Analyze the relationship between segments

4. 🔬 PRODUCT PRODUCTIVITY ANALYSIS
   • Analyze product performance metrics in depth
   • Create visualizations of top/bottom performers
   • Calculate key retail metrics (e.g., rate of sale, GMROI)

5. 📏 GAPS & OPPORTUNITIES
   • Identify underserved segments or product gaps
   • Analyze competitive positioning
   • Suggest potential new products or categories

6. 📎 RECOMMENDATIONS
   • Provide 3-5 data-driven, actionable recommendations
   • Support each with specific data points
   • Quantify potential impact where possible

7. 📈 FUTURE CONSIDERATIONS
   • Suggest forward-looking insights
   • Identify emerging trends
   • Propose "what if" scenarios

Format all currency with $ and use % symbols for percentages. Use emoji icons for section headers. Focus on being both analytical and actionable.
"""

        # Create a structured prompt that guides the response format
        user_prompt = f"""RETAIL ANALYSIS REQUEST: {prompt}

DATA CONTEXT:
- Dataset contains {len(df)} records with columns: {', '.join(column_names)}
- {json.dumps(data_stats) if data_stats else "Basic statistical information not available"}

Please create a structured retail analytics report following the format in your instructions. Include all relevant sections and use emojis for section headers. Focus your analysis specifically on answering the question above."""
        
        # 7. Make the API call with temperature = 0
        client = OpenAI(api_key=st.session_state.api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # Set to 0 for deterministic responses
            max_tokens=1500
        )
        
        # 8. Extract the response
        if not response or not response.choices or not response.choices[0].message:
            progress_container.error("❌ Failed to get a response from the API")
            return False
            
        answer_text = response.choices[0].message.content
        
        # 9. Store the results
        st.session_state.analysis_results = [{"type": "text", "content": answer_text}]
        st.session_state.analysis_complete = True
        
        # 10. Clear progress indicator
        progress_container.empty()
        
        # Let the main app display the results
        return True
            
    except Exception as e:
        progress_container.empty()
        st.error(f"Error in Chatbot Asker: {str(e)}")
        st.code(traceback.format_exc())
        return False

# --- Authentication function ---
def authenticate(username, password):
    """Simple authentication function"""
    # For demo purposes, using predefined credentials
    # In a real app, you would hash passwords and store them securely
    valid_credentials = {
        "admin": "ecowise123",
        "demo": "demo123",
        "user": "password",
        "llm_analysis": "test123456"  # Added this user from the original app
    }
    
    return username in valid_credentials and password == valid_credentials[username]

# --- Prompt Templates ---
PROMPTS = [
    {
        "id": "underperforming_skus",
        "title": "Increase Sales",
        "icon": "📈",
        "question": "Which products are under-performing this week, and what quick tactics can we apply to lift them?",
        "prompt": """Analyze the dataset to identify under-performing products based on recent sales trends.

For this analysis:
1. Identify products with declining sales or below-average performance 
2. Analyze these products by category, pricing, and profit margin
3. Determine potential reasons for under-performance (price, competition, seasonality)
4. Recommend 3-5 specific, actionable tactics to improve sales for these products
5. Suggest sustainable marketing angles that could help boost these items

Focus on quick-win strategies that can be implemented immediately."""
    },
    {
        "id": "top_growing_skus",
        "title": "Top Growing Products",
        "icon": "📊",
        "question": "List my top 5 fastest-growing products by volume and value, and suggest how to keep the momentum going.",
        "prompt": """Analyze the dataset to identify the top 5 fastest-growing products by both sales volume and revenue.

For this analysis:
1. Calculate growth rates for all products
2. Identify the top 5 products showing the strongest consistent growth
3. Analyze what factors might be driving this growth (pricing, promotions, seasonality)
4. For each product, provide specific recommendations to maintain and accelerate this growth
5. Suggest cross-selling or complementary product opportunities
6. Identify any sustainability aspects of these products that could be emphasized in marketing

Include visualizations showing the growth trends and comparisons."""
    },
    {
        "id": "pricing_optimization",
        "title": "Optimize Pricing",
        "icon": "💹",
        "question": "Which products have high margin but low sales velocity, indicating they could benefit from a price reduction?",
        "prompt": """Analyze the dataset to identify products with high profit margins but low sales velocity that could benefit from strategic price reductions.

For this analysis:
1. Calculate profit margin percentages for all products
2. Analyze the relationship between price, margin, and sales volume/velocity
3. Identify products with significantly higher-than-average margins but lower-than-average sales
4. Estimate price elasticity where possible to predict impact of price changes
5. Recommend specific price reduction targets for identified products
6. Calculate projected impact on revenue and profit from these price adjustments
7. Suggest complementary strategies to highlight the sustainable aspects of these products

Include visualizations comparing price, margin, and sales velocity."""
    },
    {
        "id": "category_performance",
        "title": "Category Analysis",
        "icon": "🗂️",
        "question": "How are my product categories performing compared to each other, and which have the most growth potential?",
        "prompt": """Perform a comprehensive analysis of product category performance and identify growth opportunities.

For this analysis:
1. Compare sales, revenue, profit, and growth metrics across all product categories
2. Analyze trends in category performance over the available time period
3. Identify highest and lowest performing categories based on multiple metrics
4. Analyze customer segments and their category preferences
5. Assess sustainability metrics across categories if available
6. Identify categories with the highest growth potential based on trends and market factors
7. Recommend specific strategies to capitalize on growth opportunities in promising categories

Include visualizations comparing category performance metrics."""
    },
    {
        "id": "customer_segments",
        "title": "Customer Segments",
        "icon": "👥",
        "question": "Which customer segments are most valuable, and how can we better target their needs?",
        "prompt": """Analyze customer segments in the dataset to identify the most valuable segments and opportunities to better serve them.

For this analysis:
1. Define and analyze customer segments based on available data (purchasing behavior, frequency, basket size, etc.)
2. Calculate lifetime value or other value metrics for each segment
3. Identify product preferences and purchasing patterns for high-value segments
4. Analyze which sustainable products resonate most with different segments
5. Recommend targeted strategies to better serve high-value segments
6. Suggest ways to increase value from other segments
7. Identify opportunities for personalized marketing or product recommendations

Include visualizations comparing segment metrics and behaviors."""
    }
]

# --- Login Page Function ---
def show_login_page():
    # Container for centering
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <div class="login-logo">🌿</div>
                <div class="login-title">Nomlab EcoWise</div>
                <div class="login-subtitle">Sustainable Category Manager</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show error message if login failed
        if st.session_state.login_error:
            st.markdown("""
            <div class="error-message">
                Invalid username or password. Please try again.
            </div>
            """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if authenticate(username, password):
                    st.session_state.logged_in = True
                    st.session_state.login_error = False
                    st.rerun()
                else:
                    st.session_state.login_attempt += 1
                    st.session_state.login_error = True
                    st.rerun()
        
        # Demo account info
        st.markdown("""
            <div style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: #666;">
                <p>Demo accounts:</p>
                <p><strong>Username:</strong> demo | <strong>Password:</strong> demo123</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="login-footer">
                © 2023 Nomlab EcoWise. All rights reserved.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Main App Function ---
def show_main_app():
    # --- Sidebar for settings ---
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # API Key input
        st.markdown("#### OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Your API key is required to use OpenAI's services"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Chatbot Mode selection - ADDED
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Chatbot Mode")
        chatbot_mode = st.radio(
            "Select Analysis Mode",
            options=["Chatbot Analysis", "Chatbot Asker"],
            index=0 if st.session_state.chatbot_mode == "Chatbot Analysis" else 1,
            help="'Analysis' uses Code Interpreter for deep analysis & visuals. 'Asker' provides faster answers based on a data summary.",
            label_visibility="collapsed"
        )
        if chatbot_mode != st.session_state.chatbot_mode:
            st.session_state.chatbot_mode = chatbot_mode
            # Clear results when switching modes
            st.session_state.analysis_complete = False 
            st.session_state.analysis_results = []
            if st.session_state.get('results_area'):
                 st.session_state.results_area.empty()
            st.rerun()

        # Model selection
        st.markdown("#### Model Selection")
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
        ]
        model_choice = st.selectbox(
            "Choose Model",
            options=model_options,
            index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0
        )
        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice
        
        # Reset analysis
        if st.button("Reset Analysis", use_container_width=True):
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = []
            st.session_state.selected_prompt = None
            st.rerun()
            
        # Logout button
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # --- Header Section ---
    st.markdown("""
    <div class="header">
        <div class="logo">
            <div class="logo-icon">🌿</div>
            <div>
                <div class="logo-text">Nomlab EcoWise</div>
                <div class="logo-subtitle">Sustainable Category Manager</div>
            </div>
        </div>
        <div class="nav-buttons">
            <button class="nav-button login-button" onclick="javascript:void(0);">Logged In</button>
            <button class="nav-button demo-button">Request Demo</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-section">
        <div class="hero-icon">
            <span style="font-size: 28px;">🌱</span>
        </div>
        <h1 class="hero-title">Optimize and balance retail operations with <span style="color: #4e9ed4;">AI</span></h1>
        <p class="hero-subtitle">Data-driven insights to help independent retailers drive sales and make sustainable decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Main Interface with Three Columns ---
    col1, col2, col3 = st.columns([1, 1, 1])

    # --- Column 1: Prompt Selection ---
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">Select Prompts</h3>', unsafe_allow_html=True)
        
        # Add a tab for predefined vs custom prompts
        predefined_tab, custom_tab = st.tabs(["Predefined Prompts", "Custom Prompt"])
        
        with predefined_tab:
            for prompt in PROMPTS:
                # Check if this prompt is selected
                is_selected = st.session_state.selected_prompt == prompt["id"]
                selected_class = "selected" if is_selected else ""
                
                # Create a clickable prompt item
                prompt_html = f"""
                <div class="prompt-item {selected_class}" id="{prompt["id"]}">
                    <div class="prompt-header">
                        <span class="prompt-icon">{prompt["icon"]}</span>
                        <strong>{prompt["title"]}</strong>
                    </div>
                    <p class="prompt-question">{prompt["question"]}</p>
                </div>
                """
                
                # Use a button that looks like the div but can trigger actions
                if st.markdown(prompt_html, unsafe_allow_html=True):
                    pass  # Markdown doesn't return a clickable element in Streamlit
                
                # Add a hidden button that actually handles the selection - fixed by removing label_visibility
                if st.button(f"Select: {prompt['title']}", key=f"btn_{prompt['id']}", help=prompt["question"]):
                    st.session_state.selected_prompt = prompt["id"]
                    st.session_state.custom_prompt_text = None
                    st.rerun()
        
        with custom_tab:
            st.markdown("""
            <p style="font-size: 0.9rem; color: #516f90; margin-bottom: 0.75rem;">
            Enter your own analysis prompt below. Be specific about what insights you're looking for.
            </p>
            """, unsafe_allow_html=True)
            
            # Initialize custom prompt state if not exists
            if 'custom_prompt_text' not in st.session_state:
                st.session_state.custom_prompt_text = None
            
            # Custom prompt text area
            custom_prompt = st.text_area(
                "Your custom prompt",
                placeholder="Example: Analyze the relationship between price and sales volume. Identify which products would benefit most from price adjustments.",
                height=150,
                key="custom_prompt_input"
            )
            
            # Button to use the custom prompt
            if st.button("Use Custom Prompt", type="primary"):
                if custom_prompt and len(custom_prompt.strip()) > 10:
                    st.session_state.custom_prompt_text = custom_prompt
                    st.session_state.selected_prompt = "custom"
                    st.success("✅ Custom prompt set! Now upload data to run the analysis.")
                    st.rerun()
                else:
                    st.error("Please enter a detailed prompt (at least 10 characters).")
            
            # Help text
            st.markdown("""
            <p style="font-size: 0.85rem; color: #516f90; margin-top: 0.75rem;">
            <strong>Tips for effective prompts:</strong>
            <ul style="margin-top: 0.25rem;">
                <li>Be specific about metrics you're interested in</li>
                <li>Mention timeframes if relevant</li>
                <li>Include business context or goals</li>
                <li>Ask for specific visualization types if needed</li>
            </ul>
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Column 2: Data Upload ---
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">Upload Data</h3>', unsafe_allow_html=True)
        
        # Data upload options
        upload_tab, sample_tab = st.tabs(["Upload Your Data", "Use Sample Data"])
        
        with upload_tab:
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx"],
                help="Upload your retail data file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file based on type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.uploaded_df = df
                    st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                    
                    # Show preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # Show quick stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{df.shape[0]:,}")
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        missing = df.isna().sum().sum()
                        st.metric("Missing Values", missing)
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.session_state.uploaded_df = None
        
        with sample_tab:
            st.markdown("Use our sample retail dataset with 300 records of sustainable product sales.")
            
            if st.button("Load Sample Data", key="load_sample"):
                with st.spinner("Generating sample data..."):
                    sample_df = create_sample_dataframe()
                    st.session_state.uploaded_df = sample_df
                
                st.success("✅ Sample data loaded successfully")
                
                # Show preview
                st.markdown("#### Sample Data Preview")
                st.dataframe(sample_df.head(5), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Analysis Button Logic (MODIFIED) ---
        analysis_button_enabled = (st.session_state.selected_prompt and st.session_state.uploaded_df is not None)
        prompt_title = "Analysis" # Default title
        final_prompt_text = None

        if analysis_button_enabled:
            if st.session_state.selected_prompt == "custom":
                if st.session_state.custom_prompt_text:
                    prompt_title = "Custom Analysis"
                    final_prompt_text = st.session_state.custom_prompt_text
                else: analysis_button_enabled = False # Disable if custom selected but no text
            else:
                selected_prompt_data = next((p for p in PROMPTS if p["id"] == st.session_state.selected_prompt), None)
                if selected_prompt_data:
                    prompt_title = selected_prompt_data['title']
                    final_prompt_text = selected_prompt_data['prompt']
                else: analysis_button_enabled = False # Disable if prompt ID not found

        # Display button if enabled
        if analysis_button_enabled:
            button_label = f"🚀 Run: {prompt_title} ({st.session_state.chatbot_mode})"
            if st.button(button_label, key="analyze_button", use_container_width=True, type="primary"):
                if not st.session_state.api_key:
                    st.sidebar.error("⚠️ OpenAI API key is required.")
                    st.error("Please enter your OpenAI API key in the sidebar.")
                elif final_prompt_text: # Ensure we have a prompt
                    # Clear previous results
                    st.session_state.analysis_results = []
                    st.session_state.analysis_complete = False
                    
                    # Important: Don't try to empty the results area here,
                    # let the analysis functions handle their own displays
                    
                    # Simply call the right function based on mode
                    if st.session_state.chatbot_mode == "Chatbot Analysis":
                        run_openai_analysis(
                            st.session_state.uploaded_df.copy(),
                            final_prompt_text
                        )
                    else: # Chatbot Asker mode
                        run_chatbot_asker(
                            st.session_state.uploaded_df.copy(),
                            final_prompt_text
                        )
                    
                    # No rerun needed - the functions handle their own displays
                else:
                    st.error("Could not determine the prompt to run.")
        else:
            st.info("Select a prompt and upload data to enable analysis.")

    # --- Column 3: Results Display (MODIFIED) ---
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="card-title">View Insights</h3>', unsafe_allow_html=True)
        
        # Define results area (don't try to clear it here)
        results_display_area = st.container()
        
        # Display existing results if analysis was previously completed
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            with results_display_area:
                result_mode = st.session_state.get('last_analysis_mode', 'Chatbot Analysis')
                title_color = "#4CAF50" if result_mode == "Chatbot Analysis" else "#4e9ed4"
                card_class = "analysis" if result_mode == "Chatbot Analysis" else "asker"
                
                st.markdown(f"<div class='results-card {card_class}'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color:{title_color};'>{result_mode} Results</h3>", unsafe_allow_html=True)
                
                for result in st.session_state.analysis_results:
                    if result["type"] == "text":
                        st.markdown(result["content"])
                    elif result["type"] == "image":
                        try:
                            image = Image.open(io.BytesIO(result["content"]))
                            st.image(image, use_column_width=True)
                            st.markdown("<div style='text-align:center; color:#516f90; font-size:0.9rem; margin-bottom:1rem;'>Visualization from Chatbot Analysis</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    elif result["type"] == "error":
                        st.warning(result["content"])
                
                # Add expander for Asker mode summary
                if result_mode == "Chatbot Asker":
                    # Create a simpler data summary directly
                    data_summary = {
                        "rows": len(st.session_state.uploaded_df),
                        "columns": len(st.session_state.uploaded_df.columns),
                        "column_names": st.session_state.uploaded_df.columns.tolist()
                    }
                    
                    # Add very basic statistics if possible
                    if 'revenue' in st.session_state.uploaded_df.columns:
                        data_summary["total_revenue"] = float(st.session_state.uploaded_df['revenue'].sum())
                    if 'profit' in st.session_state.uploaded_df.columns:
                        data_summary["total_profit"] = float(st.session_state.uploaded_df['profit'].sum())
                    
                        
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Show empty state
            with results_display_area:
                st.markdown("""
                <div class="insights-area">
                    <div class="clock-icon">⏱️</div>
                    <div class="no-insights">No insights yet</div>
                    <p>Select a prompt, upload data, and click Run.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Store the results area for other functions to use
        st.session_state['results_area'] = results_display_area
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666; font-size: 0.9rem;">
        © 2023 Nomlab EcoWise. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

# --- Main Application Logic: Show Login or Main App ---
if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page() 
