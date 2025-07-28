import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error
import pandas_datareader as pdr
import io

# --- Configuration ---
current_date = datetime(2025, 7, 28)

# --- Data Generation Functions ---

@st.cache_data(show_spinner="Generating historical loan data (Bank-like portfolio)...")
def generate_historical_loan_data(num_loans=10000): # Increased default historical loans
    """Simulates historical loan data for model training, with a bank-like FICO distribution."""
    np.random.seed(random.randint(0, 1000000))
    random.seed(random.randint(0, 1000000))

    fico_bins = [550, 600, 650, 700, 750, 820]
    # Bank-like FICO distribution for historical data: more prime, but subprime still present
    # Subprime-Low (550-599), Subprime-Mid (600-649), Fair (650-699), Good (700-749), Excellent (750-819)
    fico_probs = [0.05, 0.10, 0.20, 0.35, 0.30] # Sums to 1.0

    historical_data_list = []
    for _ in range(num_loans):
        fico_segment_idx = np.random.choice(len(fico_probs), p=fico_probs)
        fico_min = fico_bins[fico_segment_idx]
        fico_max = fico_bins[fico_segment_idx + 1]
        customer_fico = random.randint(fico_min, fico_max - 1)

        original_balance = random.uniform(10000, 50000) # General range
        loan_age = random.randint(1, 72)
        ltv_at_origination = random.uniform(0.7, 1.1)
        dti_ratio = random.uniform(0.2, 0.6)
        vehicle_type = random.choice(['Sedan', 'SUV', 'Truck', 'Coupe', 'Minivan'])
        origination_date = datetime(2018, 1, 1) + timedelta(days=random.randint(0, 365*5))
        maturity_date = datetime(2018, 1, 1) + timedelta(days=random.randint(365*3, 365*6))

        historical_data_list.append({
            'Customer FICO': customer_fico,
            'Original Balance ($)': original_balance,
            'Loan Age (months)': loan_age,
            'LTV at Origination': ltv_at_origination,
            'DTI Ratio': dti_ratio,
            'Vehicle Type': vehicle_type,
            'Origination Date': origination_date,
            'Maturity Date': maturity_date
        })

    historical_data = pd.DataFrame(historical_data_list)
    historical_data['Loan ID'] = range(1, num_loans + 1)

    # Ensure a strong FICO-PD relationship in historical data for training
    historical_data['Default Probability Base'] = (
        0.5 - (historical_data['Customer FICO'] / 1000) * 0.4 + # Stronger inverse FICO relationship
        (historical_data['Loan Age (months)'] / 72) * 0.15 +
        (historical_data['DTI Ratio'] * 0.2)
    )
    historical_data['Default Probability Base'] = np.clip(historical_data['Default Probability Base'], 0.01, 0.7)

    historical_data['Recovery Rate Base'] = (
        0.6 - (historical_data['LTV at Origination'] * 0.2) +
        (historical_data['Vehicle Type'].map({'Sedan':0.05, 'SUV':0.08, 'Truck':0.1, 'Coupe':0.02, 'Minivan':0.03}))
    )
    historical_data['Recovery Rate Base'] = np.clip(historical_data['Recovery Rate Base'], 0.1, 0.9)

    historical_data['Historical Unemployment Rate'] = np.random.uniform(0.03, 0.10, num_loans)
    historical_data['Historical Used Car Price Change'] = np.random.uniform(-0.15, 0.15, num_loans)

    historical_data['Defaulted'] = (np.random.rand(num_loans) < historical_data['Default Probability Base'] + historical_data['Historical Unemployment Rate'] * 0.5).astype(int)
    historical_data['Defaulted'] = np.clip(historical_data['Defaulted'], 0, 1)

    historical_data['LGD'] = 1 - (historical_data['Recovery Rate Base'] + historical_data['Historical Used Car Price Change'] * 0.5)
    historical_data['LGD'] = np.clip(historical_data['LGD'], 0.01, 0.99)

    return historical_data

@st.cache_data(show_spinner="Generating current loan portfolio (Bank-like portfolio)...")
def generate_current_loan_portfolio(num_loans=50000): # Increased default current loans significantly
    """
    Generates a synthetic current loan portfolio with a FICO distribution
    and loan characteristics resembling a large bank's auto loan book.
    """
    np.random.seed(random.randint(0, 1000000))
    random.seed(random.randint(0, 1000000))

    fico_bins = [550, 600, 650, 700, 750, 820]
    # Bank-like FICO distribution for current portfolio: majority prime, but significant subprime tail
    # Subprime-Low (550-599), Subprime-Mid (600-649), Fair (650-699), Good (700-749), Excellent (750-819)
    # Adjusted to ensure lower FICO segments contribute more to aggregate EL.
    fico_probs = [0.25, 0.30, 0.20, 0.15, 0.10] # Sums to 1.0 (25+30+20+15+10 = 100%)

    # Removed states as per user request
    # states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'AZ', 'VA', 'WA', 'MA', 'CO'] # Top states

    current_loan_data_list = []
    for _ in range(num_loans):
        fico_segment_idx = np.random.choice(len(fico_probs), p=fico_probs)
        fico_min = fico_bins[fico_segment_idx]
        fico_max = fico_bins[fico_segment_idx + 1]
        customer_fico = random.randint(fico_min, fico_max - 1)

        # Assign balance based on FICO segment: wider range, higher averages for higher FICO
        if customer_fico < 600: # Subprime-Low
            original_balance = random.uniform(8000, 25000)
            remaining_balance = random.uniform(4000, original_balance * 0.8)
        elif customer_fico < 650: # Subprime-Mid
            original_balance = random.uniform(12000, 35000)
            remaining_balance = random.uniform(6000, original_balance * 0.85)
        elif customer_fico < 700: # Fair
            original_balance = random.uniform(15000, 45000)
            remaining_balance = random.uniform(8000, original_balance * 0.9)
        elif customer_fico < 750: # Good
            original_balance = random.uniform(20000, 55000)
            remaining_balance = random.uniform(10000, original_balance * 0.92)
        else: # Excellent
            original_balance = random.uniform(25000, 70000) # Can have higher balances
            remaining_balance = random.uniform(12000, original_balance * 0.95)

        term = random.choice([48, 60, 72, 84])
        months_remaining = random.randint(1, term - 1 if term > 1 else 1)
        loan_age = term - months_remaining

        current_loan_data_list.append({
            'Customer FICO': customer_fico,
            'Original Balance ($)': original_balance,
            'Remaining Balance ($)': remaining_balance,
            'Term (months)': term,
            'Months Remaining': months_remaining,
            'Loan Age (months)': loan_age,
            'LTV at Origination': random.uniform(0.7, 1.05),
            'DTI Ratio': random.uniform(0.25, 0.55),
            'Vehicle Type': random.choice(['Sedan', 'SUV', 'Truck', 'Coupe', 'Minivan']),
            # 'State': random.choice(states) # Removed state as per user request
        })

    current_loan_data = pd.DataFrame(current_loan_data_list)
    current_loan_data['Loan ID'] = range(10001, 10001 + num_loans)

    current_loan_data['Origination Date'] = current_loan_data['Loan Age (months)'].apply(
        lambda age: current_date - timedelta(days=age * 30.4375)
    )
    current_loan_data['Origination Year'] = current_loan_data['Origination Date'].dt.year
    current_loan_data['Origination Quarter'] = current_loan_data['Origination Date'].dt.quarter
    return current_loan_data

@st.cache_data(show_spinner="Fetching real macroeconomic data...")
def fetch_macro_data(start_date=datetime(2000, 1, 1), end_date=datetime.now()):
    """Fetches real macroeconomic data from FRED."""
    try:
        macro_data = pdr.DataReader(['UNRATE', 'GDP'], 'fred', start_date, end_date)
        # Forward-fill GDP to align with monthly UNRATE for display purposes
        macro_data['GDP'] = macro_data['GDP'].ffill()
        return macro_data
    except Exception as e:
        st.error(f"Error fetching macro data from FRED: {e}. Using simulated values.")
        return None

# --- Model Training Functions ---

@st.cache_resource(show_spinner="Training PD and LGD models...")
def train_models(historical_data):
    """Trains Logistic Regression for PD and Linear Regression for LGD."""
    # PD Model Training
    pd_features = ['Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'Historical Unemployment Rate', 'LTV at Origination']
    X_pd_train = historical_data[pd_features]
    y_pd_train = historical_data['Defaulted']

    preprocessor_pd = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'Historical Unemployment Rate', 'LTV at Origination'])
        ])
    pd_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor_pd),
                                        ('classifier', LogisticRegression(solver='liblinear', random_state=42))])
    pd_model_pipeline.fit(X_pd_train, y_pd_train)
    pd_auc = roc_auc_score(y_pd_train, pd_model_pipeline.predict_proba(X_pd_train)[:, 1])

    # LGD Model Training
    lgd_features = ['LTV at Origination', 'Vehicle Type', 'Historical Used Car Price Change']
    X_lgd_train = historical_data[lgd_features]
    y_lgd_train = historical_data['LGD']

    preprocessor_lgd = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['LTV at Origination', 'Historical Used Car Price Change']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Vehicle Type'])
        ])
    lgd_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor_lgd),
                                         ('regressor', LinearRegression())])
    lgd_model_pipeline.fit(X_lgd_train, y_lgd_train)
    lgd_rmse = np.sqrt(mean_squared_error(y_lgd_train, lgd_model_pipeline.predict(X_lgd_train)))

    return pd_model_pipeline, lgd_model_pipeline, pd_features, lgd_features, pd_auc, lgd_rmse

# --- Excel Export Function ---
def to_excel_buffer(dataframes_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    processed_data = output.getvalue()
    return processed_data

# --- Main Streamlit App ---

st.set_page_config(layout="wide", page_title="Advanced Auto Loan Loss Forecaster")

st.title("🚗 Advanced Auto Loan Loss Forecaster (Bank-like Portfolio Simulation)")
st.write("This application simulates a large bank's auto loan portfolio and forecasts expected losses under various macroeconomic scenarios using predictive machine learning models.")

# --- About This App / Key Concepts ---
with st.expander("About This App & Key Concepts"):
    st.markdown("""
    This interactive tool demonstrates a comprehensive auto loan loss forecasting framework. It simulates a portfolio of auto loans, integrates real economic data, and uses machine learning models to predict potential credit losses under different economic conditions.

    **What is Loss Forecasting?**
    Loss forecasting is a critical process for banks and lenders to estimate potential future credit losses on their loan portfolios. This helps them set aside adequate reserves, manage risk, and comply with regulations. The core formula for Expected Loss (EL) is:
    $EL = PD \\times LGD \\times EAD$

    **Key Terms:**
    * **PD (Probability of Default):** The likelihood that a borrower will fail to make loan payments. Our model uses a Logistic Regression model to predict this.
    * **LGD (Loss Given Default):** The percentage of the loan's outstanding balance that is lost if a default occurs (i.e., after any collateral recovery). Our model uses a Linear Regression model to predict this.
    * **EAD (Exposure at Default):** The outstanding principal balance of the loan when a default occurs.
    * **FICO Segment:** A common credit scoring system indicating a borrower's creditworthiness. Lower FICO scores generally imply higher risk.
    * **Vintage Analysis:** Grouping loans by their origination period (e.g., year) to observe how different cohorts perform over time.
    * **Macroeconomic Scenarios:** Different economic conditions (e.g., 'Base Case', 'Mild Recession') used to stress-test the portfolio's performance.

    **How to Use:**
    1.  Adjust the number of historical and current loans in the sidebar.
    2.  Explore the "Data Used in the Model" section to see the generated data and loan-level predictions.
    3.  Review the "Portfolio Summary & Scenario Analysis" to understand aggregate expected losses under different economic conditions, including a 'Custom Scenario' you can define.
    4.  Analyze the "Advanced Portfolio Analysis" charts to understand risk segmentation by FICO, vintage, and vehicle type.
    5.  Use the "Download All Model Data to Excel" button to get the raw and processed data for further analysis.
    """)

# --- User Inputs ---
st.sidebar.header("Configuration")
num_historical_loans_input = st.sidebar.number_input("Number of Historical Loans (for training)", min_value=5000, max_value=20000, value=10000, step=1000)
num_current_loans_input = st.sidebar.number_input("Number of Current Loans (for forecasting)", min_value=10000, max_value=100000, value=50000, step=10000)

# --- Data Generation and Model Training ---
historical_data = generate_historical_loan_data(num_historical_loans_input)
current_loan_data = generate_current_loan_portfolio(num_current_loans_input)
macro_data = fetch_macro_data()

pd_model_pipeline, lgd_model_pipeline, pd_features, lgd_features, pd_auc, lgd_rmse = train_models(historical_data)

st.sidebar.subheader("Model Training Performance (on Simulated Data)")
st.sidebar.write(f"**PD Model (Logistic Regression) AUC:** {pd_auc:.4f}")
st.sidebar.write(f"**LGD Model (Linear Regression) RMSE:** {lgd_rmse:.4f}")
st.sidebar.info("Note: Performance metrics are based on simulated data. Real-world models require extensive, curated historical data for high accuracy.")

# --- Macroeconomic Scenarios Definition ---
# Initialize all scenario variables with default/simulated values first
unemp_rate_base = 0.04
unemp_rate_mild = 0.08
unemp_rate_severe = 0.12

gdp_growth_base = 0.005 # 0.5% growth (quarterly, annualized for FRED GDP usually)
gdp_growth_mild = -0.01 # -1% contraction
gdp_growth_severe = -0.04 # -4% contraction

used_car_price_base = 0.00
used_car_price_mild = -0.20
used_car_price_severe = -0.40

if macro_data is not None:
    try:
        unemp_rate_base = macro_data['UNRATE'].iloc[-1] / 100
    except IndexError: pass

    try:
        unemp_rate_severe = macro_data['UNRATE'].loc['2009-10-01'] / 100
    except KeyError:
        unemp_rate_severe = macro_data['UNRATE'].max() / 100 if not macro_data['UNRATE'].empty else unemp_rate_severe
    unemp_rate_severe = max(unemp_rate_severe, unemp_rate_base * 2.5)

    gdp_series = macro_data['GDP'].dropna()
    if len(gdp_series) >= 2:
        gdp_growth_base = (gdp_series.iloc[-1] / gdp_series.iloc[-2]) - 1
    else: pass

    try:
        gdp_trough_q4_2008 = gdp_series.loc['2008-10-01']
        gdp_prev_q3_2008 = gdp_series.loc['2008-07-01']
        gdp_growth_severe = (gdp_trough_q4_2008 / gdp_prev_q3_2008) - 1
    except KeyError:
        gdp_growth_severe = gdp_series.pct_change().min() if not gdp_series.empty else gdp_growth_severe
    gdp_growth_severe = min(gdp_growth_severe, gdp_growth_base * 2)

    unemp_rate_mild = unemp_rate_base + (unemp_rate_severe - unemp_rate_base) / 2
    gdp_growth_mild = gdp_growth_base + (gdp_growth_severe - gdp_growth_base) / 2

forecast_scenarios = {
    'Base Case': {
        'Unemployment Rate': unemp_rate_base,
        'GDP Growth': gdp_growth_base,
        'Used Car Price Change': used_car_price_base
    },
    'Mild Recession': {
        'Unemployment Rate': unemp_rate_mild,
        'GDP Growth': gdp_growth_mild,
        'Used Car Price Change': used_car_price_mild
    },
    'Severe Recession': { # Label simplified as per user request
        'Unemployment Rate': unemp_rate_severe,
        'GDP Growth': gdp_growth_severe,
        'Used Car Price Change': used_car_price_severe
    }
}

# --- Custom Scenario Definition ---
st.sidebar.subheader("Define Custom Scenario")
st.sidebar.markdown("Override predefined scenario values for a custom analysis.")

custom_unemp = st.sidebar.slider("Unemployment Rate (%)", min_value=1.0, max_value=20.0, value=unemp_rate_base * 100, step=0.1, format="%.1f%%", key="custom_unemp_input") / 100
custom_gdp = st.sidebar.slider("GDP Growth (Quarterly %)", min_value=-10.0, max_value=10.0, value=gdp_growth_base * 100, step=0.1, format="%.1f%%", key="custom_gdp_input") / 100
custom_used_car_price = st.sidebar.slider("Used Car Price Change (%)", min_value=-50.0, max_value=50.0, value=used_car_price_base * 100, step=1.0, format="%.0f%%", key="custom_ucpc_input") / 100

# Dynamically add custom scenario to the forecast_scenarios dictionary
forecast_scenarios['Custom Scenario'] = {
    'Unemployment Rate': custom_unemp,
    'GDP Growth': custom_gdp,
    'Used Car Price Change': custom_used_car_price
}


# --- Forecasting Expected Losses for Current Portfolio Across Scenarios ---
st.subheader("Expected Loss Forecasting")

# Prepare current loan data for prediction
current_loan_data_for_prediction_base = current_loan_data[[
    'Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'LTV at Origination', 'Vehicle Type'
]].copy()

scenario_summary = {}
for scenario_name, macro_factors in forecast_scenarios.items():
    # Create a temporary DataFrame for prediction, adding scenario-specific macro factors
    scenario_prediction_df = current_loan_data_for_prediction_base.copy()
    scenario_prediction_df['Historical Unemployment Rate'] = macro_factors['Unemployment Rate']
    scenario_prediction_df['Historical Used Car Price Change'] = macro_factors['Used Car Price Change']

    # Predict PD
    current_loan_data[f'PD ({scenario_name})'] = pd_model_pipeline.predict_proba(
        scenario_prediction_df[pd_features]
    )[:, 1]
    current_loan_data[f'PD ({scenario_name})'] = np.clip(current_loan_data[f'PD ({scenario_name})'], 0.0001, 0.9999)

    # Predict LGD
    current_loan_data[f'LGD ({scenario_name})'] = lgd_model_pipeline.predict(
        scenario_prediction_df[lgd_features]
    )
    current_loan_data[f'LGD ({scenario_name})'] = np.clip(current_loan_data[f'LGD ({scenario_name})'], 0.01, 0.99)

    # Calculate Expected Loss
    current_loan_data[f'Expected Loss (EL) ({scenario_name})'] = (
        current_loan_data['Remaining Balance ($)'] *
        current_loan_data[f'PD ({scenario_name})'] *
        current_loan_data[f'LGD ({scenario_name})']
    )

    total_el_for_scenario = current_loan_data[f'Expected Loss (EL) ({scenario_name})'].sum()
    total_portfolio_value = current_loan_data['Remaining Balance ($)'].sum() # Total portfolio value is constant
    portfolio_loss_rate = total_el_for_scenario / total_portfolio_value
    scenario_summary[scenario_name] = {
        'Total EL ($)': total_el_for_scenario,
        'Portfolio Loss Rate (%)': portfolio_loss_rate
    }


# --- Prepare Data for Excel Export ---
# Ensure all relevant dataframes are ready
# Summary dataframes
summary_metrics_df = pd.DataFrame(scenario_summary).T.reset_index().rename(columns={'index': 'Scenario'})
scenario_impact_data = []
if 'Base Case' in scenario_summary:
    base_el = scenario_summary['Base Case']['Total EL ($)']
    for scenario_name, summary_data in scenario_summary.items():
        if scenario_name != 'Base Case':
            impact_dollars = summary_data['Total EL ($)'] - base_el
            impact_percent = (impact_dollars / base_el) * 100
            scenario_impact_data.append({
                "Scenario": scenario_name,
                "Increase in Total EL ($)": impact_dollars,
                "Percentage Increase in EL (%)": impact_percent
            })
impact_df = pd.DataFrame(scenario_impact_data).set_index("Scenario")

# Grouped dataframes
def assign_pd_bucket_current(fico):
    if fico >= 750: return "Excellent"
    elif fico >= 700: return "Good"
    elif fico >= 650: return "Fair"
    elif fico >= 600: return "Subprime-Mid"
    else: return "Subprime-Low"
current_loan_data['PD Bucket'] = current_loan_data['Customer FICO'].apply(assign_pd_bucket_current)

el_distribution_base_fico = current_loan_data.groupby('PD Bucket')['Expected Loss (EL) (Base Case)'].sum().reindex(
    ["Excellent", "Good", "Fair", "Subprime-Mid", "Subprime-Low"]
).reset_index()
el_distribution_base_fico.columns = ['FICO Bucket', 'Expected Loss (Base Case)']

el_by_vintage = current_loan_data.groupby('Origination Year')['Expected Loss (EL) (Base Case)'].sum().reset_index()
el_by_vintage.columns = ['Origination Year', 'Expected Loss (Base Case)']

el_by_vehicle_type = current_loan_data.groupby('Vehicle Type')['Expected Loss (EL) (Base Case)'].sum().reset_index()
el_by_vehicle_type.columns = ['Vehicle Type', 'Expected Loss (Base Case)']

# Removed EL by State from export as per user request
# el_by_state = current_loan_data.groupby('State')['Expected Loss (EL) (Base Case)'].sum().reset_index()
# el_by_state.columns = ['State', 'Expected Loss (Base Case)']


dataframes_to_export = {
    "Historical_Loan_Data": historical_data,
    "Current_Loan_Portfolio": current_loan_data,
    "Macroeconomic_Data": macro_data.reset_index() if macro_data is not None else pd.DataFrame({"Note": ["Macro data not available"]}),
    "Portfolio_Summary_Metrics": summary_metrics_df,
    "Scenario_Impact_Summary": impact_df,
    "EL_by_FICO_Bucket": el_distribution_base_fico,
    "EL_by_Vintage": el_by_vintage,
    "EL_by_Vehicle_Type": el_by_vehicle_type,
    # "EL_by_State": el_by_state # Removed from export
}


# --- Display Data ---
st.header("Data Used in the Model")
st.markdown("This section provides a glimpse into the raw and processed data used by the forecasting model.")

# Add the Export to Excel Button prominently
st.download_button(
    label="⬇️ Download All Model Data to Excel",
    data=to_excel_buffer(dataframes_to_export),
    file_name="Auto_Loan_Loss_Forecast_Data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Download all generated loan data, macroeconomic data, and summary tables into a single Excel file."
)

with st.expander("Show Simulated Historical Loan Data (Sample)"):
    st.write("This data is generated to train the PD and LGD models. It simulates past loan performance including defaults and recoveries.")
    st.dataframe(historical_data.head())

with st.expander("Show Generated Current Loan Portfolio Data (Sample)"):
    st.write("This is the synthetic portfolio for which losses are being forecasted.")
    st.dataframe(current_loan_data.head())

# Filtered Current Loan Portfolio View
st.subheader("Filterable Current Loan Portfolio Data")
st.write("Apply filters to explore specific segments of the portfolio and see how summary metrics and charts update.")

# Interactive Filters
# Adjust columns for filters as 'State' is removed
filter_col1, filter_col2 = st.columns(2) # Changed from 3 columns to 2
with filter_col1:
    selected_fico_buckets = st.multiselect(
        "Filter by FICO Segment:",
        options=["Excellent", "Good", "Fair", "Subprime-Mid", "Subprime-Low"],
        default=["Excellent", "Good", "Fair", "Subprime-Mid", "Subprime-Low"] # All selected by default
    )
with filter_col2:
    selected_vehicle_types = st.multiselect(
        "Filter by Vehicle Type:",
        options=current_loan_data['Vehicle Type'].unique().tolist(),
        default=current_loan_data['Vehicle Type'].unique().tolist()
    )
# Removed State filter as per user request
# with filter_col3:
#     selected_states = st.multiselect(
#         "Filter by State:",
#         options=current_loan_data['State'].unique().tolist(),
#         default=current_loan_data['State'].unique().tolist()
#     )

# Apply filters - Removed state filter from logic
filtered_loan_data = current_loan_data[
    current_loan_data['PD Bucket'].isin(selected_fico_buckets) &
    current_loan_data['Vehicle Type'].isin(selected_vehicle_types)
    # & current_loan_data['State'].isin(selected_states) # Removed state filter
]

st.write(f"Displaying {len(filtered_loan_data)} out of {len(current_loan_data)} loans in filtered view.")
st.dataframe(filtered_loan_data.head(10)) # Show head of filtered data


# --- Model Interpretability ---
st.header("Model Interpretability")
st.markdown("""
    Understanding how the machine learning models make their predictions is crucial.
    The coefficients below indicate the direction and relative strength of each feature's impact on PD and LGD.
    * **Positive Coefficient:** Feature value increases, predicted PD/LGD increases.
    * **Negative Coefficient:** Feature value increases, predicted PD/LGD decreases.
    *(Note: These are based on scaled features)*
""")
interpret_col1, interpret_col2 = st.columns(2)
with interpret_col1:
    st.subheader("PD Model (Logistic Regression) Coefficients")
    # Get feature names after preprocessing
    pd_feature_names_out = pd_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    pd_coeffs = pd.DataFrame(
        {'Feature': pd_feature_names_out, 'Coefficient': pd_model_pipeline.named_steps['classifier'].coef_[0]}
    )
    st.dataframe(pd_coeffs.set_index('Feature'))

with interpret_col2:
    st.subheader("LGD Model (Linear Regression) Coefficients")
    # Get feature names after preprocessing, including one-hot encoded Vehicle Type
    lgd_feature_names_out = lgd_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    lgd_coeffs = pd.DataFrame(
        {'Feature': lgd_feature_names_out, 'Coefficient': lgd_model_pipeline.named_steps['regressor'].coef_}
    )
    st.dataframe(lgd_coeffs.set_index('Feature'))


# --- Consolidated Portfolio Summary (using filtered data) ---
st.header("Filtered Portfolio Summary & Scenario Analysis") # Updated header for filtered view

st.markdown("""
    This section presents the aggregated expected losses for the **filtered portfolio** under different economic scenarios.
    **What to look for:** Observe how 'Total Expected Loss' and 'Portfolio Loss Rate' change when you apply filters above. This highlights risk concentrations within specific segments.
""")

col_summary_display = st.columns(len(forecast_scenarios))
for i, (scenario_name, macro_factors) in enumerate(forecast_scenarios.items()):
    # Recalculate summary metrics for the filtered data
    scenario_prediction_df_filtered = filtered_loan_data[[
        'Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'LTV at Origination', 'Vehicle Type'
    ]].copy()
    scenario_prediction_df_filtered['Historical Unemployment Rate'] = macro_factors['Unemployment Rate']
    scenario_prediction_df_filtered['Historical Used Car Price Change'] = macro_factors['Used Car Price Change']

    pd_filtered = pd_model_pipeline.predict_proba(scenario_prediction_df_filtered[pd_features])[:, 1]
    lgd_filtered = lgd_model_pipeline.predict(scenario_prediction_df_filtered[lgd_features])
    el_filtered = filtered_loan_data['Remaining Balance ($)'] * np.clip(pd_filtered, 0.0001, 0.9999) * np.clip(lgd_filtered, 0.01, 0.99)

    total_el_filtered = el_filtered.sum()
    total_portfolio_value_filtered = filtered_loan_data['Remaining Balance ($)'].sum()
    portfolio_loss_rate_filtered = total_el_filtered / total_portfolio_value_filtered if total_portfolio_value_filtered > 0 else 0

    with col_summary_display[i]:
        st.subheader(f"{scenario_name} (Filtered)") # Updated subheader to indicate filter
        st.metric(label="Total Expected Loss ($)", value=f"${total_el_filtered :,.0f}")
        st.metric(label="Portfolio Loss Rate (%)", value=f"{portfolio_loss_rate_filtered:.2%}")

st.subheader("Scenario Impact (Relative to Base Case of Filtered Portfolio)")
# Recalculate impact_df for filtered data
filtered_impact_data = []
if 'Base Case' in forecast_scenarios and 'Base Case' in scenario_summary: # Ensure Base Case exists
    base_scenario_prediction_df_filtered = filtered_loan_data[[
        'Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'LTV at Origination', 'Vehicle Type'
    ]].copy()
    base_scenario_prediction_df_filtered['Historical Unemployment Rate'] = forecast_scenarios['Base Case']['Unemployment Rate']
    base_scenario_prediction_df_filtered['Historical Used Car Price Change'] = forecast_scenarios['Base Case']['Used Car Price Change']
    base_pd_filtered = pd_model_pipeline.predict_proba(base_scenario_prediction_df_filtered[pd_features])[:, 1]
    base_lgd_filtered = lgd_model_pipeline.predict(base_scenario_prediction_df_filtered[lgd_features])
    base_el_filtered_calc = (filtered_loan_data['Remaining Balance ($)'] * np.clip(base_pd_filtered, 0.0001, 0.9999) * np.clip(base_lgd_filtered, 0.01, 0.99)).sum()

    for scenario_name, macro_factors in forecast_scenarios.items():
        if scenario_name != 'Base Case':
            scenario_prediction_df_filtered = filtered_loan_data[[
                'Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'LTV at Origination', 'Vehicle Type'
            ]].copy()
            scenario_prediction_df_filtered['Historical Unemployment Rate'] = macro_factors['Unemployment Rate']
            scenario_prediction_df_filtered['Historical Used Car Price Change'] = macro_factors['Used Car Price Change']
            pd_filtered = pd_model_pipeline.predict_proba(scenario_prediction_df_filtered[pd_features])[:, 1]
            lgd_filtered = lgd_model_pipeline.predict(scenario_prediction_df_filtered[lgd_features])
            el_filtered = (filtered_loan_data['Remaining Balance ($)'] * np.clip(pd_filtered, 0.0001, 0.9999) * np.clip(lgd_filtered, 0.01, 0.99)).sum()

            impact_dollars = el_filtered - base_el_filtered_calc
            impact_percent = (impact_dollars / base_el_filtered_calc) * 100 if base_el_filtered_calc != 0 else 0
            filtered_impact_data.append({
                "Scenario": scenario_name,
                "Increase in Total EL ($)": impact_dollars,
                "Percentage Increase in EL (%)": impact_percent
            })
    st.dataframe(pd.DataFrame(filtered_impact_data).set_index("Scenario"))


# --- Advanced Analysis & Plots (using filtered data) ---
st.header("Advanced Portfolio Analysis (Filtered View)") # Updated header for filtered view
st.markdown("""
    These charts provide deeper insights into the **filtered portfolio's** risk distribution across various segments.
""")
sns.set_style("whitegrid")

# Plot 1: Total Expected Loss across Scenarios (Filtered)
st.subheader("Total Expected Loss Across Macroeconomic Scenarios (Filtered)")
st.markdown("""
    This chart visually compares the total dollar amount of expected losses under each macroeconomic scenario.
    **What to look for:** The increasing height of the bars from 'Base Case' to 'Severe Recession' clearly illustrates the impact of economic downturns on forecasted losses.
""")
scenario_els_filtered = [
    (filtered_loan_data['Remaining Balance ($)'] *
     np.clip(pd_model_pipeline.predict_proba(filtered_loan_data_for_prediction[pd_features])[:,1], 0.0001, 0.9999) *
     np.clip(lgd_model_pipeline.predict(filtered_loan_data_for_prediction[lgd_features]), 0.01, 0.99)).sum()
    for s_name, macro_factors in forecast_scenarios.items()
    for filtered_loan_data_for_prediction in [
        (filtered_loan_data[[
            'Customer FICO', 'Loan Age (months)', 'DTI Ratio', 'LTV at Origination', 'Vehicle Type'
        ]].copy().assign(
            **{'Historical Unemployment Rate': macro_factors['Unemployment Rate'],
               'Historical Used Car Price Change': macro_factors['Used Car Price Change']}
        ))
    ]
]
scenario_names_for_plot = list(forecast_scenarios.keys())

fig1, ax1 = plt.subplots(figsize=(12, 7))
sns.barplot(x=scenario_names_for_plot, y=scenario_els_filtered, palette='viridis', hue=scenario_names_for_plot, legend=False, ax=ax1)
ax1.set_title('Total Expected Loss Across Macroeconomic Scenarios (Filtered)', fontsize=16)
ax1.set_ylabel('Expected Loss ($)', fontsize=12)
ax1.set_xlabel('Scenario', fontsize=12)
ax1.ticklabel_format(style='plain', axis='y')
# Add values on top of bars
for container in ax1.containers:
    ax1.bar_label(container, fmt='${:,.0f}')
plt.tight_layout()
st.pyplot(fig1)
buf1 = io.BytesIO()
fig1.savefig(buf1, format="png")
st.download_button(label="Download Plot 1 (Scenarios) PNG", data=buf1.getvalue(), file_name="el_scenarios_filtered.png", mime="image/png")


# Plot 2: Expected Loss Distribution by FICO Score (Base Case - Filtered)
st.subheader("Expected Loss Distribution by FICO Segment (Base Case - Filtered)")
el_distribution_base_fico_filtered = filtered_loan_data.groupby('PD Bucket')['Expected Loss (EL) (Base Case)'].sum().reindex(
    ["Excellent", "Good", "Fair", "Subprime-Mid", "Subprime-Low"]
).reset_index()
el_distribution_base_fico_filtered.columns = ['FICO Bucket', 'Expected Loss (Base Case)']

fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.barplot(x=el_distribution_base_fico_filtered['FICO Bucket'], y=el_distribution_base_fico_filtered['Expected Loss (Base Case)'], palette='coolwarm', hue=el_distribution_base_fico_filtered['FICO Bucket'], legend=False, ax=ax2)
ax2.set_title('Expected Loss Distribution by FICO Segment (Base Case - Filtered)', fontsize=16)
ax2.set_xlabel('FICO Segment', fontsize=12)
ax2.set_ylabel('Expected Loss ($)', fontsize=12)
ax2.ticklabel_format(style='plain', axis='y')
# Add values on top of bars
for container in ax2.containers:
    ax2.bar_label(container, fmt='${:,.0f}')
plt.tight_layout()
st.pyplot(fig2)
buf2 = io.BytesIO()
fig2.savefig(buf2, format="png")
st.download_button(label="Download Plot 2 (FICO) PNG", data=buf2.getvalue(), file_name="el_fico_filtered.png", mime="image/png")


# Plot 3: Expected Loss by Origination Vintage (Base Case - Filtered)
st.subheader("Expected Loss by Origination Vintage (Base Case - Filtered)")
el_by_vintage_filtered = filtered_loan_data.groupby('Origination Year')['Expected Loss (EL) (Base Case)'].sum().reset_index()
el_by_vintage_filtered.columns = ['Origination Year', 'Expected Loss (Base Case)']

fig3, ax3 = plt.subplots(figsize=(12, 7))
sns.lineplot(x=el_by_vintage_filtered['Origination Year'], y=el_by_vintage_filtered['Expected Loss (Base Case)'], marker='o', color='purple', ax=ax3)
ax3.set_title('Expected Loss by Origination Vintage (Base Case - Filtered)', fontsize=16)
ax3.set_xlabel('Origination Year', fontsize=12)
ax3.set_ylabel('Expected Loss ($)', fontsize=12)
ax3.ticklabel_format(style='plain', axis='y')
# Add values above data points for line plot (requires a bit more manual work for precise placement,
# but can be done with ax.text. For simplicity, we'll skip direct label on line plot,
# but if the user insists, we can add it later using ax.text for each point.)
# Example for bar_label not directly applicable to line plots without custom iteration.
# If values are desired on line plot points, let me know, and I can add more complex code.
plt.tight_layout()
st.pyplot(fig3)
buf3 = io.BytesIO()
fig3.savefig(buf3, format="png")
st.download_button(label="Download Plot 3 (Vintage) PNG", data=buf3.getvalue(), file_name="el_vintage_filtered.png", mime="image/png")


# Plot 4: Expected Loss by Vehicle Type (Base Case - Filtered)
st.subheader("Expected Loss by Vehicle Type (Base Case - Filtered)")
el_by_vehicle_type_filtered = filtered_loan_data.groupby('Vehicle Type')['Expected Loss (EL) (Base Case)'].sum().reset_index()
el_by_vehicle_type_filtered.columns = ['Vehicle Type', 'Expected Loss (Base Case)']

fig4, ax4 = plt.subplots(figsize=(12, 7))
sns.barplot(x=el_by_vehicle_type_filtered['Vehicle Type'], y=el_by_vehicle_type_filtered['Expected Loss (Base Case)'], palette='magma', hue=el_by_vehicle_type_filtered['Vehicle Type'], legend=False, ax=ax4)
ax4.set_title('Expected Loss by Vehicle Type (Base Case - Filtered)', fontsize=16)
ax4.set_xlabel('Vehicle Type', fontsize=12)
ax4.set_ylabel('Expected Loss ($)', fontsize=12)
ax4.ticklabel_format(style='plain', axis='y')
# Add values on top of bars
for container in ax4.containers:
    ax4.bar_label(container, fmt='${:,.0f}')
plt.tight_layout()
st.pyplot(fig4)
buf4 = io.BytesIO()
fig4.savefig(buf4, format="png")
st.download_button(label="Download Plot 4 (Vehicle Type) PNG", data=buf4.getvalue(), file_name="el_vehicle_type_filtered.png", mime="image/png")


st.markdown("---")
st.subheader("Model Capabilities Summary")
st.write("""
This Streamlit application demonstrates a highly sophisticated framework for an auto loan loss forecasting system, showcasing comprehensive data handling, statistical modeling, and scenario analysis skills.

1.  **Synthetic Data Generation (Bank-like Portfolio):** Creates its own historical and current loan portfolios internally, generating new data on each run/refresh, with FICO and balance distributions resembling a large bank's portfolio.
2.  **Real Macroeconomic Data Integration:** Fetches historical Unemployment Rate and GDP data from FRED (Federal Reserve Economic Data) using `pandas_datareader`.
3.  **Predictive Modeling for PD & LGD:** Uses Logistic Regression for PD and Linear Regression for LGD, trained on simulated historical data that incorporates macro factors.
4.  **Dynamic Multi-Scenario Forecasting:** Applies the trained models to the current portfolio under explicitly defined macroeconomic scenarios (Base, Mild Recession, Severe Recession), leveraging real historical macro data points for stress definitions.
5.  **Granular Portfolio Segmentation & Vintage Analysis:** Breaks down Expected Losses by FICO segment, Origination Vintage, and Vehicle Type with corresponding visualizations.
6.  **Automated & Advanced Visualization:** Generates professional-grade plots to illustrate key findings dynamically.
""")









