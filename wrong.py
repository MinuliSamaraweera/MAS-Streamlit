import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib 
matplotlib.use('Agg')  # Use the Agg backend (for headless environments)
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    demographic_data = pd.read_csv('/Users/minu/Desktop/MAS-Streamlit/Dataset/demographic_data_dataset.csv')
    defect_data = pd.read_csv('/Users/minu/Desktop/MAS-Streamlit/Dataset/updated_worker_defect_details.csv')
    return demographic_data, defect_data

demographic_data, defect_data = load_data()

# Combine datasets on Worker_ID
combined_data = pd.merge(defect_data, demographic_data, on='Worker_ID')

# Drop unnecessary columns
fields_to_drop = ['Joining_Date', 'Gender']
combined_data.drop(columns=fields_to_drop, inplace=True)

# Convert Date column to datetime and set Date as index for ARIMA
combined_data['Date'] = pd.to_datetime(combined_data['Date'], infer_datetime_format=True)
combined_data.set_index('Date', inplace=True)

def train_arima_model(data, order=(1, 1, 1), steps=5):
    try:
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return model_fit, forecast
    except Exception as e:
        st.write(f"An error occurred while training the ARIMA model: {e}")
        return None, np.full(steps, np.nan)

def calculate_rmse(observed, forecast):
    if len(observed) == 0:
        return np.nan
    mse = mean_squared_error(observed, forecast)
    rmse = np.sqrt(mse)
    return rmse

def analyze_worker(worker_id, forecast_steps=5):
    worker_data = combined_data[combined_data['Worker_ID'] == worker_id]
    
    if worker_data.empty:
        st.write(f"No data found for worker {worker_id}.")
        return
    
    worker_name = demographic_data.loc[demographic_data['Worker_ID'] == worker_id, 'Name'].values[0]

    # Calculate last week's high and low defect types and their counts
    last_week_data = worker_data.loc[worker_data.index[-7:]]
    last_week_summary = last_week_data[['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']].sum()
    last_week_high_defect_type = last_week_summary.idxmax()
    last_week_low_defect_type = last_week_summary.idxmin()
    last_week_high_defect_count = last_week_summary.max()
    last_week_low_defect_count = last_week_summary.min()

    # Display worker details in bold and larger font size
    st.markdown(f"<h4><b>Worker Name: {worker_name}</b></h4>", unsafe_allow_html=True)
    st.markdown(f"<b>Last week's high defect type: {last_week_high_defect_type} with count: {last_week_high_defect_count}</b>", unsafe_allow_html=True)
    st.markdown(f"<b>Last week's low defect type: {last_week_low_defect_type} with count: {last_week_low_defect_count}</b>", unsafe_allow_html=True)

    # Define categorical and numerical features
    categorical_features = ['Skill_Level']
    numerical_features = ['Age', 'Production_Volume']

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # Separate features and target variables for the worker
    X_worker = worker_data.drop(columns=['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4', 'defect_count', 'count', 'Worker_ID', 'Shift', 'Name'])
    y_worker = worker_data[['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']]

    # Define and train the Random Forest model
    best_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

    # Train the best model on the full worker data
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', best_model)
    ])
    pipeline.fit(X_worker, y_worker)

    # Prepare features for the specific worker for future predictions
    X_worker_future = pd.concat([X_worker.iloc[-1:].copy()] * forecast_steps, ignore_index=True)

    # Generate future predictions using the best traditional model
    y_pred_best_model = pipeline.predict(X_worker_future)

    # Generate ARIMA forecasts for each defect type
    arima_forecasts = {}
    rmses = {}
    for defect_type in ['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']:
        model_fit, forecast = train_arima_model(worker_data[defect_type].dropna(), steps=forecast_steps)
        arima_forecasts[defect_type] = forecast
        rmses[defect_type] = calculate_rmse(worker_data[defect_type][-forecast_steps:], forecast)
    
    # Combine traditional model predictions with ARIMA forecasts
    combined_predictions = {}
    for defect_type in ['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']:
        combined_predictions[defect_type] = (y_pred_best_model[:, ['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4'].index(defect_type)] + arima_forecasts[defect_type]) / 2

    # Print future predictions
    future_dates = pd.date_range(start=worker_data.index[-1], periods=forecast_steps + 1, freq='B')[1:]
    future_predictions = pd.DataFrame(combined_predictions, index=future_dates)

    st.write(f"Future defect predictions for worker {worker_id}:")
    st.write(future_predictions)

    # Plotting the results in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, defect_type in enumerate(['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']):
        axes[i].plot(worker_data.index, worker_data[defect_type], label='Observed', color='blue')
        axes[i].plot(future_dates, future_predictions[defect_type], label='Forecast', linestyle='--', marker='o', color='purple')
        axes[i].set_title(f'Future forecast for {defect_type} (Worker {worker_id})', fontsize=10)
        axes[i].set_xlabel('Date', fontsize=8)
        axes[i].set_ylabel(f'{defect_type} Count', fontsize=8)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Plotting RMSE values
    fig, ax = plt.subplots(figsize=(8, 4))
    defect_types = list(rmses.keys())
    rmse_values = list(rmses.values())
    ax.bar(defect_types, rmse_values, color='orange')
    ax.set_title('RMSE for each defect type', fontsize=10)
    ax.set_xlabel('Defect Type', fontsize=8)
    ax.set_ylabel('RMSE', fontsize=8)
    for i, v in enumerate(rmse_values):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    st.pyplot(fig)

# Function to identify the worker with the highest defect count on the last day
def highest_defect_worker_last_day():
    try:
        last_date = combined_data.index.max()
        last_day_data = combined_data.loc[last_date]

        if last_day_data.empty:
            st.write("No defect data available for the last day.")
            return

        # Sum the defects for each worker
        last_day_data['Total_Defects'] = last_day_data[['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']].sum(axis=1)
        
        # Find the worker with the highest defects
        highest_defect_worker = last_day_data['Total_Defects'].idxmax()
        highest_defect_count = last_day_data.loc[highest_defect_worker, 'Total_Defects']
        
        # Find the defect type with the highest count for this worker
        defect_counts = last_day_data.loc[highest_defect_worker, ['Run_Off_D1', 'Open_Seam_D2', 'SPI_Errors_D3', 'High_Low_D4']]
        highest_defect_type = defect_counts.idxmax()
        highest_defect_type_count = defect_counts.max()
        
        # Search for the worker ID in the demographic dataset
        worker_info = demographic_data[demographic_data['Worker_ID'] == highest_defect_worker]

        if not worker_info.empty:
            highest_defect_worker_name = worker_info['Name'].values[0]
        else:
            highest_defect_worker_name = "Unknown"

        st.markdown(f"### Worker with Highest Defect Count on {last_date.strftime('%Y-%m-%d')}")
        st.write(f"Worker Name: {highest_defect_worker_name}")
        st.write(f"Worker ID: {highest_defect_worker}")
        st.write(f"Total Defects: {highest_defect_count}")
        st.write(f"Highest Defect Type: {highest_defect_type} with count {highest_defect_type_count}")

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Streamlit app layout
st.title("Worker Defect Analysis and Forecasting")

# Prompt user to enter Worker ID for defect prediction
worker_id = st.text_input("Enter the Worker ID:")

if worker_id:
    analyze_worker(worker_id)

# Identify and display the worker with the highest defect count on the last day
highest_defect_worker_last_day()
