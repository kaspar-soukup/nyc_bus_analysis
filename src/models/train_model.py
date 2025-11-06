"""
Models module for regression analysis.

This module contains model training and evaluation functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from typing import Tuple


def prepare_regression_data(segment_speed_df: pd.DataFrame,
                            ridership_data: pd.DataFrame,
                            vehicles_entering_manhattan: pd.DataFrame,
                            vehicles_entering_cbd: pd.DataFrame,
                            cbd_vehicle_speeds_2023_2025: pd.DataFrame,
                            cbd_bus_routes_2025: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare and merge all datasets for regression analysis.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - ridership_data: DataFrame with ridership data
    - vehicles_entering_manhattan: DataFrame with Manhattan crossing data
    - vehicles_entering_cbd: DataFrame with CBD entry data
    - cbd_vehicle_speeds_2023_2025: DataFrame with CBD vehicle speeds
    - cbd_bus_routes_2025: DataFrame with CBD bus routes
    
    Returns:
    - Tuple of (X, y) where X is feature matrix and y is target variable
    """
    # Aggregate ridership data to match bus speed granularity: Year, Month, day_of_week, Hour
    ridership_agg = ridership_data.groupby(['year', 'month', 'day_of_week', 'hour']).agg({
        'total_ridership': 'mean',
        'weekend': 'first'
    }).reset_index()

    # Aggregate vehicles entering Manhattan: Year, Month, day_of_week, Hour
    manhattan_agg = vehicles_entering_manhattan.groupby(['Year', 'Month', 'Day', 'Hour']).agg({
        'inflow': 'mean',
        'outflow': 'mean',
        'change': 'mean',
        'Weekend': 'first'
    }).reset_index()
    
    # Map day of week
    manhattan_agg['day_of_week'] = pd.to_datetime(
        manhattan_agg[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
    ).dt.day_name()

    # Aggregate CRZ entries: Year, Month, day_of_week, Hour
    crz_agg = vehicles_entering_cbd.groupby(['Year', 'Month', 'Day', 'Hour']).agg({
        'sum_crz_entries': 'mean',
        'Weekend': 'first'
    }).reset_index()
    
    # Map day of week for CRZ
    crz_agg['day_of_week'] = pd.to_datetime(
        crz_agg[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
    ).dt.day_name()

    # CBD speeds: aggregate to Year, Month
    # First, extract year and month from the month column
    cbd_vehicle_speeds_2023_2025['month_dt'] = pd.to_datetime(cbd_vehicle_speeds_2023_2025['month'])
    cbd_vehicle_speeds_2023_2025['Year'] = cbd_vehicle_speeds_2023_2025['month_dt'].dt.year
    cbd_vehicle_speeds_2023_2025['Month'] = cbd_vehicle_speeds_2023_2025['month_dt'].dt.month
    
    # Filter to CBD zone only and aggregate
    cbd_speeds_agg = cbd_vehicle_speeds_2023_2025[
        cbd_vehicle_speeds_2023_2025['zone'] == 'CBD'
    ].groupby(['Year', 'Month']).agg({
        'zonal_speed': 'mean'
    }).reset_index().rename(columns={'zonal_speed': 'Average Speed'})

    # CBD bus routes: add a flag for CBD routes
    cbd_routes_flag = cbd_bus_routes_2025[['route_id', 'cbd_relation']].copy()
    cbd_routes_flag['in_cbd'] = (cbd_routes_flag['cbd_relation'] == 'In CBD').astype(int)

    # Start with segment_speed_df
    combined_df = segment_speed_df.copy()

    # Merge ridership
    combined_df = combined_df.merge(
        ridership_agg, 
        left_on=['year', 'month', 'day_of_week', 'hour_of_day'],
        right_on=['year', 'month', 'day_of_week', 'hour'],
        how='left'
    )

    # Merge Manhattan crossings
    combined_df = combined_df.merge(
        manhattan_agg, 
        left_on=['year', 'month', 'day_of_week', 'hour_of_day'],
        right_on=['Year', 'Month', 'day_of_week', 'Hour'],
        how='left'
    )

    # Merge CRZ entries
    combined_df = combined_df.merge(
        crz_agg, 
        left_on=['year', 'month', 'day_of_week', 'hour_of_day'],
        right_on=['Year', 'Month', 'day_of_week', 'Hour'],
        how='left'
    )

    # Merge CBD speeds (on Year, Month)
    combined_df = combined_df.merge(
        cbd_speeds_agg, 
        left_on=['year', 'month'], 
        right_on=['Year', 'Month'], 
        how='left'
    )
    
    # Drop duplicate columns
    combined_df.drop(['Year_x', 'Month_x', 'Year_y', 'Month_y', 'Hour_x', 'Hour_y', 'hour'], 
                     axis=1, inplace=True, errors='ignore')

    # Merge CBD route flag
    combined_df = combined_df.merge(cbd_routes_flag[['route_id', 'in_cbd']], on='route_id', how='left')

    # Fill NaNs with 0 or mean where appropriate
    combined_df.fillna({
        'total_ridership': 0,
        'inflow': 0,
        'outflow': 0,
        'change': 0,
        'sum_crz_entries': 0,
        'Average Speed': combined_df['Average Speed'].mean() if 'Average Speed' in combined_df.columns else 0,
        'in_cbd': 0
    }, inplace=True)

    # Prepare features for regression
    features = ['year', 'month', 'hour_of_day', 'total_ridership', 'inflow', 
                'outflow', 'change', 'sum_crz_entries', 'Average Speed', 'in_cbd']
    
    # Add dummies for categorical: day_of_week, route_type, direction_id
    combined_df = pd.get_dummies(combined_df, columns=['day_of_week', 'route_type', 'direction_id'], 
                                  drop_first=True)
    features.extend([col for col in combined_df.columns if col.startswith(('day_of_week_', 'route_type_', 'direction_id_'))])

    X = combined_df[features]
    y = combined_df['average_road_speed']

    # Handle any remaining NaNs
    X = X.fillna(X.mean())

    return X, y


def train_linear_regression(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[LinearRegression, dict]:
    """
    Train a linear regression model and evaluate performance.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - test_size: Proportion of data for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - Tuple of (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R-squared: {r2:.4f}')
    
    return model, metrics


def train_statsmodels_ols(X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, 
                          random_state: int = 42) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Train an OLS model using statsmodels for detailed statistical summary.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - test_size: Proportion of data for testing
    - random_state: Random seed for reproducibility
    
    Returns:
    - Fitted statsmodels OLS results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Using statsmodels for detailed summary
    X_train_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    
    print(model_sm.summary())
    
    return model_sm


def save_model_results(model: LinearRegression, metrics: dict, 
                       feature_names: list, output_dir: str = "reports"):
    """
    Save model results and metrics to files.
    
    Parameters:
    - model: Trained model
    - metrics: Dictionary with metrics
    - feature_names: List of feature names
    - output_dir: Directory to save results
    """
    import os
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    with open(f"{output_dir}/linear_regression_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'MSE': [metrics['mse']],
        'R2': [metrics['r2']]
    })
    metrics_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)
    
    # Save feature importance (coefficients)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    coef_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    print(f"Model results saved to {output_dir}/")
