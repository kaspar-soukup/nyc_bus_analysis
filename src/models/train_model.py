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
import statsmodels.formula.api as smf
from typing import Tuple

from .did_model import build_did_outputs  # noqa: F401


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
    # Aggregate ridership data to match bus speed granularity: year, month, day_of_week, hour
    ridership_agg = ridership_data.groupby(['year', 'month', 'day_of_week', 'hour']).agg({
        'total_ridership': 'mean',
        'weekend': 'first'
    }).reset_index()

    # Aggregate vehicles entering Manhattan: year, month, day, hour (normalize to lowercase)
    manhattan_agg = vehicles_entering_manhattan.groupby(['year', 'month', 'day', 'hour']).agg({
        'inflow': 'mean',
        'outflow': 'mean',
        'change': 'mean',
        'weekend': 'first'
    }).reset_index()
    
    # Map day of week
    manhattan_agg['day_of_week'] = pd.to_datetime(
        manhattan_agg[['year', 'month', 'day']].rename(columns={'year': 'year', 'month': 'month', 'day': 'day'})
    ).dt.day_name()

    # Aggregate CRZ entries: year, month, day, hour
    crz_agg = vehicles_entering_cbd.groupby(['year', 'month', 'day', 'hour']).agg({
        'sum_crz_entries': 'mean',
        'weekend': 'first'
    }).reset_index()
    
    # Map day of week for CRZ
    crz_agg['day_of_week'] = pd.to_datetime(
        crz_agg[['year', 'month', 'day']].rename(columns={'year': 'year', 'month': 'month', 'day': 'day'})
    ).dt.day_name()

    # CBD speeds: aggregate to year, month
    # First, extract year and month from the month column
    cbd_vehicle_speeds_2023_2025['month_dt'] = pd.to_datetime(cbd_vehicle_speeds_2023_2025['month'])
    cbd_vehicle_speeds_2023_2025['year'] = cbd_vehicle_speeds_2023_2025['month_dt'].dt.year
    cbd_vehicle_speeds_2023_2025['month'] = cbd_vehicle_speeds_2023_2025['month_dt'].dt.month
    
    # Filter to CBD zone only and aggregate
    cbd_speeds_agg = cbd_vehicle_speeds_2023_2025[
        cbd_vehicle_speeds_2023_2025['zone'] == 'CBD'
    ].groupby(['year', 'month']).agg({
        'zonal_speed': 'mean'
    }).reset_index().rename(columns={'zonal_speed': 'average_speed'})

    # CBD bus routes: add a flag for CBD routes
    cbd_routes_flag = cbd_bus_routes_2025[['route_id', 'cbd_relation']].copy()
    cbd_routes_flag['in_cbd'] = (cbd_routes_flag['cbd_relation'] == 'In CBD').astype(int)

    # Start with segment_speed_df
    combined_df = segment_speed_df.copy()

    # Prepare ridership data for merge - rename hour back to match
    ridership_agg_prep = ridership_agg.rename(columns={'hour': 'hour_of_day'}).copy()
    combined_df = combined_df.merge(
        ridership_agg_prep[['year', 'month', 'day_of_week', 'hour_of_day', 'total_ridership']], 
        on=['year', 'month', 'day_of_week', 'hour_of_day'],
        how='left'
    )

    # Prepare Manhattan crossings data for merge
    manhattan_agg_prep = manhattan_agg.rename(columns={'hour': 'hour_of_day', 'day': 'day_temp'}).copy()
    combined_df = combined_df.merge(
        manhattan_agg_prep[['year', 'month', 'day_of_week', 'hour_of_day', 'inflow', 'outflow', 'change']],
        on=['year', 'month', 'day_of_week', 'hour_of_day'],
        suffixes=('_segment', '_manhattan'),
        how='left'
    )

    # Prepare CRZ entries data for merge
    crz_agg_prep = crz_agg.rename(columns={'hour': 'hour_of_day', 'day': 'day_temp'}).copy()
    combined_df = combined_df.merge(
        crz_agg_prep[['year', 'month', 'day_of_week', 'hour_of_day', 'sum_crz_entries']],
        on=['year', 'month', 'day_of_week', 'hour_of_day'],
        how='left'
    )

    # Merge CBD speeds (on year, month)
    combined_df = combined_df.merge(
        cbd_speeds_agg, 
        on=['year', 'month'],
        how='left'
    )

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

    # Convert year to categorical (string) instead of integer
    combined_df['year'] = combined_df['year'].astype(str)

    # Prepare features for regression
    features = ['year', 'month', 'hour_of_day', 'total_ridership', 'inflow', 
                'outflow', 'change', 'sum_crz_entries', 'average_speed', 'in_cbd']
    
    # Add dummies for categorical: year, day_of_week, route_type, direction_id
    combined_df = pd.get_dummies(combined_df, columns=['year', 'day_of_week', 'route_type', 'direction_id'], 
                                  drop_first=True)
    features.extend([col for col in combined_df.columns if col.startswith(('year_', 'day_of_week_', 'route_type_', 'direction_id_'))])

    X = combined_df[features]
    y = combined_df['average_road_speed']

    # Handle any remaining NaNs
    X = X.fillna(X.mean())

    return X, y


def train_linear_regression(X: pd.DataFrame, y: pd.Series, 
                           test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[LinearRegression, dict]:
    """
    Train a linear regression model and evaluate performance with statistical details.
    
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
    
    # Calculate p-values using statsmodels on full training set for robust inference
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    
    # Extract coefficients and p-values
    coefficients_df = pd.DataFrame({
        'Feature': ['intercept'] + list(X.columns),
        'Coefficient': [model.intercept_] + list(model.coef_),
        'P-Value': ols_model.pvalues.values
    })
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'coefficients_df': coefficients_df,
        'model_summary': ols_model.summary()
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


def prepare_difference_in_differences_data(
    segment_speed_df: pd.DataFrame,
    ridership_data: pd.DataFrame,
    vehicles_entering_manhattan: pd.DataFrame,
    vehicles_entering_cbd: pd.DataFrame,
    cbd_vehicle_speeds_2023_2025: pd.DataFrame,
    cbd_bus_routes_2025: pd.DataFrame,
    bus_speeds_2023_2024: pd.DataFrame,
    bus_speeds_2025: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare aggregated route-month data for the DiD regression focused on the CBD."""

    if bus_speeds_2023_2024.empty or bus_speeds_2025.empty:
        raise ValueError("Bus speed datasets for 2024 and 2025 are required for the DiD analysis.")

    speeds = pd.concat([bus_speeds_2023_2024, bus_speeds_2025], ignore_index=True)

    speeds['month'] = pd.to_datetime(speeds['month'])
    speeds['year'] = speeds['month'].dt.year
    speeds['month_num'] = speeds['month'].dt.month

    # Keep Manhattan service only
    speeds = speeds[speeds['borough'] == 'Manhattan'].copy()

    # Focus on the four months before and after Jan 2025 (Sep–Dec 2024 vs Jan–Apr 2025)
    window_mask = (
        ((speeds['year'] == 2024) & (speeds['month_num'].between(9, 12))) |
        ((speeds['year'] == 2025) & (speeds['month_num'].between(1, 4)))
    )
    speeds = speeds[window_mask].copy()

    if speeds.empty:
        raise ValueError("No Manhattan bus speed data available for the requested DiD window.")

    # Flag CBD routes (treatment) using the official list
    cbd_routes_in = set(
        cbd_bus_routes_2025.loc[cbd_bus_routes_2025['cbd_relation'] == 'In CBD', 'route_id']
    )
    speeds['treatment'] = speeds['route_id'].isin(cbd_routes_in).astype(int)
    speeds['group'] = np.where(speeds['treatment'] == 1, 'CBD', 'Non-CBD')

    # Weighted average by mileage so longer service carries more weight
    def weighted_speed(group: pd.DataFrame) -> float:
        mileage = pd.to_numeric(group['total_mileage'], errors='coerce').fillna(0)
        avg_speed = pd.to_numeric(group['average_speed'], errors='coerce').fillna(np.nan)
        if mileage.sum() == 0 or avg_speed.isna().all():
            return avg_speed.mean()
        return np.average(avg_speed.fillna(0), weights=mileage)

    agg = (
        speeds
        .groupby(['route_id', 'group', 'treatment', 'month'])
        .apply(lambda g: pd.Series({'avg_speed': weighted_speed(g)}))
        .reset_index()
    )

    agg['year'] = agg['month'].dt.year
    agg['month_num'] = agg['month'].dt.month
    agg['post'] = (agg['year'] == 2025).astype(int)
    agg['month_label'] = agg['month'].dt.strftime('%Y-%m')

    # Pre/post summary table for quick inspection
    pre_post = (
        agg.groupby(['group', 'post'])['avg_speed']
        .mean()
        .unstack()
        .rename(columns={0: 'pre', 1: 'post'})
    )

    if 'CBD' in pre_post.index and 'Non-CBD' in pre_post.index:
        pre_post['change'] = pre_post['post'] - pre_post['pre']
        did_val = pre_post.loc['CBD', 'change'] - pre_post.loc['Non-CBD', 'change']
        pre_post.loc['DiD', ['pre', 'post', 'change']] = [np.nan, np.nan, did_val]
    else:
        pre_post['change'] = np.nan
        pre_post.loc['DiD', ['pre', 'post', 'change']] = [np.nan, np.nan, np.nan]

    return agg, pre_post


def train_difference_in_differences(
    aggregated_df: pd.DataFrame,
    pre_post_summary: pd.DataFrame
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, dict]:
    """Run the DiD regression using the aggregated route-month panel."""

    formula = "avg_speed ~ post * treatment + C(route_id) + C(month_label)"
    did_model = smf.ols(formula, data=aggregated_df).fit()

    param_key = 'post:treatment'
    f_stat = getattr(did_model, 'f_statistic', 'N/A')
    if hasattr(f_stat, 'statistic'):
        f_stat = f_stat.statistic

    results = {
        'model': did_model,
        'model_summary': did_model.summary(),
        'coefficients_df': pd.DataFrame({
            'Feature': did_model.params.index,
            'Coefficient': did_model.params.values,
            'P-Value': did_model.pvalues.values,
            'Std_Error': did_model.bse.values,
            '95%_CI_Lower': did_model.conf_int()[0].values,
            '95%_CI_Upper': did_model.conf_int()[1].values
        }),
        'r_squared': did_model.rsquared,
        'adj_r_squared': did_model.rsquared_adj,
        'f_statistic': f_stat,
        'n_obs': did_model.nobs,
        'did_param': param_key,
        'pre_post_summary': pre_post_summary
    }

    print("\n" + "="*80)
    print("CBD-FOCUSED DIFFERENCE-IN-DIFFERENCES REGRESSION")
    print("="*80)
    print(f"Observations: {results['n_obs']}")
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {results['adj_r_squared']:.4f}")
    print(f"F-statistic: {results['f_statistic']}")

    if param_key in did_model.params.index:
        did_coef = did_model.params[param_key]
        did_pval = did_model.pvalues[param_key]
        print(f"DiD Coefficient (Treatment Effect): {did_coef:.6f}")
        print(f"P-value: {did_pval:.6f}")
        print(
            "Statistical Significance: "
            f"{'***' if did_pval < 0.01 else '**' if did_pval < 0.05 else '*' if did_pval < 0.10 else 'Not significant'}"
        )

    print("\nMonth + route fixed effects are included via categorical dummies.")
    print("Full model summary:")
    print(did_model.summary())
    print("="*80 + "\n")

    return did_model, results


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
