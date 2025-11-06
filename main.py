"""
Main pipeline script for NYC Bus Analysis.

This script orchestrates the entire data science pipeline:
1. Data ingestion
2. Data processing and feature engineering
3. Visualization
4. Model training

Usage:
    python main.py --fetch-data --process --visualize --model
    python main.py --visualize  # Only run visualizations
    python main.py --all  # Run full pipeline
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import *
from src.data.make_dataset import fetch_all_data
from src.features.build_features import (
    clean_segment_speed_data,
    clean_stop_data,
    clean_bus_speed_data,
    prepare_ridership_data,
    clean_vehicle_crossing_data,
    clean_crz_entries_data,
    filter_cbd_segments
)
from src.visualization.visualize import save_all_plots
from src.models.train_model import (
    prepare_regression_data,
    train_linear_regression,
    train_statsmodels_ols,
    save_model_results
)


def fetch_data(app_token: str):
    """Fetch or load all data."""
    print("=" * 80)
    print("STEP 1: DATA INGESTION")
    print("=" * 80)
    
    data = fetch_all_data(app_token, str(RAW_DATA_DIR))
    
    print("\n✓ Data ingestion complete!")
    return data


def process_data(data: dict):
    """Process and clean all data."""
    print("\n" + "=" * 80)
    print("STEP 2: DATA PROCESSING AND FEATURE ENGINEERING")
    print("=" * 80)
    
    print("\nProcessing segment speed data...")
    segment_speed_df = clean_segment_speed_data(
        data['bus_speed_seg_2025'],
        data['bus_speed_seg_2023_2024']
    )
    
    print("Processing stop data...")
    stop_data = clean_stop_data(data['stop_data'])
    
    print("Processing bus speed data...")
    bus_speed_df = clean_bus_speed_data(
        data['bus_speed_2025'],
        data['bus_speed_2023_2024']
    )
    
    print("Processing ridership data...")
    ridership_data, ridership_data_weekday, ridership_data_weekend = prepare_ridership_data(
        data['hourly_ridership_2025'],
        data['hourly_ridership_2023_2024']
    )
    
    print("Processing vehicle crossing data...")
    vehicles_entering_manhattan = clean_vehicle_crossing_data(data['hourly_crossings_2023_2025'])
    
    print("Processing CRZ entries data...")
    vehicles_entering_cbd, vehicles_entering_cbd_weekday, vehicles_entering_cbd_weekend = clean_crz_entries_data(
        data['crz_entries_2023_2025']
    )
    
    print("Filtering CBD segments...")
    segment_speed_cbd, segment_speed_cbd_weekday, segment_speed_cbd_weekend = filter_cbd_segments(
        segment_speed_df,
        data['cbd_geojson_area_2024']
    )
    
    # Save processed data
    print("\nSaving processed data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    segment_speed_df.to_csv(PROCESSED_DATA_DIR / 'segment_speed_processed.csv', index=False)
    ridership_data.to_csv(PROCESSED_DATA_DIR / 'ridership_processed.csv', index=False)
    vehicles_entering_manhattan.to_csv(PROCESSED_DATA_DIR / 'manhattan_crossings_processed.csv', index=False)
    vehicles_entering_cbd.to_csv(PROCESSED_DATA_DIR / 'cbd_entries_processed.csv', index=False)
    
    print("✓ Data processing complete!")
    
    return {
        'segment_speed_df': segment_speed_df,
        'stop_data': stop_data,
        'bus_speed_df': bus_speed_df,
        'ridership_data': ridership_data,
        'ridership_data_weekday': ridership_data_weekday,
        'ridership_data_weekend': ridership_data_weekend,
        'vehicles_entering_manhattan': vehicles_entering_manhattan,
        'vehicles_entering_cbd': vehicles_entering_cbd,
        'vehicles_entering_cbd_weekday': vehicles_entering_cbd_weekday,
        'vehicles_entering_cbd_weekend': vehicles_entering_cbd_weekend,
        'segment_speed_cbd': segment_speed_cbd,
        'cbd_bus_routes': data['cbd_bus_routes_2025'],
        'cbd_vehicle_speeds': data['cbd_vehicle_speeds_2023_2025']
    }


def create_visualizations(processed_data: dict):
    """Generate and save all visualizations."""
    print("\n" + "=" * 80)
    print("STEP 3: VISUALIZATION")
    print("=" * 80)
    
    save_all_plots(
        segment_speed_df=processed_data['segment_speed_df'],
        cbd_bus_routes=processed_data['cbd_bus_routes'],
        vehicles_entering_manhattan=processed_data['vehicles_entering_manhattan'],
        ridership_data_weekday=processed_data['ridership_data_weekday'],
        ridership_data_weekend=processed_data['ridership_data_weekend'],
        vehicles_entering_cbd=processed_data['vehicles_entering_cbd'],
        vehicles_entering_cbd_weekday=processed_data['vehicles_entering_cbd_weekday'],
        vehicles_entering_cbd_weekend=processed_data['vehicles_entering_cbd_weekend'],
        output_dir=str(FIGURES_DIR)
    )
    
    print("\n✓ Visualizations complete!")


def train_models(processed_data: dict):
    """Train and evaluate regression models."""
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING")
    print("=" * 80)
    
    print("\nPreparing regression data...")
    X, y = prepare_regression_data(
        segment_speed_df=processed_data['segment_speed_df'],
        ridership_data=processed_data['ridership_data'],
        vehicles_entering_manhattan=processed_data['vehicles_entering_manhattan'],
        vehicles_entering_cbd=processed_data['vehicles_entering_cbd'],
        cbd_vehicle_speeds_2023_2025=processed_data['cbd_vehicle_speeds'],
        cbd_bus_routes_2025=processed_data['cbd_bus_routes']
    )
    
    print("\nTraining Linear Regression model...")
    model, metrics = train_linear_regression(X, y)
    
    print("\nTraining OLS model with statsmodels...")
    try:
        model_sm = train_statsmodels_ols(X, y)
    except Exception as e:
        print(f"Statsmodels OLS failed: {e}")
        print("Skipping statsmodels analysis...")
        model_sm = None
    
    print("\nSaving model results...")
    save_model_results(model, metrics, X.columns.tolist(), output_dir=str(REPORTS_DIR))
    
    print("\n✓ Model training complete!")


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='NYC Bus Analysis Pipeline')
    parser.add_argument('--fetch-data', action='store_true', help='Fetch data from API')
    parser.add_argument('--process', action='store_true', help='Process and clean data')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--model', action='store_true', help='Train models')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Get API token from environment
    app_token = os.environ.get('APP_TOKEN')
    if not app_token and (args.fetch_data or args.all):
        print("ERROR: APP_TOKEN environment variable not set!")
        print("Please set it using: export APP_TOKEN='your_token'")
        return
    
    print("\n" + "=" * 80)
    print("NYC BUS ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Run pipeline based on arguments
    if args.all or args.fetch_data:
        data = fetch_data(app_token)
    else:
        # Load existing data if not fetching
        print("\nLoading existing data...")
        from src.data.make_dataset import (
            load_stop_data, 
            load_cbd_geojson_data
        )
        import pandas as pd
        
        data = {
            'bus_speed_seg_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Route_Segment_Speeds_Beginning_2025_20251105.csv'),
            'bus_speed_seg_2023_2024': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Route_Segment_Speeds_2023_2024.csv'),
            'bus_speed_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Speeds_Beginning_2025_20251105.csv'),
            'bus_speed_2023_2024': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Speeds_2023_2024_20251105.csv'),
            'hourly_ridership_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Hourly_Ridership_Beginning_2025_20251105.csv'),
            'hourly_ridership_2023_2024': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Hourly_Ridership_2023_2024_20251105.csv'),
            'hourly_crossings_2023_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_Bus_Hourly_Crossings_2023_2025_20251105.csv'),
            'crz_entries_2023_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_CRZ_Hourly_Entries_2023_2025_20251105.csv'),
            'cbd_vehicle_speeds_2023_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_CBD_Vehicle_Speeds_2023_2025_20251105.csv'),
            'cbd_bus_routes_2025': pd.read_csv(RAW_DATA_DIR / 'MTA_CBD_Bus_Routes_2025_20251105.csv'),
            'stop_data': load_stop_data(str(RAW_DATA_DIR / 'manhattan_stops_flat.csv')),
            'cbd_geojson_area_2024': load_cbd_geojson_data(str(RAW_DATA_DIR / 'MTA_Central_Business_District_Geofence__Beginning_June_2024_20251105.geojson'))
        }
    
    if args.all or args.process:
        processed_data = process_data(data)
    else:
        # Load processed data if available
        import pandas as pd
        if (PROCESSED_DATA_DIR / 'segment_speed_processed.csv').exists():
            print("\nLoading processed data...")
            processed_data = {
                'segment_speed_df': pd.read_csv(PROCESSED_DATA_DIR / 'segment_speed_processed.csv'),
                'ridership_data': pd.read_csv(PROCESSED_DATA_DIR / 'ridership_processed.csv'),
                'vehicles_entering_manhattan': pd.read_csv(PROCESSED_DATA_DIR / 'manhattan_crossings_processed.csv'),
                'vehicles_entering_cbd': pd.read_csv(PROCESSED_DATA_DIR / 'cbd_entries_processed.csv'),
            }
            # Need to reprocess some data for visualizations
            processed_data.update(process_data(data))
        else:
            processed_data = process_data(data)
    
    if args.all or args.visualize:
        create_visualizations(processed_data)
    
    if args.all or args.model:
        train_models(processed_data)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")
    print(f"Visualizations saved to: {FIGURES_DIR}")
    print(f"Model results saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
