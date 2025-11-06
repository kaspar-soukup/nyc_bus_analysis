"""
Data ingestion module for MTA bus analysis.

This module handles downloading data from the Socrata API and saving it locally.
All data fetching logic is centralized here using a configuration-driven approach.
"""

import pandas as pd
import geopandas as gpd
from sodapy import Socrata
import time
import os
from datetime import datetime
from typing import Optional, Dict, List


# API Dataset Configuration
API_DATASETS = {
    'bus_segment_speed_2025': {
        'dataset_id': 'kufs-yh3x',
        'base_name': 'MTA_Bus_Route_Segment_Speeds_Beginning_2025',
        'query': "SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        'limit': 100000
    },
    'bus_segment_speed_2023_2024': {
        'dataset_id': '58t6-89vi',
        'base_name': 'MTA_Bus_Route_Segment_Speeds_2023_2024',
        'query': "SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        'limit': 100000
    },
    'bus_speed_2025': {
        'dataset_id': '4u4b-jge6',
        'base_name': 'MTA_Bus_Speeds_Beginning_2025',
        'query': "SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        'limit': 1000
    },
    'bus_speed_2023_2024': {
        'dataset_id': '6ksi-7cxr',
        'base_name': 'MTA_Bus_Speeds_2023_2024',
        'query': """SELECT * WHERE Borough = 'Manhattan' 
                   AND month BETWEEN '2023-01-01T00:00:00' AND '2024-12-31T23:59:59' 
                   LIMIT {limit} OFFSET {offset}""",
        'limit': 1000
    },
    'hourly_ridership_2025': {
        'dataset_id': 'gxb3-akrn',
        'base_name': 'MTA_Bus_Hourly_Ridership_Beginning_2025',
        'query': """SELECT transit_timestamp, bus_route, sum(ridership) as total_ridership
                   WHERE caseless_starts_with(bus_route, 'M')
                   GROUP BY transit_timestamp, bus_route
                   LIMIT {limit} OFFSET {offset}""",
        'limit': 100000
    },
    'hourly_ridership_2023_2024': {
        'dataset_id': 'kv7t-n8in',
        'base_name': 'MTA_Bus_Hourly_Ridership_2023_2024',
        'query': """SELECT transit_timestamp, bus_route, sum(ridership) as total_ridership
                   WHERE caseless_starts_with(bus_route, 'M')
                   AND transit_timestamp BETWEEN '2023-01-01T00:00:00' AND '2024-12-31T23:59:59'
                   GROUP BY transit_timestamp, bus_route
                   LIMIT {limit} OFFSET {offset}""",
        'limit': 100000
    },
    'hourly_crossings_2023_2025': {
        'dataset_id': 'ebfx-2m7v',
        'base_name': 'MTA_Bus_Hourly_Crossings_2023_2025',
        'query': """SELECT date, hour, facility, direction, sum(traffic_count) as total_count
                   WHERE date BETWEEN '2023-01-01T00:00:00' AND '2025-12-31T23:59:59'
                   GROUP BY date, hour, facility, direction
                   LIMIT {limit} OFFSET {offset}""",
        'limit': 100000
    },
    'crz_entries_2023_2025': {
        'dataset_id': 't6yz-b64h',
        'base_name': 'MTA_CRZ_Hourly_Entries_2023_2025',
        'query': "SELECT * LIMIT {limit} OFFSET {offset}",
        'limit': 100000
    },
    'cbd_vehicle_speeds_2023_2025': {
        'dataset_id': '6p29-6xqn',
        'base_name': 'MTA_CBD_Vehicle_Speeds_2023_2025',
        'query': """SELECT * WHERE Month BETWEEN '2023-01-01T00:00:00' AND '2025-12-31T23:59:59'
                   LIMIT {limit} OFFSET {offset}""",
        'limit': 100
    },
    'cbd_bus_routes_2025': {
        'dataset_id': 'cgzt-smqf',
        'base_name': 'MTA_CBD_Bus_Routes_2025',
        'query': "SELECT * LIMIT {limit} OFFSET {offset}",
        'limit': 1000
    }
}

# Local file configuration
LOCAL_FILES = {
    'stop_data': 'manhattan_stops_flat.csv',
    'cbd_geojson_area_2024': 'MTA_Central_Business_District_Geofence__Beginning_June_2024_20251105.geojson'
}

# Mapping for return dictionary keys
DATA_KEY_MAPPING = {
    'bus_segment_speed_2025': 'bus_speed_seg_2025',
    'bus_segment_speed_2023_2024': 'bus_speed_seg_2023_2024',
    'bus_speed_2025': 'bus_speed_2025',
    'bus_speed_2023_2024': 'bus_speed_2023_2024',
    'hourly_ridership_2025': 'hourly_ridership_2025',
    'hourly_ridership_2023_2024': 'hourly_ridership_2023_2024',
    'hourly_crossings_2023_2025': 'hourly_crossings_2023_2025',
    'crz_entries_2023_2025': 'crz_entries_2023_2025',
    'cbd_vehicle_speeds_2023_2025': 'cbd_vehicle_speeds_2023_2025',
    'cbd_bus_routes_2025': 'cbd_bus_routes_2025'
}


def get_all_MTA_data(client: Socrata, DATASET_ID: str, query_template: str, 
                     limit: int, filename: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all data from the Socrata API using pagination.
    
    Parameters:
    - client: Socrata client instance
    - DATASET_ID: The dataset ID on the Socrata platform.
    - query_template: The SQL-like query string with placeholders for LIMIT and OFFSET.
    - limit: The number of rows to fetch per page.
    - filename (optional): If provided, the final DataFrame will be saved to a CSV file.

    Returns:
    - A Pandas DataFrame containing all the fetched data.
    """

    all_data = []
    offset = 0

    while True:
        query = query_template.format(limit=limit, offset=offset)
        result = client.get(DATASET_ID, query=query)
        data = pd.DataFrame.from_records(result)

        if data.empty:
            break
        all_data.append(data)
        offset += limit
        print(f"Fetched {len(data)} records, total offset now {offset}")

        if limit >= 100000:
            time.sleep(1)

    final_df = pd.concat(all_data, ignore_index=True)

    if filename:
        final_df.to_csv(filename, index=False)
    
    return final_df


def fetch_or_load_data(data_dir: str, config: Dict, client: Socrata) -> pd.DataFrame:
    """
    Check if data file exists, otherwise fetch from API.
    
    Parameters:
    - data_dir: Directory where data files are stored
    - config: Dictionary containing dataset_id, base_name, query, and limit
    - client: Socrata client instance
    
    Returns:
    - DataFrame with the loaded or fetched data
    """
    base_name = config['base_name']
    existing_file = next((f for f in os.listdir(data_dir) if f.startswith(base_name)), None)
    
    if existing_file:
        filename = os.path.join(data_dir, existing_file)
        print(f"Found existing file {filename}, skipping API pull.")
        df = pd.read_csv(filename)
    else:
        filename = f"{data_dir}/{base_name}_{datetime.now().strftime('%Y%m%d')}.csv"
        df = get_all_MTA_data(
            client, 
            config['dataset_id'], 
            config['query'], 
            config['limit'], 
            filename
        )
    
    return df


def fetch_api_datasets(client: Socrata, data_dir: str = "data/raw", 
                       datasets: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple datasets from API using configuration.
    
    Parameters:
    - client: Socrata client instance
    - data_dir: Directory for data storage
    - datasets: List of dataset keys to fetch. If None, fetch all.
    
    Returns:
    - Dictionary mapping dataset keys to DataFrames
    """
    if datasets is None:
        datasets = list(API_DATASETS.keys())
    
    result = {}
    for dataset_key in datasets:
        if dataset_key not in API_DATASETS:
            print(f"Warning: Unknown dataset '{dataset_key}', skipping...")
            continue
            
        config = API_DATASETS[dataset_key]
        print(f"Fetching {dataset_key}...")
        df = fetch_or_load_data(data_dir, config, client)
        
        # Use mapped key if available, otherwise use original key
        output_key = DATA_KEY_MAPPING.get(dataset_key, dataset_key)
        result[output_key] = df
    
    return result


def load_local_files(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load local CSV and GeoJSON files.
    
    Parameters:
    - data_dir: Directory where files are stored
    
    Returns:
    - Dictionary with loaded data
    """
    result = {}
    
    for key, filename in LOCAL_FILES.items():
        filepath = os.path.join(data_dir, filename)
        print(f"Loading {key} from {filepath}...")
        
        if filename.endswith('.geojson'):
            result[key] = gpd.read_file(filepath)
        else:
            result[key] = pd.read_csv(filepath)
    
    return result


def fetch_all_data(app_token: str, data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Main function to fetch or load all datasets using configuration-driven approach.
    
    Parameters:
    - app_token: Socrata API token
    - data_dir: Directory for data storage
    
    Returns:
    - dict: Dictionary with all loaded dataframes
    """
    # Initialize Socrata client
    client = Socrata("data.ny.gov", app_token=app_token, timeout=240)
    
    # Fetch all API datasets
    print("=" * 50)
    print("FETCHING API DATASETS")
    print("=" * 50)
    api_data = fetch_api_datasets(client, data_dir)
    
    # Load local files
    print("\n" + "=" * 50)
    print("LOADING LOCAL FILES")
    print("=" * 50)
    local_data = load_local_files(data_dir)
    
    # Combine and return all data
    return {**api_data, **local_data}
