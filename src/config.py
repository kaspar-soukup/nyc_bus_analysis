"""
Configuration file for NYC Bus Analysis project.

This file contains all configuration parameters, paths, and constants.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Socrata API configuration
SOCRATA_DOMAIN = "data.ny.gov"
SOCRATA_TIMEOUT = 240

# Dataset IDs
DATASET_IDS = {
    'bus_segment_speed_2025': 'kufs-yh3x',
    'bus_segment_speed_2023_2024': '58t6-89vi',
    'bus_speed_2025': '4u4b-jge6',
    'bus_speed_2023_2024': '6ksi-7cxr',
    'hourly_ridership_2025': 'gxb3-akrn',
    'hourly_ridership_2023_2024': 'kv7t-n8in',
    'hourly_crossings': 'ebfx-2m7v',
    'crz_entries': 't6yz-b64h',
    'cbd_vehicle_speeds': '6p29-6xqn',
    'cbd_bus_routes': 'cgzt-smqf'
}

# API query limits
QUERY_LIMITS = {
    'bus_segment_speed': 100000,
    'bus_speed': 1000,
    'hourly_ridership': 100000,
    'hourly_crossings': 100000,
    'crz_entries': 100000,
    'cbd_vehicle_speeds': 100,
    'cbd_bus_routes': 1000
}

# Data filters
BOROUGH_FILTER = 'Manhattan'
DATE_RANGE_2023_2024 = {
    'start': '2023-01-01T00:00:00',
    'end': '2024-12-31T23:59:59'
}
DATE_RANGE_2023_2025 = {
    'start': '2023-01-01T00:00:00',
    'end': '2025-12-31T23:59:59'
}

# Manhattan-related facilities for crossing analysis
MANHATTAN_FACILITIES = [
    'Henry Hudson Bridge',
    'Hugh L. Carey Tunnel',
    'Queens Midtown Tunnel',
    'Robert F. Kennedy Bridge Bronx',
    'Robert F. Kennedy Bridge Manhattan'
]

# Congestion pricing hours
CONGESTION_HOURS = {
    'weekday': {'start': 5, 'end': 20},
    'weekend': {'start': 9, 'end': 20}
}

# Visualization settings
PLOT_FIGSIZE = {
    'default': (12, 8),
    'wide': (14, 5),
    'comparison': (8, 5),
    'vertical': (5, 8),
    'tall': (10, 15)
}

PLOT_COLORS = {
    'year_comparison': ['darkgrey', 'grey', 'g'],
    'linestyles': ['-.', '--', '-']
}

# Map settings
MAP_CENTER = [40.7831, -73.9712]
MAP_ZOOM = 12
MAP_TILES = "CartoDB positron"

# Model settings
MODEL_TEST_SIZE = 0.2
MODEL_RANDOM_STATE = 42

# Feature engineering settings
WEEKEND_DAYS = ['Saturday', 'Sunday']
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
ALL_DAYS = WEEKDAYS + WEEKEND_DAYS

# Direction mapping
DIRECTION_MAP = {
    'North': 0,
    'East': 0,
    'South': 1,
    'West': 1,
    'N': 0,
    'E': 0,
    'S': 1,
    'W': 1
}

# Hours of day
HOURS_OF_DAY = list(range(24))

# Years for analysis
ANALYSIS_YEARS = [2023, 2024, 2025]

# Column name mappings for consistency
COLUMN_MAPPINGS = {
    'month': 'date',
    'crz_entries': 'sum_crz_entries',
    'CRZ Entries': 'sum_crz_entries',
    'toll_date': 'Toll Date',
    'Hour of Day': 'Hour',
    'hour_of_day': 'Hour',
    'total_count': 'sum_traffic_count',
    'date': 'Date',
    'hour': 'Hour'
}

# Columns to drop during cleaning
DROP_COLUMNS = {
    'segment_speed': ['timepoint_stop_georeference', 'next_timepoint_stop_georeference', 'borough'],
    'ridership': ['sum_ridership']
}

# File naming patterns
FILE_PATTERNS = {
    'segment_speed_2025': 'MTA_Bus_Route_Segment_Speeds_Beginning_2025',
    'segment_speed_2023_2024': 'MTA_Bus_Route_Segment_Speeds_2023_2024',
    'bus_speed_2025': 'MTA_Bus_Speeds_Beginning_2025',
    'bus_speed_2023_2024': 'MTA_Bus_Speeds_2023_2024',
    'ridership_2025': 'MTA_Bus_Hourly_Ridership_Beginning_2025',
    'ridership_2023_2024': 'MTA_Bus_Hourly_Ridership_2023_2024',
    'crossings': 'MTA_Bus_Hourly_Crossings_2023_2025',
    'crz_entries': 'MTA_CRZ_Hourly_Entries_2023_2025',
    'cbd_speeds': 'MTA_CBD_Vehicle_Speeds_2023_2025',
    'cbd_routes': 'MTA_CBD_Bus_Routes_2025',
    'cbd_geojson': 'MTA_Central_Business_District_Geofence__Beginning_June_2024_20251105.geojson',
    'stop_data': 'manhattan_stops_flat.csv'
}

# Regression features
REGRESSION_BASE_FEATURES = [
    'year', 'month', 'hour_of_day', 'total_ridership', 
    'inflow', 'outflow', 'change', 'sum_crz_entries', 
    'Average Speed', 'in_cbd'
]

REGRESSION_CATEGORICAL_FEATURES = ['day_of_week', 'route_type', 'direction_id']

# Target variable for regression
REGRESSION_TARGET = 'average_road_speed'
