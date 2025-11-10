"""
Feature engineering and data processing module.

This module contains all data cleaning, transformation, and feature engineering logic.
Uses a configuration-driven approach to reduce code repetition.
"""

import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# GENERIC UTILITY FUNCTIONS
# ============================================================================

def standardize_columns(df: pd.DataFrame, lowercase: bool = True, 
                       replace_spaces: bool = True) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame.
    
    Parameters:
    - df: DataFrame to process
    - lowercase: Convert to lowercase
    - replace_spaces: Replace spaces with underscores
    
    Returns:
    - DataFrame with standardized columns
    """
    df = df.copy()
    if lowercase:
        df.columns = df.columns.str.lower()
    if replace_spaces:
        df.columns = df.columns.str.replace(" ", "_")
    return df


def merge_yearly_datasets(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame, 
                         standardize: bool = True) -> pd.DataFrame:
    """
    Merge datasets from different time periods.
    
    Parameters:
    - df_2025: DataFrame with 2025 data
    - df_2023_2024: DataFrame with 2023-2024 data
    - standardize: Whether to standardize column names
    
    Returns:
    - Merged DataFrame
    """
    if standardize:
        df_2025 = standardize_columns(df_2025)
        df_2023_2024 = standardize_columns(df_2023_2024)
    
    return pd.concat([df_2025, df_2023_2024], axis=0, ignore_index=True)


def add_temporal_features(df: pd.DataFrame, timestamp_col: str, 
                         features: List[str] = None, standardize_cols: bool = False, 
                         datetime_format: str = None) -> pd.DataFrame:
    """
    Add temporal features from a timestamp column.
    
    Parameters:
    - df: DataFrame to process
    - timestamp_col: Name of the timestamp column
    - features: List of features to add ['year', 'month', 'day', 'hour', 
                'day_of_week', 'weekend', 'congestion_pricing']
    - standardize_cols: Whether to standardize all column names to lowercase with underscores
    - datetime_format: Format string for parsing datetime (optional, will infer if not provided)
    
    Returns:
    - DataFrame with added temporal features and optionally standardized column names
    """
    df = df.copy()
    
    # Standardize column names first if requested
    if standardize_cols:
        df = standardize_columns(df, lowercase=True, replace_spaces=True)
    
    if features is None:
        features = ['year', 'month', 'day', 'hour']
    
    # Normalize timestamp column name to handle both cases
    timestamp_col_normalized = timestamp_col.lower().replace(" ", "_")
    
    # Find the actual column (case-insensitive)
    actual_col = None
    for col in df.columns:
        if col.lower().replace(" ", "_") == timestamp_col_normalized:
            actual_col = col
            break
    
    if actual_col is None:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found. Available columns: {list(df.columns)}")
    
    # Convert to datetime, handling both date and datetime formats
    try:
        if datetime_format:
            ts = pd.to_datetime(df[actual_col], format=datetime_format, errors='coerce')
        else:
            # Use fast parsing with errors='coerce' to suppress warnings
            ts = pd.to_datetime(df[actual_col], errors='coerce')
    except Exception as e:
        logger.debug(f"Error parsing datetime column '{actual_col}': {e}")
        ts = pd.to_datetime(df[actual_col], errors='coerce')
    
    # Only extract temporal features if the columns don't already exist with valid values
    if 'year' in features:
        # Only extract if column doesn't exist or is all NaN
        if 'year' not in df.columns or df['year'].isna().all():
            df['year'] = ts.dt.year
    if 'month' in features:
        # Only extract if column doesn't exist or is all NaN
        if 'month' not in df.columns or df['month'].isna().all():
            df['month'] = ts.dt.month
    if 'day' in features:
        # Only extract if column doesn't exist or is all NaN
        if 'day' not in df.columns or df['day'].isna().all():
            df['day'] = ts.dt.day
    if 'hour' in features:
        # Only extract if column doesn't exist or is all NaN
        # Check for hour_of_day column first (from raw data)
        if 'hour_of_day' in df.columns:
            # Rename hour_of_day to hour if hour doesn't exist
            if 'hour' not in df.columns:
                df['hour'] = df['hour_of_day']
        elif 'hour' not in df.columns or df['hour'].isna().all():
            df['hour'] = ts.dt.hour
    if 'day_of_week' in features:
        # Only extract if column doesn't exist or is all NaN
        if 'day_of_week' not in df.columns or df['day_of_week'].isna().all():
            df['day_of_week'] = ts.dt.day_name()
    if 'weekend' in features:
        # Create weekend column from numeric weekday (0=Mon, 6=Sun)
        # Works whether day_of_week column has text or numbers
        if 'weekend' not in df.columns:
            if 'day_of_week' in df.columns and df['day_of_week'].dtype == 'object':
                # If day_of_week contains text names
                df['weekend'] = df['day_of_week'].apply(
                    lambda x: 1 if x in ['Saturday', 'Sunday'] else 0
                )
            else:
                # If working with numeric weekday values
                df['weekend'] = ts.dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    if 'congestion_pricing' in features:
        # Add congestion pricing timeframe column
        # Requires 'year', 'hour', and 'weekend' columns
        if 'year' not in df.columns:
            df['year'] = ts.dt.year
        if 'hour' not in df.columns:
            if 'hour_of_day' in df.columns:
                df['hour'] = df['hour_of_day']
            else:
                df['hour'] = ts.dt.hour
        if 'weekend' not in df.columns:
            if 'day_of_week' in df.columns and df['day_of_week'].dtype == 'object':
                df['weekend'] = df['day_of_week'].apply(
                    lambda x: 1 if x in ['Saturday', 'Sunday'] else 0
                )
            else:
                df['weekend'] = ts.dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        
        df['congestion_pricing'] = df.apply(
            lambda row: congestion_pricing_timeframe(row["weekend"], row["hour"], row["year"]), axis=1
        )
    
    return df


def clean_numeric_column(df: pd.DataFrame, col_name: str, 
                        remove_commas: bool = True) -> pd.DataFrame:
    """
    Clean and convert a column to numeric type.
    
    Parameters:
    - df: DataFrame to process
    - col_name: Name of column to clean
    - remove_commas: Whether to remove commas before conversion
    
    Returns:
    - DataFrame with cleaned column
    """
    df = df.copy()
    if remove_commas:
        df[col_name] = df[col_name].astype(str).str.replace(",", "")
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


def split_by_weekend(df: pd.DataFrame, weekend_col: str = 'weekend') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into weekday and weekend subsets.
    
    Parameters:
    - df: DataFrame to split
    - weekend_col: Name of the weekend indicator column
    
    Returns:
    - Tuple of (weekday_df, weekend_df)
    """
    weekday_df = df[df[weekend_col] == 0].copy()
    weekend_df = df[df[weekend_col] == 1].copy()
    return weekday_df, weekend_df


# ============================================================================
# DOMAIN-SPECIFIC PROCESSING FUNCTIONS
# ============================================================================, Dict, List, Tuple, Union


def clean_segment_speed_data(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge bus segment speed data from 2023-2024 and 2025.
    
    Parameters:
    - df_2025: DataFrame with 2025 segment speed data
    - df_2023_2024: DataFrame with 2023-2024 segment speed data
    
    Returns:
    - Cleaned and merged DataFrame
    """
    # Merge datasets with standardized columns
    segment_speed_df = merge_yearly_datasets(df_2025, df_2023_2024)
    
    # Drop unnecessary columns
    segment_speed_df = segment_speed_df.drop(
        ["timepoint_stop_georeference", "next_timepoint_stop_georeference", "borough"], 
        axis=1, 
        errors='ignore'
    )
    
    # Derive direction_id (0 for N/E, 1 for S/W)
    segment_speed_df["direction_id"] = segment_speed_df["direction"].apply(
        lambda x: 0 if x in ["N", "E"] else 1
    )
    segment_speed_df = segment_speed_df.drop("direction", axis=1)
    
    # Create weight columns for aggregation
    segment_speed_df["weight_distance"] = segment_speed_df["road_distance"] * segment_speed_df["bus_trip_count"]
    segment_speed_df["weight_travel_time"] = segment_speed_df["average_travel_time"] / 60 * segment_speed_df["bus_trip_count"]
    
    # Add temporal features including congestion pricing
    segment_speed_df = add_temporal_features(
        segment_speed_df, 
        "timestamp", 
        ["year", "month", "hour", "weekend", "day_of_week", "congestion_pricing"]
    )
    
    # Rename congestion_pricing to match expected column name
    if "congestion_pricing" in segment_speed_df.columns:
        segment_speed_df["congestion_pricing_timeframe"] = segment_speed_df["congestion_pricing"]

    return segment_speed_df


def add_cbd_classification(segment_speed_df: pd.DataFrame, cbd_geofence_path: str) -> pd.DataFrame:
    """
    Add CBD classification to segment speed data using geofence polygon.
    Uses segment midpoint for classification.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - cbd_geofence_path: Path to the CBD GeoJSON file
    
    Returns:
    - DataFrame with is_cbd_segment column added
    """
    import geopandas as gpd
    from shapely.geometry import Point
    
    print("Loading CBD geofence...")
    cbd_geofence = gpd.read_file(cbd_geofence_path)
    cbd_polygon = cbd_geofence.geometry.union_all()
    
    print("Calculating segment midpoints...")
    segment_speed_df = segment_speed_df.copy()
    
    # Calculate midpoint coordinates
    segment_speed_df['mid_lat'] = (segment_speed_df['timepoint_stop_latitude'] + 
                                    segment_speed_df['next_timepoint_stop_latitude']) / 2
    segment_speed_df['mid_lon'] = (segment_speed_df['timepoint_stop_longitude'] + 
                                    segment_speed_df['next_timepoint_stop_longitude']) / 2
    
    print("Classifying CBD segments (this may take 30-60 seconds)...")
    # Vectorized approach for performance
    valid_coords = segment_speed_df['mid_lon'].notna() & segment_speed_df['mid_lat'].notna()
    segment_speed_df['is_cbd_segment'] = False
    
    # Only process rows with valid coordinates
    if valid_coords.any():
        valid_df = segment_speed_df[valid_coords]
        points = [Point(lon, lat) for lon, lat in zip(valid_df['mid_lon'], valid_df['mid_lat'])]
        segment_speed_df.loc[valid_coords, 'is_cbd_segment'] = [cbd_polygon.contains(p) for p in points]
    
    # Drop temporary midpoint columns
    segment_speed_df = segment_speed_df.drop(['mid_lat', 'mid_lon'], axis=1)
    
    print(f"  CBD segments: {segment_speed_df['is_cbd_segment'].sum():,}")
    print(f"  Non-CBD segments: {(~segment_speed_df['is_cbd_segment']).sum():,}")
    
    return segment_speed_df


def create_speed_monthly_aggregated(segment_speed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create monthly aggregated speed data by route.
    Uses weight_distance and weight_travel_time to calculate weighted average speed.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data (must have is_cbd_segment column)
    
    Returns:
    - DataFrame with monthly aggregates by route
    """
    print("Creating monthly aggregated speed data...")
    
    # Filter to only include valid data
    df = segment_speed_df[
        segment_speed_df['weight_distance'].notna() &
        segment_speed_df['weight_travel_time'].notna() &
        segment_speed_df['route_id'].notna()
    ].copy()
    
    # Create aggregation
    speed_monthly = df.groupby([
        'route_id', 
        'year', 
        'month', 
        'weekend', 
        'is_cbd_segment'
    ], as_index=False).agg({
        'weight_distance': 'sum',
        'weight_travel_time': 'sum',
        'bus_trip_count': 'sum',
        'road_distance': 'sum'
    })
    
    # Calculate weighted average speed
    speed_monthly['avg_speed_mph'] = speed_monthly['weight_distance'] / speed_monthly['weight_travel_time']
    
    # Add year_month for plotting
    speed_monthly['year_month'] = speed_monthly['year'].astype(str) + '-' + speed_monthly['month'].astype(str).str.zfill(2)
    
    print(f"  Created {len(speed_monthly):,} monthly route aggregates")
    
    return speed_monthly


def create_speed_overall_aggregated(speed_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create overall aggregated speed data (not by route).
    Re-calculates weighted average speed across all routes.
    
    Parameters:
    - speed_monthly_df: DataFrame with monthly aggregates
    
    Returns:
    - DataFrame with overall aggregates (no route dimension)
    """
    print("Creating overall aggregated speed data...")
    
    # Aggregate across all routes using weighted sums
    speed_overall = speed_monthly_df.groupby([
        'year_month',
        'year',
        'month',
        'weekend',
        'is_cbd_segment'
    ], as_index=False).agg({
        'weight_distance': 'sum',
        'weight_travel_time': 'sum',
        'bus_trip_count': 'sum',
        'road_distance': 'sum'
    })
    
    # Recalculate weighted average speed
    speed_overall['avg_speed_mph'] = speed_overall['weight_distance'] / speed_overall['weight_travel_time']
    
    # Sort by year_month
    speed_overall['date_sort'] = pd.to_datetime(speed_overall['year_month'] + '-01')
    speed_overall = speed_overall.sort_values('date_sort').drop('date_sort', axis=1)
    
    print(f"  Created {len(speed_overall):,} overall monthly aggregates")
    
    return speed_overall


def congestion_pricing_timeframe(weekend: int, hour: int, year: int) -> bool:
    """
    Determine if a given time falls within the congestion pricing timeframe.
    
    Parameters:
    - weekend: 1 if weekend, 0 if weekday
    - hour: Hour of the day (0-23)
    - year: Year of the date
    
    Returns:
    - True if within congestion pricing timeframe, False otherwise
    """
    # Handle NaN values
    if pd.isna(weekend) or pd.isna(hour) or pd.isna(year):
        return False
    
    if year == 2025:
        # Weekday: 5am (5) to 9pm (21)
        if weekend == 0 and hour >= 5 and hour < 22:
            return True
        # Weekend: 9am (9) to 9pm (21)
        elif weekend == 1 and hour >= 9 and hour < 22:
            return True
    
    return False

def clean_stop_data(stop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare stop data for joining with segment data.
    
    Parameters:
    - stop_df: DataFrame with stop data
    
    Returns:
    - Cleaned DataFrame
    """
    stop_df = stop_df.copy()
    
    # Preparing route and stop order keys
    stop_df["route_id_new"] = stop_df["route_id"].apply(lambda x: x[x.find("_") + 1:])
    stop_df["stop_order_new"] = stop_df["stop_order"] + 1
    
    return stop_df


def clean_bus_speed_data(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge bus speed data from 2023-2024 and 2025.
    
    Parameters:
    - df_2025: DataFrame with 2025 bus speed data
    - df_2023_2024: DataFrame with 2023-2024 bus speed data
    
    Returns:
    - Cleaned and merged DataFrame
    """
    # Standardize column names
    df_2025 = df_2025.rename(columns={"month": "date"})
    df_2023_2024 = df_2023_2024.rename(columns={"month": "date"})
    
    # Add temporal features to both datasets including congestion pricing
    df_2025 = add_temporal_features(df_2025, 'date', ['year', 'month', 'hour', 'weekend', 'congestion_pricing'])
    df_2023_2024 = add_temporal_features(df_2023_2024, 'date', ['year', 'month', 'hour', 'weekend', 'congestion_pricing'])
    
    # Merge datasets
    return pd.concat([df_2025, df_2023_2024], axis=0, ignore_index=True)


def clean_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ridership data by parsing timestamps and converting totals to numeric.
    
    Parameters:
    - df: DataFrame with ridership data
    
    Returns:
    - Cleaned DataFrame
    """
    df = df.copy()

    # Handle different column names
    source_col = "sum_ridership" if "sum_ridership" in df.columns else "total_ridership"
    df["total_ridership"] = pd.to_numeric(
        df[source_col].astype(str).str.replace(",", ""),
        errors="coerce",
    )
    df = df.drop("sum_ridership", axis=1, errors="ignore")

    # Add temporal features
    df = add_temporal_features(df, 'transit_timestamp', ['month', 'day', 'year', 'hour'])

    return df


def prepare_ridership_data(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare ridership data for visualization by parsing timestamps and adding weekend flags.
    
    Parameters:
    - df_2025: DataFrame with 2025 ridership data
    - df_2023_2024: DataFrame with 2023-2024 ridership data
    
    Returns:
    - Tuple of (merged_df, weekday_df, weekend_df)
    """
    # Parse timestamps
    df_2025["transit_timestamp"] = pd.to_datetime(df_2025["transit_timestamp"])
    df_2023_2024["transit_timestamp"] = pd.to_datetime(
        df_2023_2024["transit_timestamp"],
        format="%m/%d/%Y %I:%M:%S %p"
    )
    
    # Clean data
    hourly_ridership_2025 = clean_ridership(df_2025)
    hourly_ridership_2023_2024 = clean_ridership(df_2023_2024)
    
    # Merge datasets
    ridership_data = pd.concat([hourly_ridership_2025, hourly_ridership_2023_2024], axis=0, ignore_index=True)
    
    # Add additional temporal features including congestion pricing
    ridership_data = add_temporal_features(ridership_data, 'transit_timestamp', ['year', 'hour', 'weekend', 'day_of_week', 'congestion_pricing'])
    
    # Split by weekend
    weekday_df, weekend_df = split_by_weekend(ridership_data)
    
    return ridership_data, weekday_df, weekend_df


def clean_vehicle_crossing_data(crossings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process vehicle crossing data for Manhattan.
    
    Parameters:
    - crossings_df: DataFrame with crossing data
    
    Returns:
    - Cleaned DataFrame with Manhattan-specific crossings
    """
    # Handle empty DataFrame
    if crossings_df.empty or len(crossings_df) == 0:
        return pd.DataFrame()
    
    vehicles_entering = crossings_df.copy()

    # Find relevant facilities related to Manhattan
    relevant_facilities = vehicles_entering.groupby(["facility", "direction"])["total_count"].sum().reset_index()
    mask = (relevant_facilities["facility"].str.contains("manhattan", case=False) | 
            relevant_facilities["direction"].str.contains("manhattan", case=False))
    relevant_facilities = relevant_facilities.loc[mask, "facility"].unique()

    vehicles_entering_manhattan = vehicles_entering[vehicles_entering["facility"].isin(relevant_facilities)].copy()
    
    # Calculate inflow/outflow
    vehicles_entering_manhattan["inflow_cat"] = vehicles_entering_manhattan["direction"].apply(
        lambda x: 1 if "manhattan" in x.lower() else 0
    )
    vehicles_entering_manhattan["inflow"] = (vehicles_entering_manhattan["total_count"] * 
                                             vehicles_entering_manhattan["inflow_cat"])
    vehicles_entering_manhattan["outflow"] = -(vehicles_entering_manhattan["total_count"] * 
                                               (1 - vehicles_entering_manhattan["inflow_cat"]))
    vehicles_entering_manhattan["change"] = (vehicles_entering_manhattan["inflow"] + 
                                             vehicles_entering_manhattan["outflow"])

    # Rename and clean columns
    vehicles_entering_manhattan = vehicles_entering_manhattan.rename(
        columns={"total_count": "sum_traffic_count", "toll_date": "date"}
    )
    
    # Clean numeric column
    vehicles_entering_manhattan = clean_numeric_column(vehicles_entering_manhattan, "sum_traffic_count")
    
    # Add temporal features including congestion pricing
    # Note: This data only has date level, so congestion_pricing will be False/True based on weekday/weekend and year
    vehicles_entering_manhattan = add_temporal_features(
        vehicles_entering_manhattan, 'date', ['year', 'month', 'day', 'hour', 'weekend', 'congestion_pricing']
    )
    
    return vehicles_entering_manhattan


def clean_crz_entries_data(crz_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean and process Congestion Relief Zone entries data.
    
    Parameters:
    - crz_df: DataFrame with CRZ entries
    
    Returns:
    - Tuple of (full_df, weekday_df, weekend_df)
    """
    # Handle empty DataFrame
    if crz_df.empty or len(crz_df) == 0:
        empty = pd.DataFrame()
        return empty, empty, empty
    
    vehicles_entering_cbd = crz_df.copy()
    
    # Rename columns to handle different formats, standardize to lowercase with underscores
    vehicles_entering_cbd = vehicles_entering_cbd.rename(columns={
        "crz_entries": "sum_crz_entries",
        "CRZ Entries": "sum_crz_entries",
        "toll_date": "toll_date",
        "Toll Date": "toll_date",
        "Hour of Day": "hour",
        "hour_of_day": "hour",
        "Hour": "hour"
    })
    
    # Clean numeric column
    vehicles_entering_cbd = clean_numeric_column(vehicles_entering_cbd, "sum_crz_entries")
    
    # Add temporal features including congestion pricing
    vehicles_entering_cbd = add_temporal_features(
        vehicles_entering_cbd, 'toll_date', ['year', 'month', 'day', 'hour', 'weekend', 'day_of_week', 'congestion_pricing']
    )
    
    # Clean Hour column - now using lowercase
    vehicles_entering_cbd["hour"] = pd.to_numeric(vehicles_entering_cbd["hour"], errors="coerce")

    # Aggregate by time periods using lowercase column names
    vehicles_entering_per_hour = vehicles_entering_cbd.groupby(
        ["month", "day", "weekend", "hour"]
    )["sum_crz_entries"].sum().reset_index()

    # Split by weekend
    weekday_df, weekend_df = split_by_weekend(vehicles_entering_per_hour, 'weekend')
    
    return vehicles_entering_cbd, weekday_df, weekend_df


def interpolate_missing_bus_stops(speed_data: pd.DataFrame, stop_data: pd.DataFrame, 
                                   month_filter: Optional[int] = None, 
                                   save_to_csv: bool = False, 
                                   output_filename: str = "extended_list.csv") -> pd.DataFrame:
    """
    Interpolate missing bus stop segments between timepoints.

    Parameters
    ----------
    speed_data : pd.DataFrame
        DataFrame containing bus speed data with columns:
        ['year', 'month', 'day_of_week', 'hour_of_day', 'route_id', 'direction_id',
         'stop_order', 'timepoint_stop_id', 'timepoint_stop_name',
         'timepoint_stop_latitude', 'timepoint_stop_longitude',
         'next_timepoint_stop_id', 'next_timepoint_stop_name',
         'next_timepoint_stop_latitude', 'next_timepoint_stop_longitude',
         'road_distance', 'average_travel_time', 'average_road_speed']
    stop_data : pd.DataFrame
        DataFrame containing stop reference data with columns:
        ['route_id_new', 'stop_order_new', 'direction_id', 'stop_code',
         'stop_id', 'stop_name', 'latitude', 'longitude']
    month_filter : int, optional
        Filter data to specific month (1-12). If None, processes all months.
    save_to_csv : bool, default False
        Whether to save the result to a CSV file.
    output_filename : str, default "extended_list.csv"
        Name of the output CSV file if save_to_csv is True.
    """

    subset = speed_data.loc[:, [
        "year", "month", "day_of_week", "hour_of_day", "route_id", "direction_id",
        "stop_order", "timepoint_stop_id", "timepoint_stop_name",
        "timepoint_stop_latitude", "timepoint_stop_longitude",
        "next_timepoint_stop_id", "next_timepoint_stop_name",
        "next_timepoint_stop_latitude", "next_timepoint_stop_longitude",
        "road_distance", "average_travel_time", "average_road_speed"
    ]].copy()

    subset = subset.sort_values([
        "year", "month", "day_of_week", "hour_of_day", "route_id", "direction_id", "stop_order"
    ])
    subset["key"] = subset.loc[:, ["route_id", "direction_id"]].astype(str).agg("-".join, axis=1)

    if month_filter is not None:
        subset = subset[subset["month"] == month_filter]

    stops_subset = stop_data[["route_id_new", "stop_order_new", "direction_id", "stop_code"]].copy()
    stops_subset["key"] = stops_subset.loc[:, ["route_id_new", "direction_id"]].astype(str).agg("-".join, axis=1)

    merged_data = subset.merge(
        stops_subset,
        left_on=["key", "next_timepoint_stop_id"],
        right_on=["key", "stop_code"],
        how="left",
        suffixes=["", "_next_stop"]
    )

    merged_data.drop(["route_id_new", "direction_id_next_stop", "stop_code"], axis=1, inplace=True)
    merged_data.rename(columns={"stop_order_new": "next_stop_order"}, inplace=True)

    merged_data["missing_segments"] = merged_data.apply(
        lambda x: list(range(int(x["stop_order"]), int(x["next_stop_order"]) + 1))
        if pd.notna(x["next_stop_order"]) and x["next_stop_order"] - x["stop_order"] > 1
        else None,
        axis=1
    )

    original_rows = merged_data[merged_data["missing_segments"].isna()].copy()
    rows_to_expand = merged_data[merged_data["missing_segments"].notna()].copy()

    expanded_segments = rows_to_expand.explode("missing_segments")
    expanded_segments["target_stop_order"] = expanded_segments["missing_segments"].astype(int)
    expanded_segments["next_stop_order"] = expanded_segments["missing_segments"].astype(int) + 1

    stop_data_for_merge = stop_data[[
        "route_id_new", "direction_id", "stop_order", "stop_id", "stop_name", "latitude", "longitude"
    ]].copy()
    stop_data_for_merge["stop_order"] = stop_data_for_merge["stop_order"].astype(int)

    merged_expanded = expanded_segments.merge(
        stop_data_for_merge,
        left_on=["route_id", "direction_id", "target_stop_order"],
        right_on=["route_id_new", "direction_id", "stop_order"],
        how="left",
        suffixes=["", "_target"]
    )

    merged_expanded = merged_expanded.merge(
        stop_data_for_merge,
        left_on=["route_id", "direction_id", "next_stop_order"],
        right_on=["route_id_new", "direction_id", "stop_order"],
        how="left",
        suffixes=["", "_next"]
    )

    expanded_segments = expanded_segments.drop(["next_stop_order", "missing_segments"], axis=1)
    original_rows = original_rows.drop(["next_stop_order", "missing_segments"], axis=1)

    interpolated_rows = merged_expanded[[
        "year", "month", "day_of_week", "hour_of_day", "route_id",
        "direction_id", "target_stop_order",
        "stop_id", "stop_name", "latitude", "longitude",
        "stop_id_next", "stop_name_next",
        "latitude_next", "longitude_next", "average_road_speed",
        "key", "stop_order_next"
    ]].rename(columns={
        "target_stop_order": "stop_order",
        "stop_id": "timepoint_stop_id",
        "stop_name": "timepoint_stop_name",
        "latitude": "timepoint_stop_latitude",
        "longitude": "timepoint_stop_longitude",
        "stop_id_next": "next_timepoint_stop_id",
        "stop_name_next": "next_timepoint_stop_name",
        "latitude_next": "next_timepoint_stop_latitude",
        "longitude_next": "next_timepoint_stop_longitude",
        "stop_order_next": "next_stop_order"
    })

    interpolated_rows["road_distance"] = None
    interpolated_rows["average_travel_time"] = None

    original_rows["next_stop_order"] = original_rows["stop_order"] + 1
    original_rows_keep = original_rows[~original_rows["next_stop_order"].isna()].drop_duplicates()

    result_df = pd.concat([original_rows_keep, interpolated_rows], axis=0)
    result_df = result_df.dropna(subset=[
        "next_timepoint_stop_latitude",
        "timepoint_stop_latitude",
        "next_timepoint_stop_longitude",
        "timepoint_stop_longitude"
    ])

    if save_to_csv:
        result_df.to_csv(output_filename, index=False)

    return result_df


def filter_cbd_segments(segment_speed_df: pd.DataFrame, 
                        cbd_geojson: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter segment speed data to only include stops within the CBD polygon.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - cbd_geojson: GeoDataFrame with CBD polygon
    
    Returns:
    - Tuple of (cbd_gdf, weekday_gdf, weekend_gdf)
    """
    # Handle empty CBD geojson
    if cbd_geojson.empty:
        logger.warning("CBD GeoJSON is empty, returning full segment speed data")
        segment_speed_gdf = gpd.GeoDataFrame(
            segment_speed_df,
            geometry=gpd.points_from_xy(
                segment_speed_df["timepoint_stop_longitude"], 
                segment_speed_df["timepoint_stop_latitude"]
            ),
            crs="EPSG:4326",
        )
        segment_speed_cbd_weekend = segment_speed_gdf[
            segment_speed_gdf["day_of_week"].isin(["Saturday", "Sunday"])
        ]
        segment_speed_cbd_weekday = segment_speed_gdf[
            ~segment_speed_gdf["day_of_week"].isin(["Saturday", "Sunday"])
        ]
        return segment_speed_gdf, segment_speed_cbd_weekday, segment_speed_cbd_weekend
    
    cbd_polygon = cbd_geojson.geometry.union_all()
    
    segment_speed_gdf = gpd.GeoDataFrame(
        segment_speed_df,
        geometry=gpd.points_from_xy(
            segment_speed_df["timepoint_stop_longitude"], 
            segment_speed_df["timepoint_stop_latitude"]
        ),
        crs="EPSG:4326",
    )
    
    segment_speed_cbd_geom = segment_speed_gdf[segment_speed_gdf.within(cbd_polygon)]
    
    # Split by weekend (based on day_of_week column)
    segment_speed_cbd_weekend = segment_speed_cbd_geom[
        segment_speed_cbd_geom["day_of_week"].isin(["Saturday", "Sunday"])
    ]
    segment_speed_cbd_weekday = segment_speed_cbd_geom[
        ~segment_speed_cbd_geom["day_of_week"].isin(["Saturday", "Sunday"])
    ]
    
    return segment_speed_cbd_geom, segment_speed_cbd_weekday, segment_speed_cbd_weekend
