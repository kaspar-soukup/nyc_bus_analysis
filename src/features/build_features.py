"""
Feature engineering and data processing module.

This module contains all data cleaning, transformation, and feature engineering logic.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Optional


def clean_segment_speed_data(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge bus segment speed data from 2023-2024 and 2025.
    
    Parameters:
    - df_2025: DataFrame with 2025 segment speed data
    - df_2023_2024: DataFrame with 2023-2024 segment speed data
    
    Returns:
    - Cleaned and merged DataFrame
    """
    # Making columns lowercase and replacing spaces with underscores
    df_2025.columns = df_2025.columns.str.lower().str.replace(" ", "_")
    df_2023_2024.columns = df_2023_2024.columns.str.lower().str.replace(" ", "_")
    
    # Merging datasets
    segment_speed_df = pd.concat([df_2025, df_2023_2024], axis=0, ignore_index=False)
    
    # Dropping unnecessary columns
    segment_speed_df = segment_speed_df.drop(
        ["timepoint_stop_georeference", "next_timepoint_stop_georeference", "borough"], 
        axis=1, 
        errors='ignore'
    )
    
    # Deriving direction_id (0 for N/E, 1 for S/W)
    segment_speed_df["direction_id"] = segment_speed_df["direction"].apply(
        lambda x: 0 if x in ["N", "E"] else 1
    )
    segment_speed_df = segment_speed_df.drop("direction", axis=1)
    
    # Creating weight columns for aggregation
    segment_speed_df["weight_distance"] = segment_speed_df["road_distance"] * segment_speed_df["bus_trip_count"]
    segment_speed_df["weighted_avg_speed"] = segment_speed_df["average_road_speed"] * segment_speed_df["weight_distance"]
    segment_speed_df["weight_travel_time"] = segment_speed_df["average_travel_time"] / 60 * segment_speed_df["bus_trip_count"]
    
    # Adding weekend column
    segment_speed_df["weekend"] = segment_speed_df["day_of_week"].apply(
        lambda x: 1 if x in ["Saturday", "Sunday"] else 0
    )
    
    return segment_speed_df


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
    # Rename month column to date
    df_2025 = df_2025.rename(columns={"month": "date"})
    df_2023_2024 = df_2023_2024.rename(columns={"month": "date"})
    
    # Add month and year columns
    df_2025["month"] = pd.to_datetime(df_2025["date"]).dt.month
    df_2025["year"] = pd.to_datetime(df_2025["date"]).dt.year
    
    df_2023_2024["month"] = pd.to_datetime(df_2023_2024["date"]).dt.month
    df_2023_2024["year"] = pd.to_datetime(df_2023_2024["date"]).dt.year
    
    # Merge datasets
    bus_speed_df = pd.concat([df_2025, df_2023_2024], axis=0, ignore_index=False)
    
    return bus_speed_df


def clean_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ridership data by parsing timestamps and converting totals to numeric.
    
    Parameters:
    - df: DataFrame with ridership data
    
    Returns:
    - Cleaned DataFrame
    """
    df = df.copy()

    source_col = "sum_ridership" if "sum_ridership" in df.columns else "total_ridership"
    df["total_ridership"] = pd.to_numeric(
        df[source_col].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    df = df.drop("sum_ridership", axis=1, errors="ignore")

    ts = pd.to_datetime(df["transit_timestamp"])
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["year"] = ts.dt.year
    df["hour"] = ts.dt.hour

    return df


def prepare_ridership_data(df_2025: pd.DataFrame, df_2023_2024: pd.DataFrame) -> tuple:
    """
    Prepare ridership data for visualization by parsing timestamps and adding weekend flags.
    
    Parameters:
    - df_2025: DataFrame with 2025 ridership data
    - df_2023_2024: DataFrame with 2023-2024 ridership data
    
    Returns:
    - tuple: (merged_df, weekday_df, weekend_df)
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
    ridership_data["transit_timestamp"] = pd.to_datetime(ridership_data["transit_timestamp"], format="%m/%d/%Y %I:%M:%S %p")
    ridership_data["weekend"] = ridership_data["transit_timestamp"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    ridership_data["day_of_week"] = ridership_data["transit_timestamp"].dt.day_name()
    
    ridership_data_weekday = ridership_data[ridership_data["weekend"] == 0]
    ridership_data_weekend = ridership_data[ridership_data["weekend"] == 1]
    
    return ridership_data, ridership_data_weekday, ridership_data_weekend


def clean_vehicle_crossing_data(crossings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process vehicle crossing data for Manhattan.
    
    Parameters:
    - crossings_df: DataFrame with crossing data
    
    Returns:
    - Cleaned DataFrame with Manhattan-specific crossings
    """
    vehicles_entering = crossings_df.copy()

    # Finding relevant facilities related to Manhattan
    relevant_facilities = vehicles_entering.groupby(["facility", "direction"])["total_count"].sum().reset_index()
    mask = (relevant_facilities["facility"].str.contains("manhattan", case=False) | 
            relevant_facilities["direction"].str.contains("manhattan", case=False))
    relevant_facilities = relevant_facilities.loc[mask, "facility"].unique()

    vehicles_entering_manhattan = vehicles_entering[vehicles_entering["facility"].isin(relevant_facilities)]
    vehicles_entering_manhattan.loc[:, "inflow_cat"] = vehicles_entering_manhattan["direction"].apply(
        lambda x: 1 if "manhattan" in x.lower() else 0
    )
    vehicles_entering_manhattan.loc[:, "inflow"] = (vehicles_entering_manhattan["total_count"] * 
                                                     vehicles_entering_manhattan["inflow_cat"])
    vehicles_entering_manhattan.loc[:, "outflow"] = -(vehicles_entering_manhattan["total_count"] * 
                                                       (1 - vehicles_entering_manhattan["inflow_cat"]))
    vehicles_entering_manhattan.loc[:, "change"] = (vehicles_entering_manhattan["inflow"] + 
                                                     vehicles_entering_manhattan["outflow"])

    vehicles_entering_manhattan = vehicles_entering_manhattan.rename(
        columns={"total_count": "sum_traffic_count", "date": "Date", "hour": "Hour"}
    )
    vehicles_entering_manhattan["sum_traffic_count"] = pd.to_numeric(
        vehicles_entering_manhattan["sum_traffic_count"].astype(str).str.replace(",", ""), 
        errors="coerce"
    )
    vehicles_entering_manhattan["Date"] = pd.to_datetime(vehicles_entering_manhattan["Date"])
    vehicles_entering_manhattan["Weekend"] = vehicles_entering_manhattan["Date"].dt.weekday.apply(
        lambda x: 1 if x >= 5 else 0
    )
    vehicles_entering_manhattan["Year"] = vehicles_entering_manhattan["Date"].dt.year
    vehicles_entering_manhattan["Month"] = vehicles_entering_manhattan["Date"].dt.month
    vehicles_entering_manhattan["Day"] = vehicles_entering_manhattan["Date"].dt.day
    
    return vehicles_entering_manhattan


def clean_crz_entries_data(crz_df: pd.DataFrame) -> tuple:
    """
    Clean and process Congestion Relief Zone entries data.
    
    Parameters:
    - crz_df: DataFrame with CRZ entries
    
    Returns:
    - tuple: (full_df, weekday_df, weekend_df)
    """
    vehicles_entering_cbd = crz_df.copy()
    
    # Rename columns to handle different formats
    vehicles_entering_cbd = vehicles_entering_cbd.rename(columns={
        "crz_entries": "sum_crz_entries",
        "CRZ Entries": "sum_crz_entries",
        "toll_date": "Toll Date",
        "Hour of Day": "Hour",
        "hour_of_day": "Hour"
    })
    
    vehicles_entering_cbd["sum_crz_entries"] = pd.to_numeric(
        vehicles_entering_cbd["sum_crz_entries"].astype(str).str.replace(",", ""), 
        errors="coerce"
    )
    vehicles_entering_cbd["Date"] = pd.to_datetime(vehicles_entering_cbd["Toll Date"])
    vehicles_entering_cbd["Weekend"] = vehicles_entering_cbd["Date"].dt.weekday.apply(
        lambda x: 1 if x >= 5 else 0
    )
    vehicles_entering_cbd["Year"] = vehicles_entering_cbd["Date"].dt.year
    vehicles_entering_cbd["Month"] = vehicles_entering_cbd["Date"].dt.month
    vehicles_entering_cbd["Day"] = vehicles_entering_cbd["Date"].dt.day
    vehicles_entering_cbd["Hour"] = pd.to_numeric(vehicles_entering_cbd["Hour"], errors="coerce")

    vehicles_entering_per_hour = vehicles_entering_cbd.groupby(
        ["Month", "Day", "Weekend", "Hour"]
    )["sum_crz_entries"].sum().reset_index()

    vehicles_entering_cbd_weekday = vehicles_entering_per_hour[vehicles_entering_per_hour["Weekend"] == 0]
    vehicles_entering_cbd_weekend = vehicles_entering_per_hour[vehicles_entering_per_hour["Weekend"] == 1]
    
    return vehicles_entering_cbd, vehicles_entering_cbd_weekday, vehicles_entering_cbd_weekend


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
                        cbd_geojson: gpd.GeoDataFrame) -> tuple:
    """
    Filter segment speed data to only include stops within the CBD polygon.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - cbd_geojson: GeoDataFrame with CBD polygon
    
    Returns:
    - tuple: (cbd_gdf, weekday_gdf, weekend_gdf)
    """
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
    
    segment_speed_cbd_weekend = segment_speed_cbd_geom[
        segment_speed_cbd_geom["day_of_week"].isin(["Saturday", "Sunday"])
    ]
    segment_speed_cbd_weekday = segment_speed_cbd_geom[
        ~segment_speed_cbd_geom["day_of_week"].isin(["Saturday", "Sunday"])
    ]
    
    return segment_speed_cbd_geom, segment_speed_cbd_weekday, segment_speed_cbd_weekend
