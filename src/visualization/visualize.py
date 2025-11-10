"""
Visualization module for bus analysis.

This module contains all plotting and visualization functions.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from typing import Optional


def plot_speed_by_hour_and_bus_type(segment_speed_df: pd.DataFrame, 
                                      figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot average speed by hour of day for each bus type.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    options = list(segment_speed_df["route_type"].dropna().unique())[:4]
    hours = list(range(24))

    for i, ax in enumerate(axes.flatten()):
        if i < len(options):
            condition = options[i]
            subset = segment_speed_df[segment_speed_df["route_type"] == condition]
            if len(subset) > 0:
                grouped = subset.groupby(["hour_of_day"])[["weight_travel_time", "weight_distance"]].sum()
                grouped = grouped.reindex(hours, fill_value=1)  # Avoid division by zero
                grouped["avg_speed"] = grouped["weight_distance"] / grouped["weight_travel_time"].replace(0, 1)
                grouped["avg_speed"].plot(kind="line", ax=ax, marker='o')
                ax.set_title(f"Hourly avg. speed for {condition}")
                ax.set_ylabel("Speed (mph)")
                ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    return fig


def plot_speed_by_hour_and_weekday(segment_speed_df: pd.DataFrame, 
                                     figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot average speed by hour of day for each weekday.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True, sharey=True)
    weekday_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = list(range(24))

    for i, ax in enumerate(axes.flatten()):
        if i < len(weekday_options):
            condition = weekday_options[i]
            subset = segment_speed_df[segment_speed_df["day_of_week"] == condition]
            if len(subset) > 0:
                grouped = subset.groupby(["hour_of_day"])[["weight_travel_time", "weight_distance"]].sum()
                grouped = grouped.reindex(hours, fill_value=1)
                grouped["avg_speed"] = grouped["weight_distance"] / grouped["weight_travel_time"].replace(0, 1)
                grouped["avg_speed"].plot(kind="line", ax=ax, marker='o')
                ax.set_title(f"{condition}")
                ax.set_ylabel("Speed (mph)")
                ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    plt.suptitle("Average Speed by Hour and Weekday", fontsize=14, y=1.00)
    plt.tight_layout()
    return fig


def plot_speed_comparison_by_year(segment_speed_df: pd.DataFrame, 
                                   figsize: tuple = (8, 5)) -> plt.Figure:
    """
    Plot comparison of average speeds across 2023, 2024, and 2025.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    figs, axes = plt.subplots(figsize=figsize, sharex=True, sharey=True)
    comparison_years = [2023, 2024, 2025]
    comparison_colors = ["darkgrey", "grey", "g"]
    linestyles = ['-.', '--', '-']
    hours = list(range(24))

    for idx, year in enumerate(comparison_years):
        subset = segment_speed_df[segment_speed_df["year"] == year]
        subset = subset.groupby(["hour_of_day"])[["weight_travel_time", "weight_distance"]].sum().reindex(hours)
        subset["avg_speed"] = subset["weight_distance"] / subset["weight_travel_time"]
        subset["avg_speed"].plot(kind="line", alpha=0.8, color=comparison_colors[idx], 
                                 label=year, linestyle=linestyles[idx])
    plt.legend()
    
    return figs


def plot_speed_change_vs_2023(segment_speed_df: pd.DataFrame, 
                               figsize: tuple = (10, 15)) -> plt.Figure:
    """
    Plot percentage change in average bus route speeds vs 2023 baseline.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    bus_speed = segment_speed_df.groupby(["year", "route_id"])[["weight_travel_time", "weight_distance"]].sum().unstack(level=0)
    bus_speed[2023] = bus_speed[("weight_distance", 2023)] / bus_speed[("weight_travel_time", 2023)]
    bus_speed[2024] = bus_speed[("weight_distance", 2024)] / bus_speed[("weight_travel_time", 2024)]
    bus_speed[2025] = bus_speed[("weight_distance", 2025)] / bus_speed[("weight_travel_time", 2025)]

    bus_speed["25_perc"] = (bus_speed[2025] - bus_speed[2023]) / bus_speed[2023].fillna(-np.inf)
    bus_speed["24_perc"] = (bus_speed[2024] - bus_speed[2023]) / bus_speed[2023].fillna(-np.inf)

    bus_speed["25_faster"] = (bus_speed[2025] - bus_speed[2023]) / bus_speed[2023].fillna(-np.inf)
    bus_speed = bus_speed.sort_values(by=["25_faster"])

    order = bus_speed.index.tolist()
    fig = plt.figure(figsize=figsize)
    bus_speed.loc[order, ["24_perc", "25_perc"]].plot(kind="barh", figsize=figsize, legend=True)
    plt.title("Percentage Change in Average Bus Route Speeds (2023-2025)")
    plt.xlabel("Percentage Change")
    plt.ylabel("Bus Route ID")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.legend(["0% Change", "2024 vs 2023", "2025 vs 2023"])
    plt.tight_layout(pad=0.5)
    
    return fig


def plot_speed_change_2025_vs_2024(segment_speed_df: pd.DataFrame, 
                                    figsize: tuple = (5, 8)) -> plt.Figure:
    """
    Plot percentage change in average bus route speeds: 2025 vs 2024.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    bus_speed = segment_speed_df.groupby(["year", "route_id"])[["weight_travel_time", "weight_distance"]].sum().unstack(level=0)
    bus_speed[2024] = bus_speed[("weight_distance", 2024)] / bus_speed[("weight_travel_time", 2024)]
    bus_speed[2025] = bus_speed[("weight_distance", 2025)] / bus_speed[("weight_travel_time", 2025)]

    bus_speed["25_perc_24"] = (bus_speed[2025] - bus_speed[2024]) / bus_speed[2024].fillna(-np.inf)
    bus_speed = bus_speed.sort_values(by=["25_perc_24"])
    colors_speed = np.where(bus_speed["25_perc_24"] >= 0, 'g', 'r')

    fig = plt.figure(figsize=figsize)
    bus_speed["25_perc_24"].plot(kind="barh", figsize=figsize, color=colors_speed)
    plt.title("Percentage Change in Average Bus Route Speeds (2025 vs 2024)")
    plt.xlabel("Percentage Change")
    plt.ylabel("Bus Route ID")
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(["0% Change", "2025 vs 2024"])
    plt.tight_layout(pad=0.5)
    
    return fig


def plot_cbd_speeds_comparison(segment_speed_cbd: pd.DataFrame, 
                                cbd_bus_routes: pd.DataFrame,
                                figsize: tuple = (14, 5)) -> plt.Figure:
    """
    Plot average CBD bus speeds for weekday and weekend, comparing across years.
    
    Parameters:
    - segment_speed_cbd: DataFrame with CBD segment speed data
    - cbd_bus_routes: DataFrame with CBD bus routes
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    bus_routes_in_cbd = cbd_bus_routes[cbd_bus_routes["cbd_relation"] == "In CBD"].copy()
    segment_speed_cbd = segment_speed_cbd[segment_speed_cbd["route_id"].isin(bus_routes_in_cbd["route_id"])].copy()
    segment_speed_cbd["weekday"] = segment_speed_cbd["day_of_week"].apply(
        lambda x: 1 if x in ["Saturday", "Sunday"] else 0
    )

    hours = list(range(24))
    
    weekday_data = segment_speed_cbd[segment_speed_cbd["weekday"] == 0].groupby(
        ["year", "hour_of_day"]
    )[["weight_travel_time", "weight_distance"]].sum()
    weekday_data["avg_speed"] = weekday_data["weight_distance"] / weekday_data["weight_travel_time"]
    average_speed_cbd_weekday = weekday_data["avg_speed"].unstack(level=0)

    weekend_data = segment_speed_cbd[segment_speed_cbd["weekday"] == 1].groupby(
        ["year", "hour_of_day"]
    )[["weight_travel_time", "weight_distance"]].sum()
    weekend_data["avg_speed"] = weekend_data["weight_distance"] / weekend_data["weight_travel_time"]
    average_speed_cbd_weekend = weekend_data["avg_speed"].unstack(level=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    average_speed_cbd_weekday.reindex(hours).plot(kind="line", ax=axes[0])
    axes[0].set_title("Weekday: Average Road Speed for Bus Routes in CBD")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Average Road Speed (mph)")
    axes[0].axvspan(5, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[0].legend(loc="upper right")

    average_speed_cbd_weekend.reindex(hours).plot(kind="line", ax=axes[1])
    axes[1].set_title("Weekend: Average Road Speed for Bus Routes in CBD")
    axes[1].set_xlabel("Hour of Day")
    axes[1].axvspan(9, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    
    return fig


def plot_vehicles_entering_manhattan(vehicles_entering_manhattan: pd.DataFrame, 
                                      figsize: tuple = (14, 5)) -> plt.Figure:
    """
    Plot average vehicles entering Manhattan by hour for weekday and weekend.
    
    Parameters:
    - vehicles_entering_manhattan: DataFrame with vehicle crossing data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    grouped_vehicles_entering = vehicles_entering_manhattan.groupby(
        ["year", "month", "day", "weekend", "hour"]
    )[["inflow", "outflow", "change"]].sum().reset_index()

    vehicles_entering_manhattan_weekend = grouped_vehicles_entering[grouped_vehicles_entering["weekend"] == 1]
    vehicles_entering_manhattan_weekday = grouped_vehicles_entering[grouped_vehicles_entering["weekend"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Weekday
    weekday_avg = vehicles_entering_manhattan_weekday.groupby(
        ["year", "hour"]
    )[["inflow", "outflow", "change"]].mean().unstack(level=0)
    weekday_avg.plot(ax=axes[0])
    axes[0].axvspan(5, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[0].set_ylabel("Average Number of Vehicles")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_title("Weekday: Average Vehicles Entering Manhattan")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].legend(loc="lower right")
    axes[0].grid(visible=True)

    # Weekend
    weekend_avg = vehicles_entering_manhattan_weekend.groupby(
        ["year", "hour"]
    )[["inflow", "outflow", "change"]].mean().unstack(level=0)
    weekend_avg.plot(ax=axes[1])
    axes[1].axvspan(9, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[1].set_ylabel("Average Number of Vehicles")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_title("Weekend: Average Vehicles Entering Manhattan")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].legend(loc="lower right")
    axes[1].grid(visible=True)

    plt.tight_layout()
    
    return fig


def plot_ridership_by_hour(ridership_data_weekday: pd.DataFrame, 
                            ridership_data_weekend: pd.DataFrame,
                            figsize: tuple = (14, 5)) -> plt.Figure:
    """
    Plot average hourly ridership for weekday and weekend.
    
    Parameters:
    - ridership_data_weekday: DataFrame with weekday ridership
    - ridership_data_weekend: DataFrame with weekend ridership
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Plot weekday
    ridership_data_weekday.groupby(["year", "hour"])["total_ridership"].mean().unstack(level=0).plot(
        kind="line", ax=axes[0]
    )
    axes[0].set_title("Weekday: Average Hourly Ridership in Manhattan Bus Routes")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Average Ridership")
    axes[0].axvspan(5, 20, color="lightblue", alpha=0.3, label="Cong. Pricing Hours")
    axes[0].legend(loc="upper right")

    # Plot weekend
    ridership_data_weekend.groupby(["year", "hour"])["total_ridership"].mean().unstack(level=0).plot(
        kind="line", ax=axes[1]
    )
    axes[1].set_title("Weekend: Average Hourly Ridership in Manhattan Bus Routes")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Average Ridership")
    axes[1].axvspan(9, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    
    return fig


def plot_cbd_entries_by_month(vehicles_entering_cbd: pd.DataFrame, 
                               figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot vehicles entering CBD per month.
    
    Parameters:
    - vehicles_entering_cbd: DataFrame with CBD entry data
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    vehicles_entering_cbd.groupby(
        ["year", "month", "weekend"]
    )["sum_crz_entries"].sum().unstack(level=2).plot(kind="bar", figsize=figsize, stacked=True)
    
    return fig


def plot_cbd_entries_by_hour(vehicles_entering_cbd_weekday: pd.DataFrame,
                              vehicles_entering_cbd_weekend: pd.DataFrame,
                              figsize: tuple = (14, 5)) -> plt.Figure:
    """
    Plot average vehicles entering CBD by hour for weekday and weekend.
    
    Parameters:
    - vehicles_entering_cbd_weekday: DataFrame with weekday CBD entries
    - vehicles_entering_cbd_weekend: DataFrame with weekend CBD entries
    - figsize: Figure size tuple
    
    Returns:
    - matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    vehicles_entering_cbd_weekday.groupby("hour")["sum_crz_entries"].mean().plot(kind="line", ax=axes[0])
    axes[0].axvspan(5, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[0].set_ylabel("Average Number of Vehicles Entering CBD")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_title("Weekday: Average Vehicles Entering CBD")
    axes[0].legend(loc="lower right")

    vehicles_entering_cbd_weekend.groupby("hour")["sum_crz_entries"].mean().plot(kind="line", ax=axes[1])
    axes[1].axvspan(9, 20, color="lightblue", alpha=0.3, label="Congestion Pricing Hours")
    axes[1].set_ylabel("Average Number of Vehicles Entering CBD")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_title("Weekend: Average Vehicles Entering CBD")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    
    return fig


def plot_bus_routes(data_df: pd.DataFrame, year: Optional[int] = None, 
                    month: Optional[int] = None, weekday: Optional[str] = None, 
                    hour: Optional[int] = None, route_id: Optional[str] = None, 
                    marker: Optional[bool] = None) -> folium.Map:
    """
    Plot NYC bus routes with color indicating average speed on an interactive map.
    
    Parameters:
    - data_df: DataFrame with bus route segment data
    - year: Filter by year
    - month: Filter by month
    - weekday: Filter by day of week
    - hour: Filter by hour
    - route_id: Filter by specific route
    - marker: Whether to show stop markers
    
    Returns:
    - folium Map object
    """
    df = data_df.copy()
    if month is not None:
        df = df[df["month"] == month]
    if weekday is not None:
        df = df[df["day_of_week"] == weekday]
    if route_id is not None:
        df = df[df["route_id"] == route_id]
    if hour is not None:
        df = df[df["hour_of_day"] == hour]
    if year is not None:
        df = df[df["year"] == year]

    routes = sorted(df["route_id"].dropna().unique())
    cmap_routes = plt.get_cmap("Paired", len(routes))
    route_colors = {r: mcolors.to_hex(cmap_routes(i)) for i, r in enumerate(routes)}

    speeds = df["average_road_speed"].dropna()
    min_speed, max_speed = speeds.min(), speeds.max()
    norm = mcolors.Normalize(vmin=min_speed, vmax=max_speed)
    cmap_speed = plt.get_cmap("RdYlGn")

    folium_map = folium.Map(location=[40.7831, -73.9712], zoom_start=12, tiles="CartoDB positron")
    marker_cluster = MarkerCluster(name="Stops", disableClusteringAtZoom=10, maxClusterRadius=60).add_to(folium_map)

    for _, row in df.iterrows():
        speed = row["average_road_speed"]
        lat1 = row.get("timepoint_stop_latitude")
        lon1 = row.get("timepoint_stop_longitude")
        lat2 = row.get("next_timepoint_stop_latitude")
        lon2 = row.get("next_timepoint_stop_longitude")
        line_coords = [[lat1, lon1], [lat2, lon2]]
        color = mcolors.to_hex(cmap_speed(norm(speed)))
        route_color = route_colors.get(row["route_id"])
        stop_names = [row["timepoint_stop_name"], row["next_timepoint_stop_name"]]

        if marker:
            for idx in range(2):
                folium.CircleMarker(
                    location=line_coords[idx],
                    radius=7,
                    color=route_color,
                    fill_color=route_color,
                    fill=True,
                    popup=f"Route: {row['route_id']} \n Stop: {stop_names[idx]}",
                ).add_to(marker_cluster)

        folium.PolyLine(
            locations=line_coords,
            color=color,
            weight=3,
            opacity=0.8,
            popup=f"{row['route_id']} | {speed:.1f} mph"
        ).add_to(folium_map)

    return folium_map


def save_all_plots(segment_speed_df: pd.DataFrame, 
                   cbd_bus_routes: pd.DataFrame,
                   vehicles_entering_manhattan: pd.DataFrame,
                   ridership_data_weekday: pd.DataFrame,
                   ridership_data_weekend: pd.DataFrame,
                   vehicles_entering_cbd: pd.DataFrame,
                   vehicles_entering_cbd_weekday: pd.DataFrame,
                   vehicles_entering_cbd_weekend: pd.DataFrame,
                   output_dir: str = "reports/figures"):
    """
    Generate and save all plots to the specified directory.
    
    Parameters:
    - segment_speed_df: DataFrame with segment speed data
    - cbd_bus_routes: DataFrame with CBD bus routes
    - vehicles_entering_manhattan: DataFrame with vehicle crossing data
    - ridership_data_weekday: DataFrame with weekday ridership
    - ridership_data_weekend: DataFrame with weekend ridership
    - vehicles_entering_cbd: DataFrame with CBD entry data
    - vehicles_entering_cbd_weekday: DataFrame with weekday CBD entries
    - vehicles_entering_cbd_weekend: DataFrame with weekend CBD entries
    - output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating speed by hour and bus type plot...")
    fig = plot_speed_by_hour_and_bus_type(segment_speed_df)
    fig.savefig(f"{output_dir}/speed_by_hour_and_bus_type.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating speed by hour and weekday plot...")
    fig = plot_speed_by_hour_and_weekday(segment_speed_df)
    fig.savefig(f"{output_dir}/speed_by_hour_and_weekday.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating speed comparison by year plot...")
    fig = plot_speed_comparison_by_year(segment_speed_df)
    fig.savefig(f"{output_dir}/speed_comparison_by_year.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating speed change vs 2023 plot...")
    fig = plot_speed_change_vs_2023(segment_speed_df)
    fig.savefig(f"{output_dir}/speed_change_vs_2023.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating speed change 2025 vs 2024 plot...")
    fig = plot_speed_change_2025_vs_2024(segment_speed_df)
    fig.savefig(f"{output_dir}/speed_change_2025_vs_2024.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating CBD speeds comparison plot...")
    fig = plot_cbd_speeds_comparison(segment_speed_df, cbd_bus_routes)
    fig.savefig(f"{output_dir}/cbd_speeds_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating vehicles entering Manhattan plot...")
    fig = plot_vehicles_entering_manhattan(vehicles_entering_manhattan)
    fig.savefig(f"{output_dir}/vehicles_entering_manhattan.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating ridership by hour plot...")
    fig = plot_ridership_by_hour(ridership_data_weekday, ridership_data_weekend)
    fig.savefig(f"{output_dir}/ridership_by_hour.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating CBD entries by month plot...")
    fig = plot_cbd_entries_by_month(vehicles_entering_cbd)
    fig.savefig(f"{output_dir}/cbd_entries_by_month.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Generating CBD entries by hour plot...")
    fig = plot_cbd_entries_by_hour(vehicles_entering_cbd_weekday, vehicles_entering_cbd_weekend)
    fig.savefig(f"{output_dir}/cbd_entries_by_hour.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"All plots saved to {output_dir}/")
