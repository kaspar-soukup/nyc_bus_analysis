"""
NYC Bus Speed Analysis Dashboard
Interactive visualization of bus speeds before and after congestion pricing
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="NYC Bus Speed Analysis",
    page_icon="🚌",
    layout="wide"
)

# Define paths
PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

@st.cache_data
def load_data():
    """Load pre-processed data from pipeline"""
    speed_overall = pd.read_csv(PROCESSED_DIR / "speed_overall.csv")
    speed_monthly = pd.read_csv(PROCESSED_DIR / "speed_monthly.csv")
    
    # Convert year_month to datetime
    speed_overall['year_month'] = pd.to_datetime(speed_overall['year_month'] + '-01')
    speed_monthly['year_month'] = pd.to_datetime(speed_monthly['year_month'] + '-01')
    
    return speed_overall, speed_monthly

# Load data
try:
    speed_overall, speed_monthly = load_data()
except FileNotFoundError:
    st.error("⚠️ Processed data files not found. Please run the data pipeline first: `python main.py --process`")
    st.stop()

# Header
st.title("🚌 NYC Bus Speed Analysis")
st.markdown("### Impact of Congestion Pricing on Bus Speeds")
st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")
view_type = st.sidebar.radio(
    "View Type",
    ["Overall (All Routes Combined)", "By Individual Routes"]
)

show_cbd_only = st.sidebar.checkbox("Show CBD segments only", value=False)
show_rolling_avg = st.sidebar.checkbox("Show 3-month rolling average", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.info(
    "This dashboard shows NYC bus speeds before and after the introduction of "
    "congestion pricing on January 5, 2025. Data is classified by CBD segments "
    "using the official MTA geofence."
)

# Main content
if view_type == "Overall (All Routes Combined)":
    st.header("📊 Overall Bus Speed Trends")
    
    # Filter data
    data = speed_overall.copy()
    if show_cbd_only:
        data = data[data['is_cbd_segment'] == True]
    
    # Create plots for weekday and weekend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    pricing_start = pd.Timestamp('2025-01-05')
    
    for ax, weekend_val, title in [(ax1, 0, 'Weekdays'), (ax2, 1, 'Weekends')]:
        ax.set_facecolor('white')
        data_subset = data[data['weekend'] == weekend_val]
        
        # Determine which categories to show
        if show_cbd_only:
            categories = [True]
        else:
            categories = [True, False]
        
        for is_cbd in categories:
            subset = data_subset[data_subset['is_cbd_segment'] == is_cbd].sort_values('year_month')
            if len(subset) == 0:
                continue
                
            label = 'CBD Segments' if is_cbd else 'Non-CBD Segments'
            color = 'darkred' if is_cbd else 'darkblue'
            
            # Plot monthly data
            ax.plot(subset['year_month'], subset['avg_speed_mph'], 
                   marker='o', label=label, color=color, 
                   linewidth=1.5, alpha=0.4, markersize=4)
            
            # Add rolling average if enabled
            if show_rolling_avg:
                rolling_avg = subset['avg_speed_mph'].rolling(window=3, center=True).mean()
                ax.plot(subset['year_month'], rolling_avg, 
                       label=f'{label} (3-mo avg)', 
                       color=color, linewidth=2.5, alpha=1.0)
        
        # Add pricing start line
        ax.axvline(pricing_start, color='green', linestyle='--', 
                  linewidth=2, label='Congestion Pricing Start', alpha=0.7)
        
        # Formatting
        ax.set_title(f'Average Bus Speed - {title}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Average Speed (mph)', fontsize=11)
        ax.grid(True, alpha=0.3, color='gray')
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle('NYC Bus Speeds: Geofence-Based CBD Classification', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Weekday Speeds**")
        weekday_data = data[data['weekend'] == 0]
        if show_cbd_only:
            weekday_cbd = weekday_data[weekday_data['is_cbd_segment'] == True]
            st.metric("CBD Average", f"{weekday_cbd['avg_speed_mph'].mean():.2f} mph")
            st.metric("CBD Std Dev", f"{weekday_cbd['avg_speed_mph'].std():.2f} mph")
        else:
            cbd_data = weekday_data[weekday_data['is_cbd_segment'] == True]
            non_cbd_data = weekday_data[weekday_data['is_cbd_segment'] == False]
            st.metric("CBD Average", f"{cbd_data['avg_speed_mph'].mean():.2f} mph")
            st.metric("Non-CBD Average", f"{non_cbd_data['avg_speed_mph'].mean():.2f} mph")
    
    with col2:
        st.markdown("**Weekend Speeds**")
        weekend_data = data[data['weekend'] == 1]
        if show_cbd_only:
            weekend_cbd = weekend_data[weekend_data['is_cbd_segment'] == True]
            st.metric("CBD Average", f"{weekend_cbd['avg_speed_mph'].mean():.2f} mph")
            st.metric("CBD Std Dev", f"{weekend_cbd['avg_speed_mph'].std():.2f} mph")
        else:
            cbd_data = weekend_data[weekend_data['is_cbd_segment'] == True]
            non_cbd_data = weekend_data[weekend_data['is_cbd_segment'] == False]
            st.metric("CBD Average", f"{cbd_data['avg_speed_mph'].mean():.2f} mph")
            st.metric("Non-CBD Average", f"{non_cbd_data['avg_speed_mph'].mean():.2f} mph")

else:  # By Individual Routes
    st.header("📊 Individual Route Analysis")
    
    # Get list of routes
    available_routes = sorted(speed_monthly['route_id'].unique())
    
    # Route selector
    selected_routes = st.multiselect(
        "Select routes to display",
        available_routes,
        default=available_routes[:5] if len(available_routes) >= 5 else available_routes
    )
    
    if not selected_routes:
        st.warning("Please select at least one route to display.")
    else:
        # Filter data
        route_data = speed_monthly[speed_monthly['route_id'].isin(selected_routes)]
        
        if show_cbd_only:
            route_data = route_data[route_data['is_cbd_segment'] == True]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor('white')
        
        pricing_start = pd.Timestamp('2025-01-05')
        
        for ax, weekend_val, title in [(ax1, 0, 'Weekdays'), (ax2, 1, 'Weekends')]:
            ax.set_facecolor('white')
            
            for route in selected_routes:
                subset = route_data[
                    (route_data['route_id'] == route) & 
                    (route_data['weekend'] == weekend_val)
                ].sort_values('year_month')
                
                if len(subset) == 0:
                    continue
                
                # Plot route
                ax.plot(subset['year_month'], subset['avg_speed_mph'], 
                       marker='o', label=f'Route {route}', 
                       linewidth=1.5, alpha=0.7, markersize=3)
            
            # Add pricing start line
            ax.axvline(pricing_start, color='green', linestyle='--', 
                      linewidth=2, label='Congestion Pricing Start', alpha=0.7)
            
            # Formatting
            ax.set_title(f'Average Bus Speed by Route - {title}', fontsize=13, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Average Speed (mph)', fontsize=11)
            ax.grid(True, alpha=0.3, color='gray')
            ax.legend(loc='best', fontsize=8, ncol=2)
        
        plt.suptitle(f'Bus Speeds for Selected Routes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Route comparison table
        st.subheader("📊 Route Comparison")
        
        comparison_data = []
        for route in selected_routes:
            route_subset = route_data[route_data['route_id'] == route]
            weekday_avg = route_subset[route_subset['weekend'] == 0]['avg_speed_mph'].mean()
            weekend_avg = route_subset[route_subset['weekend'] == 1]['avg_speed_mph'].mean()
            
            comparison_data.append({
                'Route': route,
                'Weekday Avg (mph)': f"{weekday_avg:.2f}",
                'Weekend Avg (mph)': f"{weekend_avg:.2f}",
                'Total Observations': len(route_subset)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "**Data Source:** MTA Bus Speed Data | "
    "**Congestion Pricing Start Date:** January 5, 2025 | "
    "**Last Updated:** November 2025"
)
