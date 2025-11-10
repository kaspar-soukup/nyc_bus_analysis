import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(page_title="NYC Bus Speed Analysis", page_icon="🚌", layout="wide")
st.title("🚌 NYC Bus Speed & Congestion Pricing Analysis")

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

# Load pre-processed data with caching
@st.cache_data
def load_data():
    """Load all pre-processed datasets."""
    speed_overall = pd.read_csv(PROCESSED_DIR / "speed_overall.csv")
    speed_monthly = pd.read_csv(PROCESSED_DIR / "speed_monthly.csv")
    segment_speed = pd.read_csv(PROCESSED_DIR / "segment_speed_processed.csv")
    
    # Convert year_month to datetime for plotting
    speed_overall['year_month'] = pd.to_datetime(speed_overall['year_month'] + '-01')
    speed_monthly['year_month'] = pd.to_datetime(speed_monthly['year_month'] + '-01')
    
    return speed_overall, speed_monthly, segment_speed

with st.spinner('Loading data...'):
    speed_overall, speed_monthly, segment_speed = load_data()

# Section 1: Average Bus Speeds Over Time
st.header("📊 Average Bus Speeds Based on Segments")
st.markdown("**Data Range:** 2023 - September 2025")

# Create plots with updated style
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

for ax, weekend_val, title in [(ax1, 0, 'Weekdays'), (ax2, 1, 'Weekends')]:
    ax.set_facecolor('white')
    data_subset = speed_overall[speed_overall['weekend'] == weekend_val]
    
    for is_cbd in [True, False]:
        data = data_subset[data_subset['is_cbd_segment'] == is_cbd].sort_values('year_month')
        label = 'CBD Segments' if is_cbd else 'Non-CBD Segments'
        color = 'darkred' if is_cbd else 'darkblue'
        
        # Plot with rolling average
        ax.plot(data['year_month'], data['avg_speed_mph'], marker='o', label=label, 
                color=color, linewidth=1.5, alpha=0.4, markersize=3)
        rolling_avg = data['avg_speed_mph'].rolling(window=3, center=True).mean()
        ax.plot(data['year_month'], rolling_avg, label=f'{label} (3-mo avg)', 
                color=color, linewidth=2.5, alpha=1.0)
    
    # Add shaded area for congestion pricing period
    pricing_start = pd.Timestamp('2025-01-05')
    if data_subset['year_month'].max() >= pricing_start:
        ax.axvspan(pricing_start, data_subset['year_month'].max(), alpha=0.15, color='lightblue', 
                   label='Congestion Pricing Active')
    
    ax.set_title(f'Average Bus Speed - {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Speed (mph)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# Section 2: Difference-in-Differences Analysis
st.header("📈 Difference-in-Differences Analysis")

# Categorize periods
def categorize_period_3mo(row):
    year, month = row['year'], row['month']
    if year == 2023 and month >= 10: return 'oct_dec_2023'
    elif year == 2024 and month <= 3: return 'jan_mar_2024'
    elif year == 2024 and month >= 10: return 'oct_dec_2024'
    elif year == 2025 and month <= 3: return 'jan_mar_2025'
    return 'other'

def categorize_period_6mo(row):
    year, month = row['year'], row['month']
    if year == 2023 and month >= 7: return 'jul_dec_2023'
    elif year == 2024 and month <= 6: return 'jan_jun_2024'
    elif year == 2024 and month >= 7: return 'jul_dec_2024'
    elif year == 2025 and month <= 6: return 'jan_jun_2025'
    return 'other'

# Pricing hours filter
def in_pricing_hours(row):
    hour = row.get('hour', row.get('hour_of_day', 0))
    weekend = row['weekend']
    if weekend == 0: return 5 <= hour < 22  # Weekday 5am-9pm
    return 9 <= hour < 22  # Weekend 9am-9pm

# Prepare segment data for DiD
segment_speed['timestamp'] = pd.to_datetime(segment_speed['timestamp'], errors='coerce')
segment_speed['year_month'] = segment_speed['timestamp'].dt.to_period('M').dt.to_timestamp()
segment_speed['in_pricing_hours'] = segment_speed.apply(in_pricing_hours, axis=1)

col1, col2 = st.columns(2)

# 3-month DiD
with col1:
    st.subheader("3-Month DiD Analysis")
    segment_speed['period_3mo'] = segment_speed.apply(categorize_period_3mo, axis=1)
    did_data_3mo = segment_speed[
        (segment_speed['period_3mo'] != 'other') & 
        (segment_speed['is_cbd_segment'] == True) & 
        (segment_speed['in_pricing_hours'] == True)
    ].copy()
    
    # Aggregate and run regression
    did_agg_3mo = did_data_3mo.groupby(['route_id', 'period_3mo']).agg({
        'weight_distance': 'sum', 'weight_travel_time': 'sum'
    }).reset_index()
    did_agg_3mo['avg_speed_mph'] = did_agg_3mo['weight_distance'] / did_agg_3mo['weight_travel_time']
    did_agg_3mo['treatment'] = did_agg_3mo['period_3mo'].isin(['jan_mar_2025']).astype(int)
    did_agg_3mo['post'] = did_agg_3mo['period_3mo'].isin(['jan_mar_2024', 'jan_mar_2025']).astype(int)
    did_agg_3mo['did'] = did_agg_3mo['treatment'] * did_agg_3mo['post']
    
    model_3mo = smf.ols('avg_speed_mph ~ treatment + post + did + C(route_id)', data=did_agg_3mo).fit()
    
    st.metric("DiD Coefficient", f"{model_3mo.params['did']:.4f} mph")
    st.metric("P-value", f"{model_3mo.pvalues['did']:.4f}")
    st.metric("Observations", f"{len(did_agg_3mo):,}")

# 6-month DiD
with col2:
    st.subheader("6-Month DiD Analysis")
    segment_speed['period_6mo'] = segment_speed.apply(categorize_period_6mo, axis=1)
    did_data_6mo = segment_speed[
        (segment_speed['period_6mo'] != 'other') & 
        (segment_speed['is_cbd_segment'] == True) & 
        (segment_speed['in_pricing_hours'] == True)
    ].copy()
    
    # Aggregate and run regression
    did_agg_6mo = did_data_6mo.groupby(['route_id', 'period_6mo']).agg({
        'weight_distance': 'sum', 'weight_travel_time': 'sum'
    }).reset_index()
    did_agg_6mo['avg_speed_mph'] = did_agg_6mo['weight_distance'] / did_agg_6mo['weight_travel_time']
    did_agg_6mo['treatment'] = did_agg_6mo['period_6mo'].isin(['jan_jun_2025']).astype(int)
    did_agg_6mo['post'] = did_agg_6mo['period_6mo'].isin(['jan_jun_2024', 'jan_jun_2025']).astype(int)
    did_agg_6mo['did'] = did_agg_6mo['treatment'] * did_agg_6mo['post']
    
    model_6mo = smf.ols('avg_speed_mph ~ treatment + post + did + C(route_id)', data=did_agg_6mo).fit()
    
    st.metric("DiD Coefficient", f"{model_6mo.params['did']:.4f} mph")
    st.metric("P-value", f"{model_6mo.pvalues['did']:.4f}")
    st.metric("Observations", f"{len(did_agg_6mo):,}")

# Section 3: Fastest and Slowest Buses
st.header("🏆 Fastest and Slowest Bus Routes")

# Last month and same month last year
current_date = segment_speed['timestamp'].max()
last_month = current_date.replace(day=1)
last_year_month = last_month - pd.DateOffset(years=1)

col1, col2 = st.columns(2)

for col, target_month, title in [(col1, last_month, f"Last Month ({last_month.strftime('%B %Y')})"),
                                   (col2, last_year_month, f"Same Month Last Year ({last_year_month.strftime('%B %Y')})")]:
    with col:
        st.subheader(title)
        
        month_data = segment_speed[
            (segment_speed['year_month'] == target_month) & 
            (segment_speed['is_cbd_segment'] == True)
        ]
        
        route_speeds = month_data.groupby('route_id').agg({
            'weight_distance': 'sum',
            'weight_travel_time': 'sum'
        }).reset_index()
        route_speeds['avg_speed_mph'] = route_speeds['weight_distance'] / route_speeds['weight_travel_time']
        route_speeds = route_speeds.sort_values('avg_speed_mph')
        
        # Slowest 5
        st.write("**🐌 Slowest 5 Routes:**")
        slowest = route_speeds.head(5)
        for idx, row in slowest.iterrows():
            st.write(f"• {row['route_id']}: {row['avg_speed_mph']:.2f} mph")
        
        st.write("")
        
        # Fastest 5
        st.write("**🚀 Fastest 5 Routes:**")
        fastest = route_speeds.tail(5)
        for idx, row in fastest[::-1].iterrows():
            st.write(f"• {row['route_id']}: {row['avg_speed_mph']:.2f} mph")

# Footer
st.markdown("---")
st.caption("Data Source: NYC MTA | Analysis: CBD Congestion Pricing Impact Study")
