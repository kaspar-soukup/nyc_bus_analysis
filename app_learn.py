"""
Streamlit Learning App - NYC Bus Analysis
Simple version with basic filters and map visualization
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="NYC Bus Analysis", page_icon="🚌", layout="wide")
st.title("🚌 NYC Bus Speed Analysis")

@st.cache_data
def load_data():
    """Load and prepare the bus data"""
    df = pd.read_csv("visualization_24_25.csv")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df['hour_of_day'] = pd.to_numeric(df['hour_of_day'], errors='coerce')
    df['average_road_speed'] = pd.to_numeric(df['average_road_speed'], errors='coerce')
    return df

with st.spinner("Loading data..."):
    data_df = load_data()

st.success(f"✅ Data loaded! {len(data_df):,} rows, {data_df['route_id'].nunique()} routes")

if len(data_df) == 0:
    st.error("❌ DataFrame is empty! Check extended_list.csv file.")
    st.stop()

# Sidebar filters
st.sidebar.header("🎛️ Filters")

years = sorted(data_df["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", options=["All"] + list(years))

routes = sorted(data_df['route_id'].dropna().unique())
selected_route = st.sidebar.selectbox("Select Route", options=['All Routes'] + list(routes))

hour = st.sidebar.selectbox("Select Hour", options=[2,8,14,20])

days = data_df["day_of_week"].unique()
selected_day = st.sidebar.selectbox("Select Day", options=['All'] + list(days))

show_markers = st.sidebar.toggle("Show Markers", value=False)

# Apply filters
fdf = data_df.copy()
if selected_year != "All":
    fdf = fdf[fdf['year'] == selected_year]
if selected_route != 'All Routes':
    fdf = fdf[fdf['route_id'] == selected_route]
if selected_day != 'All':
    fdf = fdf[fdf['day_of_week'] == selected_day]
fdf = fdf[fdf['hour_of_day'] == hour]

# Summary metrics
st.subheader("📊 Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Speed", f"{fdf['average_road_speed'].mean():.1f} mph")
col2.metric("Segments", f"{len(fdf):,}")
col3.metric("Routes", f"{fdf['route_id'].nunique()}")

# Data preview
st.subheader("Data Preview")
st.dataframe(fdf[['year', 'month', 'day_of_week', 'hour_of_day', 'route_id', 'average_road_speed']].head(10))

# Map functions
def get_speed_color(speed):
    """Map speed to color"""
    if pd.isna(speed):
        return "#888888"
    elif speed < 4:
        return "#E00505"  # Dark red
    elif speed < 6:
        return "#E99217"  # Red/Orange
    elif speed < 10:
        return "#FBF30A"  # Yellow
    elif speed < 14:
        return "#C6F40E"  # Light green
    else:
        return "#009700"  # Dark green

@st.cache_data
def precompute_route_colors(data):
    """Pre-compute route colors"""
    routes = sorted(data['route_id'].dropna().unique())
    cmap = plt.get_cmap('Paired', len(routes))
    return {r: mcolors.to_hex(cmap(i)) for i, r in enumerate(routes)}

def create_map(df, show_markers=False):
    """Create Folium map"""
    if len(df) == 0:
        return None
    
    # Sample for performance
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=42)
    
    route_colors = precompute_route_colors(df)
    
    m = folium.Map(
        location=[40.7831, -73.9712],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    marker_cluster = MarkerCluster().add_to(m) if show_markers else None
    
    valid_rows = df.dropna(subset=['timepoint_stop_latitude', 'timepoint_stop_longitude', 
                                     'next_timepoint_stop_latitude', 'next_timepoint_stop_longitude'])
    
    for _, row in valid_rows.iterrows():
        speed = row["average_road_speed"]
        coords = [[row["timepoint_stop_latitude"], row["timepoint_stop_longitude"]], 
                  [row["next_timepoint_stop_latitude"], row["next_timepoint_stop_longitude"]]]
        
        color = get_speed_color(speed)
        route_color = route_colors.get(row["route_id"], "#888888")
        
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=2,
            opacity=0.7,
            popup=f"<b>{row['route_id']}</b> | {speed:.1f} mph"
        ).add_to(m)
        
        if show_markers and marker_cluster:
            for i, stop_name in enumerate([row["timepoint_stop_name"], row["next_timepoint_stop_name"]]):
                folium.CircleMarker(
                    location=coords[i],
                    radius=3,
                    color=route_color,
                    fill_color=route_color,
                    fill=True,
                    popup=f"<b>{row['route_id']}</b><br>{stop_name}",
                    weight=1,
                ).add_to(marker_cluster)
    
    return m

def create_heatmap(df):
    """Create a heatmap visualization using speed as intensity"""
    if len(df) == 0:
        return None
    
    # Prepare heatmap data: [latitude, longitude, intensity]
    heatmap_data = df[['timepoint_stop_latitude', 'timepoint_stop_longitude', 'average_road_speed']].dropna().values.tolist()
    
    if len(heatmap_data) == 0:
        return None
    
    # Create base map
    hm = folium.Map(
        location=[40.7831, -73.9712],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    # Add heatmap layer with gradient
    HeatMap(
        heatmap_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=15,
        blur=15,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    ).add_to(hm)
    
    return hm

# Map display
st.subheader("🗺️ Interactive Map")
st.markdown("""
**Speed Color Legend:**
- 🔴 **Dark Red** = < 4 mph
- 🔴 **Red/Orange** = 4-6 mph
- 🟡 **Yellow** = 6-10 mph
- 🟢 **Light Green** = 10-14 mph
- 🟢 **Dark Green** = ≥ 14 mph
""")

with st.spinner("Generating map..."):
    bus_map = create_map(fdf, show_markers=show_markers)
    if bus_map:
        st_folium(bus_map, width=1400, height=600)
    else:
        st.error("No data to display")

# Heatmap display
st.subheader("🔥 Traffic Heatmap (by Speed)")
st.markdown("""
**Heatmap Intensity Legend:**
- 🔵 **Blue** = Lowest speeds (congested)
- 🟦 **Cyan/Green** = Moderate speeds
- 🟨 **Yellow** = Good speeds
- 🔴 **Red** = Highest speeds (free-flowing)
""")

with st.spinner("Generating heatmap..."):
    heatmap = create_heatmap(fdf)
    if heatmap:
        st_folium(heatmap, width=1400, height=600)
    else:
        st.warning("No data available for heatmap")

# Charts
st.subheader("📈 Charts")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    fdf['average_road_speed'].hist(bins=30, ax=ax1, color="#1f77b4", edgecolor="black")
    ax1.set_xlabel("Speed (mph)")
    ax1.set_ylabel("Count")
    ax1.set_title("Speed Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    hourly = fdf.groupby('hour_of_day')['average_road_speed'].mean()
    hourly.plot(ax=ax2, marker='o', color="#2ca02c")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Avg Speed (mph)")
    ax2.set_title("Average Speed by Hour")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.markdown("---")
st.caption("NYC Bus Analysis - Simple Streamlit App")
