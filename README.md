# NYC Bus Analysis Project

A data science project analyzing NYC MTA bus performance, congestion pricing impacts, and traffic patterns in Manhattan using geofence-based CBD classification and Difference-in-Differences (DiD) analysis.

## 🎯 Project Overview

This project analyzes the impact of NYC's congestion pricing (started January 5, 2025) on bus speeds in the Central Business District (CBD). Key features:

- **Geofence-based CBD Classification**: Uses actual geographic boundaries instead of route lists
- **Pre-processed Data Pipeline**: All data cleaning and aggregation happens in one place
- **Interactive Streamlit Dashboard**: Real-time visualization of analysis results
- **Difference-in-Differences Analysis**: 3-month and 6-month comparisons to measure congestion pricing effects

## 📊 Quick Start

### Run the Full Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Set your MTA API token (optional, for fetching new data)
export APP_TOKEN='your_token_here'

# Run the complete pipeline (fetch, process, visualize)
python main.py --all

# Or just process existing data
python main.py --process
```

### Launch the Streamlit App
```bash
streamlit run app.py
```
Access at: http://localhost:8501

The app shows:
- **Overall Trends**: Speed graphs (2023-2025) with 3-month rolling averages
- **CBD vs Non-CBD**: Compare speeds inside and outside congestion pricing zone
- **Individual Routes**: Analyze specific bus routes over time
- **Summary Statistics**: Average speeds by day type and location

### Explore the Analysis Notebook
Open `notebooks/Analysis.ipynb` for detailed statistical analysis, visualizations, and interpretations.

## 🏗️ Project Structure

```
Bus Project Data Science/
├── README.md                  # This file
├── requirements.txt          # Python dependencies
├── main.py                   # Main data pipeline (fetch → process → visualize)
├── app.py                    # Streamlit dashboard (visualization only)
│
├── Data/                     # Data directory
│   ├── raw/                  # Original MTA data from API
│   ├── processed/            # Pipeline output - ready for analysis
│   │   ├── segment_speed_processed.csv    # With is_cbd_segment column
│   │   ├── speed_monthly.csv              # Monthly aggregates by route
│   │   ├── speed_overall.csv              # Overall monthly aggregates
│   │   ├── bus_speed_processed.csv
│   │   ├── ridership_processed.csv
│   │   └── cbd_entries_processed.csv
│   └── interim/              # Intermediate processing (if needed)
│
├── notebooks/                # Jupyter notebooks
│   └── Analysis.ipynb        # Main analysis notebook (USE THIS)
│
├── src/                      # Source code modules
│   ├── config.py            # Configuration and constants
│   ├── data/
│   │   └── make_dataset.py  # API calls and data ingestion
│   ├── features/
│   │   └── build_features.py # Data cleaning, CBD classification, aggregation
│   ├── models/
│   │   └── train_model.py   # Statistical models
│   └── visualization/
│       └── visualize.py     # Plotting functions
│
└── reports/                 # Generated outputs
    ├── figures/             # Saved plots
    └── *.csv               # Model results and metrics
```

## 🔄 Data Pipeline

The pipeline (`main.py`) performs three main steps:

### 1. Data Ingestion (`--fetch-data`)
- Fetches data from NYC MTA Open Data API
- Downloads: segment speeds, bus speeds, ridership, crossings, CBD routes, geofence
- Saves raw data to `Data/raw/`

### 2. Data Processing (`--process`) ⭐ **Core Pipeline**
- **Cleans and merges** datasets (2023-2024 + 2025 data)
- **CBD Classification**: Uses CBD geofence GeoJSON to classify segments
  - Calculates segment midpoints from start/end coordinates
  - Checks if midpoint falls within CBD polygon
  - Adds `is_cbd_segment` boolean column
- **Creates Aggregations**:
  - `speed_monthly.csv` - Monthly aggregates by route (4,000+ rows)
  - `speed_overall.csv` - Overall monthly aggregates (130+ rows)
- **Adds Temporal Features**: hour, weekend, congestion_pricing_timeframe
- Outputs to `Data/processed/`

**Why this matters**: All downstream analysis (notebook + Streamlit app) uses the same pre-processed data, ensuring consistency.

### 3. Visualization (`--visualize`)
- Generates plots for speed trends, ridership, crossings
- Saves to `reports/figures/`

## 📓 Analysis Notebook

`notebooks/Analysis.ipynb` contains:
- **Data Loading**: Uses pre-processed CSVs from pipeline
- **Exploratory Analysis**: Speed trends, seasonality, day-of-week patterns
- **Geofence Visualization**: Maps showing CBD boundaries and bus routes
- **Difference-in-Differences (DiD)**:
  - 3-month analysis (Oct-Dec vs Jan-Mar comparisons)
  - 6-month analysis (Jul-Dec vs Jan-Jun comparisons)
  - Controls for route fixed effects
  - Filters to pricing hours only (weekday 5am-9pm, weekend 9am-9pm)
- **Statistical Tests**: Significance testing, confidence intervals
- **Interpretations**: What the results mean for policy

**To run**: 
1. First run `python main.py --process` to generate processed data
2. Open notebook and run all cells

## 🎨 Streamlit Dashboard

`app.py` provides an interactive web interface for exploring bus speed data:

**Features:**
- **Overall View**: Combined speed trends for all routes with CBD vs Non-CBD comparison
- **Individual Route View**: Select and compare specific bus routes
- **Interactive Filters**:
  - Show CBD segments only
  - Toggle 3-month rolling averages
  - Weekday vs Weekend comparisons
- **Visual Indicators**: Congestion pricing start date marked on all graphs
- **Summary Statistics**: Average speeds, standard deviations by location and day type

**Performance**: Loads in 2-3 seconds using pre-processed data!

**Note**: For statistical analysis (DiD regression), see `notebooks/Analysis.ipynb`

## 📈 Key Findings

The analysis uses Difference-in-Differences (DiD) methodology to estimate the causal effect of congestion pricing:

- **3-Month DiD**: Compares Q4 2023/2024 to Q1 2024/2025
- **6-Month DiD**: Compares Jul-Dec 2023/2024 to Jan-Jun 2024/2025

**Analysis focuses on:**
- CBD segments only (geofence-based classification)
- Pricing hours only (weekday 5am-9pm, weekend 9am-9pm)
- Route fixed effects to control for inherent route differences

For detailed results and statistical significance, see `notebooks/Analysis.ipynb`.

## 🛠️ Technical Details

### CBD Classification Method
- **Input**: Segment start/end lat/lon coordinates
- **Process**: Calculate midpoint, check containment in CBD geofence polygon
- **Library**: Shapely for geometric operations (vectorized for performance)
- **Time**: ~30-60 seconds for 2.2M rows

### Weighted Speed Calculation
```python
weight_distance = road_distance × bus_trip_count
weight_travel_time = average_travel_time × bus_trip_count
avg_speed_mph = sum(weight_distance) / sum(weight_travel_time)
```

### DiD Model Specification
```python
avg_speed_mph ~ treatment + post + did + C(route_id)
```
Where:
- `treatment = 1` only for Jan-Mar 2025 (or Jan-Jun 2025 in 6-month)
- `post = 1` for second half of periods (2024 and 2025)
- `did = treatment × post` (the coefficient of interest)
- Route fixed effects control for inherent route differences

## 🔧 Configuration

Edit `src/config.py` to change:
- Data directories
- API endpoints  
- Date ranges
- Analysis parameters

## 📦 Dependencies

Key packages:
- `pandas`, `numpy` - Data manipulation
- `geopandas`, `shapely` - Geospatial operations
- `matplotlib` - Visualization
- `statsmodels` - Regression analysis
- `streamlit` - Web dashboard
- `sodapy` - MTA API client

See `requirements.txt` for complete list.

## 💡 Tips

- Run `python main.py --process` after updating raw data
- Streamlit app auto-reloads when files change during development
- Check `reports/` for saved figures and model outputs
- Notebook kernel should be restarted after running pipeline for fresh data

## 🚀 Deployment

### Streamlit Cloud

The app is deployment-ready! The repository includes the minimal processed data files needed (`speed_overall.csv` and `speed_monthly.csv` - only ~344KB total).

**To deploy:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file path to: `app.py`
4. Deploy!

**Note:** Large raw data files (>1GB) are excluded from the repository via `.gitignore`. The Streamlit app only needs the small aggregated files which are included.

## 🤝 Contributing

When making changes:
1. Update pipeline functions in `src/features/build_features.py`
2. Re-run `python main.py --process` to regenerate processed data
3. Verify both notebook and Streamlit app produce consistent results
4. Update this README for major feature additions

---

**Note**: This project uses geofence-based CBD classification for more accurate geographic analysis compared to route-based methods.
