# Code Improvement Suggestions

## Overview
This document contains suggestions for improving the code quality, maintainability, and efficiency of your NYC Bus Analysis project. Your existing code has been preserved in the modular pipeline, and these are recommendations for future enhancements.

---

## 1. Function Extraction and Modularity

### Current Issues
- Many code blocks in the notebook perform similar operations repeatedly (e.g., data loading, column renaming, date parsing)
- Complex operations are embedded inline without reusable functions

### Recommendations

#### 1.1 Extract Repetitive Data Loading Patterns
**Current Pattern:**
```python
query_25 = """SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}"""
base_name = "MTA_Bus_Route_Segment_Speeds_Beginning_2025"
existing_file = next((f for f in os.listdir("Data") if f.startswith(base_name)), None)
if existing_file:
    filename = os.path.join("Data", existing_file)
    print(f"Found existing file {filename}, skipping API pull.")
else:
    filename = f"Data/{base_name}_{datetime.now().strftime('%Y%m%d')}.csv"
    df = get_all_MTA_data("kufs-yh3x", query_25, 100000, filename)
```

**Suggested Improvement:**
✅ Already implemented in `src/data/make_dataset.py` as `fetch_or_load_data()` function.

---

#### 1.2 Create Helper Functions for Date/Time Operations
**Current Pattern:**
```python
df["month"] = pd.to_datetime(df["date"]).dt.month
df["year"] = pd.to_datetime(df["date"]).dt.year
df["weekend"] = df["transit_timestamp"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
```

**Suggested Function:**
```python
def add_temporal_features(df, datetime_col='transit_timestamp'):
    """Add standardized temporal features to dataframe."""
    df = df.copy()
    ts = pd.to_datetime(df[datetime_col])
    df['year'] = ts.dt.year
    df['month'] = ts.dt.month
    df['day'] = ts.dt.day
    df['hour'] = ts.dt.hour
    df['day_of_week'] = ts.dt.day_name()
    df['weekend'] = ts.dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    return df
```

**Benefits:**
- Consistent temporal features across all datasets
- Single source of truth for weekend definition
- Easy to modify if business logic changes

---

#### 1.3 Standardize Column Name Cleaning
**Current Pattern:**
```python
bus_speed_seg_2025.columns = bus_speed_seg_2025.columns.str.lower()
bus_speed_seg_2025.columns = bus_speed_seg_2025.columns.str.replace(" ", "_")
```

**Suggested Function:**
```python
def standardize_column_names(df):
    """Standardize column names: lowercase, spaces to underscores."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df
```

---

## 2. Data Validation and Error Handling

### Current Issues
- No validation that expected columns exist before operations
- Missing error handling for API failures
- No data quality checks

### Recommendations

#### 2.1 Add Input Validation
**Example:**
```python
def validate_dataframe_schema(df, required_columns, df_name="DataFrame"):
    """Validate that required columns exist in dataframe."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{df_name} missing required columns: {missing_cols}")
    return True

def clean_segment_speed_data(df_2025, df_2023_2024):
    required_cols = ['year', 'month', 'day_of_week', 'hour_of_day', 'route_id']
    validate_dataframe_schema(df_2025, required_cols, "2025 segment data")
    validate_dataframe_schema(df_2023_2024, required_cols, "2023-2024 segment data")
    # ... rest of function
```

#### 2.2 Add Data Quality Checks
**Example:**
```python
def check_data_quality(df, name="dataset"):
    """Print data quality report."""
    print(f"\n=== Data Quality Report: {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Check for suspicious values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in {col}")
```

#### 2.3 Add Try-Except Blocks for API Calls
**Example:**
```python
def fetch_or_load_data(data_dir, base_name, dataset_id, query, limit, client):
    try:
        existing_file = next((f for f in os.listdir(data_dir) if f.startswith(base_name)), None)
        
        if existing_file:
            filename = os.path.join(data_dir, existing_file)
            print(f"Found existing file {filename}")
            df = pd.read_csv(filename)
        else:
            filename = f"{data_dir}/{base_name}_{datetime.now().strftime('%Y%m%d')}.csv"
            df = get_all_MTA_data(client, dataset_id, query, limit, filename)
        
        return filename, df
    
    except FileNotFoundError as e:
        print(f"Error: Data directory not found - {e}")
        raise
    except Exception as e:
        print(f"Error fetching/loading data: {e}")
        raise
```

---

## 3. Configuration Management

### Current Issues
- Magic numbers and strings scattered throughout code
- Dataset IDs hardcoded in multiple places
- No central configuration

### Recommendations

#### 3.1 Use Configuration File (Already Implemented)
✅ `src/config.py` has been created with centralized configuration.

#### 3.2 Consider Using Environment Variables
**Create `.env` file:**
```bash
APP_TOKEN=your_token_here
DATA_DIR=data/raw
PROCESSED_DIR=data/processed
```

**Load in code:**
```python
from dotenv import load_dotenv
load_dotenv()

app_token = os.getenv('APP_TOKEN')
```

---

## 4. Code Organization and Structure

### Current Issues
- Long cells with multiple operations
- Mixing of concerns (data loading + processing + visualization)
- Difficult to test individual components

### Recommendations

#### 4.1 Separate Concerns
**Current Pattern:**
```python
# All in one cell
df = pd.read_csv("file.csv")
df["new_col"] = df["old_col"] * 2
df.groupby("key").sum().plot()
plt.show()
```

**Suggested Pattern:**
```python
# Data loading
df = load_data("file.csv")

# Processing (separate function)
df = process_data(df)

# Visualization (separate function)
plot_results(df)
```

#### 4.2 Create Analysis Classes
**Example:**
```python
class BusSpeedAnalyzer:
    """Encapsulate bus speed analysis logic."""
    
    def __init__(self, segment_df, speed_df):
        self.segment_df = segment_df
        self.speed_df = speed_df
        
    def calculate_weighted_averages(self):
        """Calculate weighted average speeds."""
        # Implementation
        
    def compare_years(self, year1, year2):
        """Compare speeds between two years."""
        # Implementation
        
    def plot_hourly_trends(self):
        """Plot hourly speed trends."""
        # Implementation
```

**Benefits:**
- Related functionality grouped together
- State management (data) separated from operations
- Easier to test and maintain

---

## 5. Performance Optimization

### Current Issues
- Multiple passes over same data
- Inefficient groupby operations
- Memory-intensive operations not optimized

### Recommendations

#### 5.1 Reduce Memory Usage
**Example:**
```python
def optimize_dtypes(df):
    """Optimize dataframe memory usage."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'object':
            # Try converting to category if few unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
    
    return df
```

#### 5.2 Combine Multiple Operations
**Current Pattern:**
```python
df = df[df["year"] == 2025]
df = df[df["month"] == 8]
df = df[df["hour_of_day"].isin([2, 8, 14, 20])]
```

**Optimized Pattern:**
```python
mask = (df["year"] == 2025) & (df["month"] == 8) & (df["hour_of_day"].isin([2, 8, 14, 20]))
df = df[mask]
```

#### 5.3 Use Vectorized Operations
**Current Pattern:**
```python
df["weekend"] = df["day_of_week"].apply(lambda x: 1 if x in ["Saturday", "Sunday"] else 0)
```

**Optimized Pattern:**
```python
df["weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
```

---

## 6. Plotting and Visualization

### Current Issues
- Repetitive plotting code
- Hard to maintain consistent styling
- Plot parameters not easily adjustable

### Recommendations

#### 6.1 Create Plot Template Functions
**Example:**
```python
def create_comparison_plot(data_dict, title, ylabel, figsize=(14, 5)):
    """Create standardized comparison plot for weekday/weekend."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    # Weekday
    data_dict['weekday'].plot(ax=axes[0])
    axes[0].set_title(f"Weekday: {title}")
    axes[0].set_ylabel(ylabel)
    axes[0].axvspan(5, 20, color="lightblue", alpha=0.3, label="Cong. Pricing")
    axes[0].legend()
    
    # Weekend
    data_dict['weekend'].plot(ax=axes[1])
    axes[1].set_title(f"Weekend: {title}")
    axes[1].set_ylabel(ylabel)
    axes[1].axvspan(9, 20, color="lightblue", alpha=0.3, label="Cong. Pricing")
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

#### 6.2 Create Style Configuration
**Example:**
```python
# In config.py or separate style file
PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
}

# Apply in plotting code
plt.rcParams.update(PLOT_STYLE)
```

---

## 7. Documentation

### Current Issues
- Limited function documentation
- No docstrings for complex operations
- Assumptions not documented

### Recommendations

#### 7.1 Add Comprehensive Docstrings
**Example:**
```python
def interpolate_missing_bus_stops(speed_data, stop_data, month_filter=None, 
                                   save_to_csv=False, output_filename="extended_list.csv"):
    """
    Interpolate missing bus stop segments between timepoints.
    
    This function fills gaps in stop sequences by using stop reference data
    to add intermediate stops. Speed data from the parent segment is inherited.
    
    Parameters
    ----------
    speed_data : pd.DataFrame
        Bus speed data with columns: year, month, route_id, stop_order, etc.
        Must include timepoint information and speed metrics.
    stop_data : pd.DataFrame
        Reference data with all stops including: route_id, stop_order, 
        direction_id, coordinates, etc.
    month_filter : int, optional
        Filter to specific month (1-12). Default None processes all months.
    save_to_csv : bool, default False
        Whether to save results to CSV file.
    output_filename : str, default "extended_list.csv"
        Filename for saved CSV if save_to_csv=True.
    
    Returns
    -------
    pd.DataFrame
        Complete dataset with interpolated stops, no gaps in sequences.
        
    Examples
    --------
    >>> extended_df = interpolate_missing_bus_stops(
    ...     speed_data=segment_df, 
    ...     stop_data=stops_df,
    ...     month_filter=8
    ... )
    
    Notes
    -----
    - Interpolated stops have NULL values for road_distance and travel_time
    - Speed values are inherited from parent segment
    - Assumes stops are ordered sequentially
    
    See Also
    --------
    clean_stop_data : Prepare stop data for joining
    """
    # Implementation
```

#### 7.2 Add Inline Comments for Complex Logic
**Example:**
```python
# Calculate weighted average speed per route
# Weight by both distance traveled and number of trips to account for 
# route frequency and segment importance
segment_speed_df["weight_distance"] = (
    segment_speed_df["road_distance"] * segment_speed_df["bus_trip_count"]
)
segment_speed_df["weighted_avg_speed"] = (
    segment_speed_df["average_road_speed"] * segment_speed_df["weight_distance"]
)
```

---

## 8. Testing

### Current Issues
- No automated tests
- Manual verification only
- Changes could break existing functionality

### Recommendations

#### 8.1 Create Unit Tests
**Example structure:**
```python
# tests/test_features.py
import pytest
import pandas as pd
from src.features.build_features import clean_ridership

def test_clean_ridership_removes_sum_ridership_column():
    """Test that sum_ridership column is removed."""
    df = pd.DataFrame({
        'sum_ridership': [100, 200],
        'transit_timestamp': ['2025-01-01', '2025-01-02']
    })
    result = clean_ridership(df)
    assert 'sum_ridership' not in result.columns
    assert 'total_ridership' in result.columns

def test_clean_ridership_parses_dates():
    """Test that temporal features are added."""
    df = pd.DataFrame({
        'total_ridership': [100, 200],
        'transit_timestamp': ['2025-01-01', '2025-01-02']
    })
    result = clean_ridership(df)
    assert 'year' in result.columns
    assert 'month' in result.columns
    assert 'hour' in result.columns
```

#### 8.2 Create Integration Tests
**Example:**
```python
# tests/test_pipeline.py
def test_full_pipeline_runs():
    """Test that full pipeline executes without errors."""
    # Test with small sample data
    sample_data = create_sample_data()
    
    # Run through each step
    processed = process_data(sample_data)
    assert processed is not None
    
    # Verify output structure
    assert 'segment_speed_df' in processed
    assert len(processed['segment_speed_df']) > 0
```

---

## 9. Logging

### Current Issues
- Only print statements for tracking
- No log files for debugging
- Difficult to trace issues in production

### Recommendations

#### 9.1 Implement Proper Logging
**Example:**
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bus_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def fetch_data(app_token):
    """Fetch data with proper logging."""
    logger.info("Starting data fetch process")
    
    try:
        data = fetch_all_data(app_token)
        logger.info(f"Successfully fetched {len(data)} datasets")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        raise
```

---

## 10. Specific Code Improvements

### 10.1 Simplify Direction Mapping
**Current:**
```python
segment_speed_df["direction_id"] = segment_speed_df["direction"].apply(
    lambda x: 0 if x in ["N", "E"] else 1
)
```

**Suggested:**
```python
DIRECTION_MAP = {'N': 0, 'E': 0, 'S': 1, 'W': 1}
segment_speed_df["direction_id"] = segment_speed_df["direction"].map(DIRECTION_MAP)
```

### 10.2 Avoid Chain Indexing Warnings
**Current:**
```python
vehicles_entering_manhattan.loc[:, "inflow_cat"] = ...
```

**Suggested:**
```python
vehicles_entering_manhattan = vehicles_entering_manhattan.copy()
vehicles_entering_manhattan["inflow_cat"] = ...
```

### 10.3 Use Query for Complex Filters
**Current:**
```python
visualization_data = segment_speed_df[
    ((segment_speed_df["year"] == 2024) | (segment_speed_df["year"] == 2025)) &
    (segment_speed_df["month"] == 8) &
    (segment_speed_df["hour_of_day"].isin([2, 8, 14, 20]))
]
```

**Suggested:**
```python
visualization_data = segment_speed_df.query(
    "year in [2024, 2025] and month == 8 and hour_of_day in [2, 8, 14, 20]"
)
```

---

## 11. Future Enhancements

### 11.1 Add Data Caching
Consider using `joblib` or `pickle` to cache expensive operations:
```python
from joblib import Memory

memory = Memory("cachedir", verbose=0)

@memory.cache
def expensive_computation(data):
    # Your expensive computation here
    pass
```

### 11.2 Parallelize API Calls
Use `concurrent.futures` for parallel API requests:
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(fetch_dataset, dataset_id) for dataset_id in dataset_ids]
    results = [f.result() for f in futures]
```

### 11.3 Add Progress Bars
Use `tqdm` for long-running operations:
```python
from tqdm import tqdm

for route in tqdm(routes, desc="Processing routes"):
    process_route(route)
```

### 11.4 Create a Dashboard
Consider using Streamlit or Dash to create an interactive dashboard:
```python
import streamlit as st

st.title("NYC Bus Speed Analysis")
year = st.selectbox("Select Year", [2023, 2024, 2025])
# ... interactive visualizations
```

---

## Priority Recommendations

### High Priority (Implement First)
1. ✅ **Function extraction** - Already done in pipeline
2. **Add input validation** - Prevents runtime errors
3. **Implement logging** - Essential for debugging
4. **Add docstrings** - Improves code understanding

### Medium Priority
5. **Create unit tests** - Ensures code reliability
6. **Optimize performance** - Use vectorized operations
7. **Error handling** - Add try-except blocks
8. **Data quality checks** - Validate data integrity

### Low Priority (Nice to Have)
9. **Create analysis classes** - Better organization
10. **Add caching** - Performance optimization
11. **Build dashboard** - User-friendly interface
12. **Parallelize operations** - Speed up execution

---

## Conclusion

Your code has been successfully transformed into a production-ready data science pipeline following cookiecutter standards. The suggestions above are recommendations for future improvements to make the code even more robust, maintainable, and efficient.

The current structure already implements many best practices:
- ✅ Modular code organization
- ✅ Separation of concerns
- ✅ Configuration management
- ✅ Reusable functions
- ✅ Clear documentation structure

Focus on implementing the high-priority recommendations first, then gradually work through medium and low priority items as time permits.
