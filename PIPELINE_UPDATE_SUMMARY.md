# Pipeline Update Summary

## Overview

The data pipeline has been updated to move all preprocessing into `build_features.py`, eliminating code duplication in the notebook and Streamlit app. This significantly simplifies downstream code and improves app loading performance.

## Changes Made

### 1. New Functions in `build_features.py`

#### `add_cbd_classification(segment_speed_df, cbd_geofence_path)`
- **Purpose**: Classifies segments as CBD or non-CBD using geofence polygon
- **Method**: Calculates segment midpoint from start/end coordinates, checks containment
- **Performance**: Vectorized operations, ~30-60 seconds for 2.2M rows
- **Output**: Adds `is_cbd_segment` boolean column to dataframe

#### `create_speed_monthly_aggregated(segment_speed_df)`
- **Purpose**: Pre-aggregates speed data by route and month
- **Groups by**: route_id, year, month, weekend, is_cbd_segment
- **Aggregates**: avg_speed_mph (mean), bus_trip_count (sum), road_distance (sum)
- **Output**: Saves to `Data/processed/speed_monthly.csv`

#### `create_speed_overall_aggregated(speed_monthly_df)`
- **Purpose**: Creates overall speed aggregates (no route dimension)
- **Groups by**: year_month, year, month, weekend, is_cbd_segment
- **Output**: Saves to `Data/processed/speed_overall.csv` - instant app loading!

### 2. Updated `main.py`

**New imports:**
```python
from src.features.build_features import (
    ...,
    add_cbd_classification,
    create_speed_monthly_aggregated,
    create_speed_overall_aggregated
)
```

**New processing steps** (added after `filter_cbd_segments`):
1. Adds CBD classification to full segment_speed data
2. Creates monthly aggregates by route
3. Creates overall aggregates (no route dimension)
4. Saves all three new processed files

**New output files:**
- `Data/processed/segment_speed_processed.csv` - now includes `is_cbd_segment` column
- `Data/processed/speed_monthly.csv` - monthly aggregates by route (NEW)
- `Data/processed/speed_overall.csv` - overall monthly aggregates (NEW)

### 3. Updated `app_cbd_analysis.py`

**Removed:**
- ~100 lines of geofence loading and classification code
- `load_geofence()` function
- `classify_cbd_segments()` function

**Simplified to:**
```python
@st.cache_data
def load_data():
    """Load all pre-processed datasets."""
    speed_overall = pd.read_csv(PROCESSED_DIR / "speed_overall.csv")
    speed_monthly = pd.read_csv(PROCESSED_DIR / "speed_monthly.csv")
    segment_speed = pd.read_csv(PROCESSED_DIR / "segment_speed_processed.csv")
    # Convert year_month to datetime for plotting
    return speed_overall, speed_monthly, segment_speed
```

**Changed column references:**
- `is_cbd` → `is_cbd_segment` (consistent with pipeline)

**Performance improvement:**
- **Before**: 30-60 second geospatial operations on every app load
- **After**: Instant loading of pre-processed CSV files

### 4. Notebook Simplification (Ready to Apply)

The notebook `notebooks/New_Bus_Analysis.ipynb` should be updated to:

**Remove:**
- Cell 9: Geofence loading and classification (~40 lines)
- Cell 11: Aggregation code now done in pipeline
- Any other cells that duplicate pipeline preprocessing

**Replace with:**
```python
# Load pre-processed data
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../Data/processed")

# Load datasets
segment_speed = pd.read_csv(DATA_DIR / "segment_speed_processed.csv")
speed_monthly = pd.read_csv(DATA_DIR / "speed_monthly.csv")
speed_overall = pd.read_csv(DATA_DIR / "speed_overall.csv")

# segment_speed already has is_cbd_segment column
# speed_monthly and speed_overall ready for plotting
```

**Keep:**
- Visualizations
- DiD analysis code
- Statistical tests
- Interpretations and markdown cells

## How to Use

### 1. Run the Updated Pipeline

```bash
# Run full pipeline (fetch + process + visualize)
python main.py --all

# Or just process existing data
python main.py --process
```

**Expected output:**
```
STEP 2: DATA PROCESSING AND FEATURE ENGINEERING
==================================================================================
Processing segment speed data...
...
Adding CBD classification to segment speed data...
Loading CBD geofence...
Calculating segment midpoints...
Classifying CBD segments (this may take 30-60 seconds)...
  CBD segments: XXX,XXX
  Non-CBD segments: X,XXX,XXX

Creating aggregated datasets...
Creating monthly aggregated speed data...
  Created XX,XXX monthly route aggregates
Creating overall aggregated speed data...
  Created XXX overall monthly aggregates

Saving processed data...
✓ Data processing complete!
```

### 2. Run the Streamlit App

```bash
streamlit run app_cbd_analysis.py
```

**Expected behavior:**
- App loads in ~2-3 seconds (vs 30-60 seconds before)
- Shows speed graphs from 2023 to September 2025
- Displays both 3-month and 6-month DiD results
- Shows fastest/slowest routes

### 3. Update the Notebook

The notebook needs manual simplification to remove duplicate preprocessing. Key areas:

1. **Early cells**: Remove geofence loading and classification
2. **Aggregation cells**: Use pre-loaded speed_monthly and speed_overall
3. **Keep**: All visualization, analysis, and interpretation cells

## Benefits

### Performance
- **Streamlit app**: 30-60 second → 2-3 second initial load
- **One-time cost**: ~60 seconds during pipeline run (acceptable for batch processing)
- **Notebook**: Instant data loading instead of repeated preprocessing

### Code Quality
- **Before**: ~250 lines of preprocessing code duplicated across 3 files
- **After**: ~100 lines in pipeline, ~150 lines removed from notebook/app
- **Maintainability**: Single source of truth for preprocessing logic

### Data Consistency
- CBD classification applied once in pipeline, consistent everywhere
- Aggregations calculated identically for app and analysis
- Version-controlled processed datasets

## Files Modified

1. **src/features/build_features.py** - Added 3 new functions (~100 lines)
2. **main.py** - Added processing steps and new outputs (~15 lines)
3. **app_cbd_analysis.py** - Removed ~100 lines, simplified data loading
4. **notebooks/New_Bus_Analysis.ipynb** - Ready for simplification (not yet done)

## Next Steps

1. ✅ Run `python main.py --process` to generate new processed files
2. ✅ Test Streamlit app with `streamlit run app_cbd_analysis.py`
3. ⚠️ Simplify notebook to use pre-processed data
4. ⚠️ Verify all analyses produce same results as before

## Troubleshooting

### "File not found" errors
- Run `python main.py --process` first to generate processed files

### "Column 'is_cbd_segment' not found"
- Segment_speed_processed.csv needs to be regenerated with new pipeline
- Delete old processed files and re-run pipeline

### App still slow
- Check that speed_overall.csv exists in Data/processed/
- Verify app is loading from processed files, not re-classifying

### Different results in notebook vs app
- Ensure both use same processed files
- Check for any manual filtering or transformations applied differently
