# NYC Bus Analysis Project

A data science project analyzing NYC MTA bus performance, congestion pricing impacts, and traffic patterns in Manhattan.

## Project Structure

```
Bus Project Data Science/
├── README.md                   # This file
├── SUGGESTIONS.md             # Code improvement recommendations
├── requirements.txt           # Python dependencies
├── main.py                    # Main pipeline orchestration script
│
├── data/                      # Data directory
│   ├── raw/                   # Original, immutable data
│   ├── processed/             # Cleaned, transformed data
│   └── interim/               # Intermediate processing results
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── New_Bus_Analysis.ipynb # Original analysis notebook
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── config.py             # Configuration and constants
│   │
│   ├── data/                 # Data fetching and loading
│   │   ├── __init__.py
│   │   └── make_dataset.py   # API calls and data ingestion
│   │
│   ├── features/             # Feature engineering and data processing
│   │   ├── __init__.py
│   │   └── build_features.py # Data cleaning and transformations
│   │
│   ├── models/               # Model training and evaluation
│   │   ├── __init__.py
│   │   └── train_model.py    # Regression models
│   │
│   └── visualization/        # Plotting and visualization
│       ├── __init__.py
│       └── visualize.py      # All plotting functions
│
└── reports/                  # Generated analysis outputs
    ├── figures/              # Generated plots and charts
    ├── model_metrics.csv     # Model performance metrics
    └── feature_importance.csv # Model coefficients
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file or export your Socrata API token:

```bash
export APP_TOKEN='your_socrata_api_token'
```

You can get a Socrata API token from: https://data.ny.gov/

### 3. Ensure Data Directory Structure

The pipeline will automatically create necessary directories, but your raw data should be in:
- `data/raw/` (or `Data/` for existing files)

## Usage

### Running the Complete Pipeline

```bash
# Run the full pipeline (fetch, process, visualize, model)
python main.py --all

# Or run individual steps:
python main.py --fetch-data    # Fetch data from API
python main.py --process       # Process and clean data
python main.py --visualize     # Generate visualizations
python main.py --model         # Train models
```

### Using Individual Modules

You can also import and use individual modules in your own scripts or notebooks:

```python
# Data ingestion
from src.data.make_dataset import fetch_all_data

# Feature engineering
from src.features.build_features import clean_segment_speed_data

# Visualization
from src.visualization.visualize import plot_speed_comparison_by_year

# Modeling
from src.models.train_model import train_linear_regression
```

## Data Sources

All data comes from the NYC Open Data portal (data.ny.gov):

1. **Bus Segment Speeds** - Speed data for specific route segments
2. **Bus Speeds** - Overall bus route speeds
3. **Hourly Ridership** - Passenger counts by hour
4. **Bridge/Tunnel Crossings** - Vehicle counts entering Manhattan
5. **Congestion Relief Zone (CRZ) Entries** - Vehicles entering CBD
6. **CBD Vehicle Speeds** - Average speeds in Central Business District
7. **CBD Bus Routes** - Bus routes operating in CBD
8. **Stop Data** - Bus stop locations and metadata
9. **CBD Geofence** - Geographic boundary of CBD

## Analysis Components

### 1. Data Ingestion (`src/data/make_dataset.py`)
- Fetches data from Socrata API
- Implements caching to avoid redundant API calls
- Handles pagination for large datasets
- Loads local files when available

### 2. Feature Engineering (`src/features/build_features.py`)
- Cleans and standardizes column names
- Parses timestamps and creates temporal features
- Calculates weighted averages for speed metrics
- Interpolates missing bus stops
- Filters data by geographic boundaries (CBD)

### 3. Visualization (`src/visualization/visualize.py`)
- Speed trends by hour, day, and bus type
- Year-over-year comparisons (2023-2025)
- CBD-specific analysis
- Vehicle crossing patterns
- Ridership trends
- Interactive Folium maps

### 4. Modeling (`src/models/train_model.py`)
- Linear regression for speed prediction
- Feature importance analysis
- Model evaluation metrics
- Statsmodels OLS for statistical inference

## Key Findings (From Original Notebook)

- Analysis of Manhattan bus speeds from 2023-2025
- Impact of congestion pricing on bus speeds
- Comparison of weekday vs weekend patterns
- CBD-specific speed improvements
- Correlation between ridership and vehicle entries

## Configuration

All configuration parameters are centralized in `src/config.py`:
- Dataset IDs and API endpoints
- File paths and naming patterns
- Analysis parameters (date ranges, filters)
- Visualization settings
- Model hyperparameters

## Code Quality

Your original code has been preserved and refactored into a production-ready pipeline:
- ✅ Modular structure with clear separation of concerns
- ✅ Reusable functions with consistent interfaces
- ✅ Comprehensive docstrings
- ✅ Centralized configuration
- ✅ Type hints where applicable

See `SUGGESTIONS.md` for detailed recommendations on further improvements.

## Output

### Processed Data
- `data/processed/segment_speed_processed.csv`
- `data/processed/ridership_processed.csv`
- `data/processed/manhattan_crossings_processed.csv`
- `data/processed/cbd_entries_processed.csv`

### Visualizations
- `reports/figures/*.png` - All generated plots

### Model Results
- `reports/linear_regression_model.pkl` - Trained model
- `reports/model_metrics.csv` - Performance metrics
- `reports/feature_importance.csv` - Feature coefficients

## Development

### Running Tests (Future)
```bash
pytest tests/
```

### Linting (Future)
```bash
flake8 src/
black src/
```

### Adding New Features
1. Add new functions to appropriate module in `src/`
2. Update `config.py` if new parameters needed
3. Update `main.py` if pipeline integration required
4. Add tests in `tests/`

## Contributing

When making changes:
1. Follow the existing code structure
2. Add docstrings to all functions
3. Update this README if adding major features
4. Consider the suggestions in `SUGGESTIONS.md`

## License

[Your License Here]

## Contact

[Your Contact Information]

---

**Note**: This project structure follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template, a standardized approach to organizing data science projects.
