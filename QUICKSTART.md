# 🚌 Quick Start Guide

## Launch the Streamlit Dashboard

### Option 1: Simple Launch
```bash
streamlit run app_cbd_analysis.py
```

### Option 2: With Custom Port
```bash
streamlit run app_cbd_analysis.py --server.port 8502
```

### Option 3: Open in Browser Automatically
```bash
streamlit run app_cbd_analysis.py --server.headless false
```

## What You'll See

### 📊 Page 1: Speed Trends
- **Weekday speeds** (top chart)
- **Weekend speeds** (bottom chart)
- Each chart shows:
  - Monthly data points (light colored)
  - 3-month rolling average (bold lines)
  - Blue shaded area = Congestion pricing active
  - Red = CBD segments, Blue = Non-CBD segments

### 📈 Page 2: DiD Results
Two columns showing:
- **Left**: 3-month comparison
- **Right**: 6-month comparison

Each displays:
- DiD Coefficient (speed change in mph)
- P-value (statistical significance)
- Number of observations

### 🏆 Page 3: Route Rankings
Two columns showing:
- **Left**: Last month's rankings
- **Right**: Same month last year

Each displays:
- 5 slowest CBD routes
- 5 fastest CBD routes

## Troubleshooting

### "FileNotFoundError: Data directory not found"
Run the data processing pipeline first:
```bash
python main.py --process
```

### "MemoryError" or Slow Loading
The segment speed data is large (50+ MB). First load takes 30-60 seconds.
After that, Streamlit's caching makes it fast.

### Port Already in Use
Try a different port:
```bash
streamlit run app_cbd_analysis.py --server.port 8503
```

### App Won't Open in Browser
Manually navigate to: `http://localhost:8501`

## Keyboard Shortcuts in App

- `R` - Rerun the app
- `C` - Clear cache
- `Ctrl+C` in terminal - Stop the app

## Tips for Best Experience

1. **Use a modern browser** (Chrome, Firefox, Safari, Edge)
2. **Expand your browser window** for better visualization
3. **Wait for "Running..." indicator** to disappear before interacting
4. **Refresh the page** if data seems stale

## Understanding the Results

### Positive DiD Coefficient
✅ Bus speeds INCREASED in CBD after pricing
- Example: +0.50 mph means buses got 0.5 mph faster

### Negative DiD Coefficient  
❌ Bus speeds DECREASED in CBD after pricing
- Example: -0.30 mph means buses got 0.3 mph slower

### P-value < 0.05
📊 Result is statistically significant
- We can be confident the effect is real, not random

### P-value ≥ 0.05
⚠️ Result is NOT statistically significant
- Effect could be due to random variation
