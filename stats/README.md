# Download Statistics

This directory contains automated download statistics from PyPI for llamonitor-async.

## Files

- `download_summary.json` - Latest summary statistics (updated daily)
- `downloads_YYYY-MM-DD.json` - Full daily statistics snapshots
- `downloads_YYYY-MM-DD.csv` - CSV format for analysis

## Automated Collection

Statistics are automatically collected daily via GitHub Actions at 00:00 UTC.

See `.github/workflows/collect_download_stats.yml` for details.

## Manual Collection

To manually collect statistics:

```bash
# Install dependencies
pip install pypistats pandas

# Run collection script
python scripts/fetch_download_stats.py --days 30
```

## Data Structure

### Summary JSON (`download_summary.json`)

```json
{
  "package": "llamonitor-async",
  "timestamp": "2025-01-15T12:00:00",
  "total_downloads": 1234,
  "average_daily": 41.1,
  "top_versions": [
    {"version": "3.10", "downloads": 500},
    {"version": "3.11", "downloads": 450}
  ],
  "top_systems": [
    {"system": "Linux", "downloads": 600},
    {"system": "Windows", "downloads": 400}
  ]
}
```

### Full Statistics JSON (`downloads_YYYY-MM-DD.json`)

Contains complete data from pypistats including:
- Overall statistics (month, week)
- Recent daily downloads
- Python version breakdown
- Operating system breakdown

### CSV Format (`downloads_YYYY-MM-DD.csv`)

Daily time series data with columns:
- `date` - Download date
- `downloads` - Number of downloads
- `category` - Breakdown category (if applicable)

## Visualization

### Using Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('stats/downloads_2025-01-15.csv')

# Plot time series
df.plot(x='date', y='downloads', kind='line')
plt.title('llamonitor-async Daily Downloads')
plt.show()
```

### Using Excel/Google Sheets

1. Import any CSV file
2. Create charts from the data
3. Use pivot tables for analysis

### Using Grafana

1. Import JSON files as a data source
2. Create custom dashboards
3. Set up alerts for download thresholds

## Badges

The package README displays real-time download badges from pepy.tech:

- Total downloads
- Monthly downloads
- Weekly downloads

These badges automatically update without needing this data collection.

## Data Retention

- Daily snapshots are kept indefinitely in git history
- The latest summary is always available at `download_summary.json`
- Individual CSV/JSON files can be archived or deleted as needed

## Privacy

All statistics come from PyPI's public API and contain no personally identifiable information. Only aggregate download counts are tracked.
