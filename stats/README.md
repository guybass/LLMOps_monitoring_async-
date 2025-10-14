# Download Statistics

This directory contains automated download statistics from PyPI for llamonitor-async.

## Files

- `download_summary.json` - Latest summary statistics (updated daily)
- `downloads_YYYY-MM-DD.json` - Full daily statistics snapshots
- `downloads_YYYY-MM-DD.csv` - CSV format for analysis

## Automated Collection

Statistics are automatically collected daily via GitHub Actions.
See `.github/workflows/collect_download_stats.yml` for details.

## Manual Collection

To manually collect statistics:

```bash
pip install pypistats pandas
python scripts/fetch_download_stats.py --days 30
```

## Visualization

You can visualize this data using:
- Excel/Google Sheets (import CSV files)
- Python pandas/matplotlib
- Grafana (import JSON files)
- Any BI tool
