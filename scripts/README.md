# Scripts

This directory contains utility scripts for llamonitor-async package maintenance and analytics.

## Available Scripts

### `fetch_download_stats.py`

Fetches download statistics from PyPI using the pypistats API.

**Installation:**
```bash
pip install pypistats pandas requests
```

**Usage:**
```bash
# Fetch last 30 days of statistics
python scripts/fetch_download_stats.py

# Fetch last 90 days
python scripts/fetch_download_stats.py --days 90

# Custom output directory
python scripts/fetch_download_stats.py --output-dir ./my_stats

# Suppress terminal output
python scripts/fetch_download_stats.py --no-display
```

**Options:**
- `--output-dir DIR` - Directory to save statistics (default: `./stats`)
- `--days N` - Number of days to fetch (1-180, default: 30)
- `--package NAME` - Package name (default: `llamonitor-async`)
- `--no-display` - Don't display summary in terminal

**Output Files:**
- `downloads_YYYY-MM-DD.json` - Full statistics snapshot
- `downloads_YYYY-MM-DD.csv` - CSV format for analysis
- `download_summary.json` - Summary statistics

**Collected Metrics:**
- Total downloads (overall, monthly, weekly, daily)
- Downloads by Python version
- Downloads by operating system
- Time series data for trend analysis

## Automated Collection

Download statistics are automatically collected daily via GitHub Actions.

See `.github/workflows/collect_download_stats.yml` for the automation configuration.

## Adding New Scripts

When adding new scripts:

1. Add a shebang line: `#!/usr/bin/env python3`
2. Make it executable: `chmod +x scripts/your_script.py`
3. Add proper argument parsing with `argparse`
4. Include docstrings and help text
5. Update this README with usage instructions
6. Add any new dependencies to `pyproject.toml`

## Dependencies

Scripts may have additional dependencies beyond the main package:

- `pypistats` - PyPI statistics API client
- `pandas` - Data analysis (optional, for CSV export)
- `requests` - HTTP client

Install all script dependencies:
```bash
pip install -r scripts/requirements.txt  # If available
# Or individually:
pip install pypistats pandas requests
```
