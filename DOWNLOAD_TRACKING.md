# Download Tracking System

llamonitor-async includes a comprehensive download tracking system to monitor package adoption and usage across the Python ecosystem.

## Overview

The system consists of three components:

1. **Real-time Badges** - Live download counts displayed in README
2. **Automated Collection** - Daily statistics collection via GitHub Actions
3. **Manual Tools** - Python scripts for on-demand analysis

## Real-Time Badges

The package README displays live download statistics using badges from [pepy.tech](https://pepy.tech):

```markdown
[![Downloads](https://static.pepy.tech/badge/llamonitor-async)](https://pepy.tech/project/llamonitor-async)
[![Downloads/Month](https://static.pepy.tech/badge/llamonitor-async/month)](https://pepy.tech/project/llamonitor-async)
[![Downloads/Week](https://static.pepy.tech/badge/llamonitor-async/week)](https://pepy.tech/project/llamonitor-async)
```

These badges update automatically without any action required.

## Automated Collection

### GitHub Actions Workflow

Statistics are automatically collected daily at 00:00 UTC and committed to the repository.

**Workflow File:** `.github/workflows/collect_download_stats.yml`

**What it does:**
1. Fetches latest statistics from PyPI
2. Saves data as JSON and CSV files
3. Commits changes to the `stats/` directory
4. Creates a workflow summary

**Manual Trigger:**

You can manually trigger the workflow from GitHub:

1. Go to **Actions** tab
2. Select **Collect Download Statistics**
3. Click **Run workflow**
4. Optionally specify number of days to fetch (1-180)

### Collected Metrics

The automated system collects:

- **Overall Downloads**
  - Total all-time downloads
  - Monthly downloads
  - Weekly downloads
  - Daily downloads

- **Python Versions**
  - Downloads by Python 3.8, 3.9, 3.10, 3.11, 3.12+
  - Version adoption trends

- **Operating Systems**
  - Downloads by Linux, Windows, macOS
  - Platform distribution

- **Time Series Data**
  - Daily download counts
  - Trend analysis over time

## Manual Collection

### Installation

Install required dependencies:

```bash
pip install pypistats pandas requests
```

### Basic Usage

```bash
# Fetch last 30 days
python scripts/fetch_download_stats.py

# Fetch last 90 days
python scripts/fetch_download_stats.py --days 90

# Custom output directory
python scripts/fetch_download_stats.py --output-dir ./my_stats

# Suppress display
python scripts/fetch_download_stats.py --no-display
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir DIR` | Output directory for statistics | `./stats` |
| `--days N` | Number of days to fetch (1-180) | `30` |
| `--package NAME` | Package name to track | `llamonitor-async` |
| `--no-display` | Suppress terminal output | `False` |

### Output Files

The script generates three types of files:

1. **`downloads_YYYY-MM-DD.json`**
   - Complete statistics snapshot
   - All metrics in structured format
   - Suitable for programmatic analysis

2. **`downloads_YYYY-MM-DD.csv`**
   - Time series data in CSV format
   - Easy to import into Excel/Google Sheets
   - Compatible with data visualization tools

3. **`download_summary.json`**
   - Latest summary statistics
   - Key metrics at a glance
   - Updated with each run

## Data Analysis

### Using Python

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load summary
with open('stats/download_summary.json') as f:
    summary = json.load(f)

print(f"Total Downloads: {summary['total_downloads']:,}")
print(f"Average Daily: {summary['average_daily']:.1f}")

# Load time series
df = pd.read_csv('stats/downloads_2025-01-15.csv')
df['date'] = pd.to_datetime(df['date'])

# Plot trend
df.plot(x='date', y='downloads', kind='line', figsize=(12, 6))
plt.title('llamonitor-async Download Trend')
plt.xlabel('Date')
plt.ylabel('Downloads')
plt.grid(True)
plt.show()

# Calculate growth rate
df['rolling_avg'] = df['downloads'].rolling(window=7).mean()
df.plot(x='date', y=['downloads', 'rolling_avg'], figsize=(12, 6))
plt.title('Daily Downloads with 7-Day Moving Average')
plt.legend(['Daily', '7-Day Average'])
plt.show()
```

### Using Pandas

```python
import pandas as pd

# Load data
df = pd.read_csv('stats/downloads_2025-01-15.csv')

# Summary statistics
print(df['downloads'].describe())

# Find peak day
peak = df.loc[df['downloads'].idxmax()]
print(f"Peak day: {peak['date']} with {peak['downloads']} downloads")

# Calculate cumulative downloads
df['cumulative'] = df['downloads'].cumsum()
print(f"Total downloads: {df['cumulative'].iloc[-1]:,}")
```

### Using Excel/Google Sheets

1. Open Excel or Google Sheets
2. Import any CSV file from `stats/`
3. Create pivot tables for analysis
4. Build charts and dashboards
5. Export reports

### Using Grafana

1. Set up Grafana with JSON data source plugin
2. Import `stats/downloads_*.json` files
3. Create custom dashboards
4. Set up alerts for download thresholds
5. Share dashboards with team

## API Access

### PyPI Stats API

Direct API access without using the script:

```python
import pypistats

# Get overall stats
overall = pypistats.overall("llamonitor-async", period="month")
print(overall)

# Get recent stats
recent = pypistats.recent("llamonitor-async", period="day")
print(recent)

# Get Python version breakdown
python_versions = pypistats.python_major("llamonitor-async")
print(python_versions)

# Get system breakdown
systems = pypistats.system("llamonitor-async")
print(systems)
```

### Pepy.tech API

```python
import requests

# Get download stats from pepy.tech
response = requests.get('https://api.pepy.tech/api/v2/projects/llamonitor-async')
data = response.json()

print(f"Total downloads: {data['total_downloads']:,}")
print(f"Recent downloads: {data['downloads']:,}")
```

## Data Structure

### Summary JSON Schema

```json
{
  "package": "string",
  "timestamp": "ISO 8601 datetime",
  "total_downloads": "integer",
  "average_daily": "float",
  "top_versions": [
    {
      "version": "string",
      "downloads": "integer"
    }
  ],
  "top_systems": [
    {
      "system": "string",
      "downloads": "integer"
    }
  ]
}
```

### Full Statistics Schema

```json
{
  "package": "string",
  "timestamp": "ISO 8601 datetime",
  "overall_month": {
    "data": [
      {
        "category": "string",
        "downloads": "integer"
      }
    ]
  },
  "recent": {
    "data": [
      {
        "date": "YYYY-MM-DD",
        "downloads": "integer"
      }
    ]
  },
  "python_versions": {
    "data": [
      {
        "category": "string",
        "downloads": "integer"
      }
    ]
  },
  "systems": {
    "data": [
      {
        "category": "string",
        "downloads": "integer"
      }
    ]
  }
}
```

## Use Cases

### 1. Monitor Package Adoption

Track how quickly the package is being adopted after releases:

```python
# Compare download growth across versions
df = pd.read_csv('stats/downloads_2025-01-15.csv')
df['week'] = pd.to_datetime(df['date']).dt.to_period('W')
weekly = df.groupby('week')['downloads'].sum()
growth_rate = weekly.pct_change()
print(f"Average weekly growth: {growth_rate.mean():.1%}")
```

### 2. Identify Popular Platforms

Understand which platforms your users prefer:

```python
with open('stats/download_summary.json') as f:
    summary = json.load(f)

# Show platform distribution
for system in summary['top_systems']:
    percentage = system['downloads'] / summary['total_downloads'] * 100
    print(f"{system['system']}: {percentage:.1f}%")
```

### 3. Plan Python Version Support

Make data-driven decisions about which Python versions to support:

```python
# Analyze Python version usage
for version in summary['top_versions']:
    percentage = version['downloads'] / summary['total_downloads'] * 100
    print(f"Python {version['version']}: {percentage:.1f}%")
```

### 4. Measure Release Impact

Compare downloads before and after a release:

```python
release_date = '2025-01-10'
before = df[df['date'] < release_date]['downloads'].mean()
after = df[df['date'] >= release_date]['downloads'].mean()
impact = (after - before) / before * 100
print(f"Download increase after release: {impact:.1f}%")
```

### 5. Generate Reports

Create automated reports for stakeholders:

```python
def generate_report(summary):
    report = f"""
    # llamonitor-async Download Report

    Generated: {summary['timestamp']}

    ## Key Metrics
    - Total Downloads: {summary['total_downloads']:,}
    - Average Daily: {summary['average_daily']:.1f}

    ## Top Python Versions
    """
    for v in summary['top_versions'][:3]:
        report += f"\n- Python {v['version']}: {v['downloads']:,}"

    return report

with open('stats/download_summary.json') as f:
    print(generate_report(json.load(f)))
```

## Best Practices

### 1. Regular Monitoring

- Check stats weekly to identify trends
- Set up alerts for unusual patterns
- Review after each release

### 2. Data Retention

- Keep daily snapshots for trend analysis
- Archive old data periodically
- Back up critical statistics

### 3. Privacy Compliance

- Only collect aggregate statistics
- No personally identifiable information
- Comply with PyPI's terms of service

### 4. Automation

- Let GitHub Actions handle daily collection
- Set up notifications for milestones
- Integrate with project management tools

### 5. Sharing

- Include stats in project updates
- Share milestones with community
- Use data for fundraising/sponsorship

## Troubleshooting

### Issue: Script fails with "pypistats not installed"

**Solution:**
```bash
pip install pypistats pandas requests
```

### Issue: GitHub Action fails to commit

**Possible causes:**
- Missing write permissions
- No changes to commit
- Git configuration issues

**Solution:**
Check workflow permissions in repository settings.

### Issue: No data for recent dates

**Reason:** PyPI statistics have a delay of 24-48 hours.

**Solution:** Request data from 2-3 days ago.

### Issue: Rate limiting errors

**Reason:** PyPI API has rate limits.

**Solution:**
- Don't run script too frequently
- Use GitHub Actions schedule
- Cache results locally

## Resources

### Official APIs

- [PyPI Stats](https://pypistats.org/) - Official PyPI download statistics
- [Pepy.tech](https://pepy.tech/) - Alternative statistics provider
- [PyPI API](https://warehouse.pypa.io/api-reference/) - PyPI REST API

### Documentation

- [pypistats package](https://github.com/hugovk/pypistats) - Python client for PyPI stats
- [GitHub Actions](https://docs.github.com/en/actions) - Workflow automation
- [pandas](https://pandas.pydata.org/) - Data analysis library

### Community

- Share insights in GitHub Discussions
- Report issues with tracking system
- Contribute improvements to scripts

## Future Enhancements

Planned improvements to the tracking system:

- [ ] Real-time dashboard web interface
- [ ] Automated weekly/monthly reports via email
- [ ] Integration with Discord/Slack notifications
- [ ] Comparative analysis with similar packages
- [ ] Download forecast models
- [ ] Geographic distribution tracking
- [ ] Retention and churn analysis
- [ ] Automated social media updates for milestones

## Contributing

Contributions to improve the download tracking system are welcome!

Areas for contribution:
1. Enhanced visualization scripts
2. Additional analysis tools
3. Integration with BI platforms
4. Improved documentation
5. New tracking metrics

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Questions or suggestions?** Open an issue or discussion on GitHub!
