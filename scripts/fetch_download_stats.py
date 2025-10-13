#!/usr/bin/env python3
"""
Download Statistics Fetcher for llamonitor-async

Fetches download statistics from PyPI using the pypistats API and saves them
as JSON and CSV for tracking and visualization.

Usage:
    python scripts/fetch_download_stats.py
    python scripts/fetch_download_stats.py --output-dir ./custom_stats
    python scripts/fetch_download_stats.py --days 180

Requirements:
    pip install pypistats requests
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import pypistats
except ImportError:
    print("Error: pypistats not installed. Run: pip install pypistats")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Warning: pandas not installed. CSV export will be limited.")
    pd = None


PACKAGE_NAME = "llamonitor-async"


class DownloadStatsCollector:
    """Collects and stores PyPI download statistics."""

    def __init__(self, package_name: str = PACKAGE_NAME, output_dir: str = "./stats"):
        self.package_name = package_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.utcnow().isoformat()

    def fetch_overall_stats(self, period: str = "month") -> Optional[Dict[str, Any]]:
        """
        Fetch overall download statistics.

        Args:
            period: 'day', 'week', or 'month'

        Returns:
            Dictionary with download statistics or None on error
        """
        try:
            print(f"Fetching {period} statistics for {self.package_name}...")
            data = pypistats.overall(self.package_name, period=period, format="json")
            return json.loads(data) if isinstance(data, str) else data
        except Exception as e:
            print(f"Error fetching overall stats: {e}")
            return None

    def fetch_recent_stats(self, days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Fetch recent download statistics.

        Args:
            days: Number of days to fetch (max 180)

        Returns:
            Dictionary with download statistics or None on error
        """
        try:
            print(f"Fetching last {days} days of statistics...")
            data = pypistats.recent(self.package_name, period="day", format="json")
            result = json.loads(data) if isinstance(data, str) else data

            # Filter to requested days
            if result and "data" in result:
                cutoff_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                result["data"] = [
                    item for item in result["data"]
                    if item.get("date", "") >= cutoff_date
                ]

            return result
        except Exception as e:
            print(f"Error fetching recent stats: {e}")
            return None

    def fetch_python_versions(self) -> Optional[Dict[str, Any]]:
        """Fetch download statistics by Python version."""
        try:
            print("Fetching Python version statistics...")
            data = pypistats.python_major(self.package_name, format="json")
            return json.loads(data) if isinstance(data, str) else data
        except Exception as e:
            print(f"Error fetching Python version stats: {e}")
            return None

    def fetch_system_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch download statistics by operating system."""
        try:
            print("Fetching operating system statistics...")
            data = pypistats.system(self.package_name, format="json")
            return json.loads(data) if isinstance(data, str) else data
        except Exception as e:
            print(f"Error fetching system stats: {e}")
            return None

    def calculate_summary(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from raw data."""
        summary = {
            "package": self.package_name,
            "timestamp": self.timestamp,
            "total_downloads": 0,
            "average_daily": 0,
            "top_versions": [],
            "top_systems": [],
        }

        # Calculate totals from recent stats
        if stats.get("recent") and "data" in stats["recent"]:
            data_points = stats["recent"]["data"]
            total = sum(item.get("downloads", 0) for item in data_points)
            summary["total_downloads"] = total
            summary["average_daily"] = total / len(data_points) if data_points else 0

        # Extract top Python versions
        if stats.get("python_versions") and "data" in stats["python_versions"]:
            versions = sorted(
                stats["python_versions"]["data"],
                key=lambda x: x.get("downloads", 0),
                reverse=True
            )[:5]
            summary["top_versions"] = [
                {"version": v.get("category", "unknown"), "downloads": v.get("downloads", 0)}
                for v in versions
            ]

        # Extract top operating systems
        if stats.get("systems") and "data" in stats["systems"]:
            systems = sorted(
                stats["systems"]["data"],
                key=lambda x: x.get("downloads", 0),
                reverse=True
            )[:5]
            summary["top_systems"] = [
                {"system": s.get("category", "unknown"), "downloads": s.get("downloads", 0)}
                for s in systems
            ]

        return summary

    def save_stats(self, stats: Dict[str, Any]):
        """Save statistics to JSON and CSV files."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        # Save full JSON
        json_path = self.output_dir / f"downloads_{date_str}.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved full stats to {json_path}")

        # Save summary JSON
        summary = self.calculate_summary(stats)
        summary_path = self.output_dir / "download_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_path}")

        # Save CSV if pandas available
        if pd and stats.get("recent") and "data" in stats["recent"]:
            try:
                df = pd.DataFrame(stats["recent"]["data"])
                csv_path = self.output_dir / f"downloads_{date_str}.csv"
                df.to_csv(csv_path, index=False)
                print(f"✓ Saved CSV to {csv_path}")
            except Exception as e:
                print(f"Warning: Could not save CSV: {e}")

    def display_summary(self, summary: Dict[str, Any]):
        """Display summary statistics in the terminal."""
        print("\n" + "="*60)
        print(f"📊 Download Statistics for {summary['package']}")
        print("="*60)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Downloads: {summary['total_downloads']:,}")
        print(f"Average Daily: {summary['average_daily']:.1f}")

        if summary.get("top_versions"):
            print("\nTop Python Versions:")
            for v in summary["top_versions"]:
                print(f"  • Python {v['version']}: {v['downloads']:,} downloads")

        if summary.get("top_systems"):
            print("\nTop Operating Systems:")
            for s in summary["top_systems"]:
                print(f"  • {s['system']}: {s['downloads']:,} downloads")

        print("="*60 + "\n")

    def collect_all(self, days: int = 30) -> Dict[str, Any]:
        """Collect all available statistics."""
        stats = {
            "package": self.package_name,
            "timestamp": self.timestamp,
            "overall_month": self.fetch_overall_stats("month"),
            "overall_week": self.fetch_overall_stats("week"),
            "recent": self.fetch_recent_stats(days),
            "python_versions": self.fetch_python_versions(),
            "systems": self.fetch_system_stats(),
        }
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fetch download statistics for llamonitor-async from PyPI"
    )
    parser.add_argument(
        "--output-dir",
        default="./stats",
        help="Directory to save statistics (default: ./stats)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch (default: 30, max: 180)"
    )
    parser.add_argument(
        "--package",
        default=PACKAGE_NAME,
        help=f"Package name (default: {PACKAGE_NAME})"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display summary in terminal"
    )

    args = parser.parse_args()

    # Validate days
    if args.days < 1 or args.days > 180:
        print("Error: days must be between 1 and 180")
        sys.exit(1)

    # Collect statistics
    collector = DownloadStatsCollector(
        package_name=args.package,
        output_dir=args.output_dir
    )

    stats = collector.collect_all(days=args.days)

    # Save statistics
    collector.save_stats(stats)

    # Display summary
    if not args.no_display:
        summary = collector.calculate_summary(stats)
        collector.display_summary(summary)

    print("✅ Download statistics collection complete!")


if __name__ == "__main__":
    main()
