"""
Results Analyzer for LLMOps Monitoring

Reads Parquet files and displays:
- Capacity metrics by node/function
- Graph topology (parent-child relationships)
- Session/trace grouping
- Summary statistics

Run with: python analyze_results.py [data_directory]
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configure logging for script
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.error("pandas not installed. Run: pip install pandas pyarrow")
    sys.exit(1)


def load_monitoring_data(data_dir: str = "./test_monitoring_data") -> pd.DataFrame:
    """Load all Parquet files from monitoring output."""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Directory not found: {data_path}")
        logger.info("Run a test first to generate data:")
        logger.info("  python test_basic_monitoring.py")
        sys.exit(1)

    parquet_files = list(data_path.rglob("*.parquet"))

    if not parquet_files:
        logger.error(f"No Parquet files found in {data_path}")
        sys.exit(1)

    logger.info(f"Loading {len(parquet_files)} Parquet file(s)...")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"✓ Loaded {len(df)} events\n")
    return df


def display_summary(df: pd.DataFrame):
    """Display overall summary statistics."""
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print()

    print(f"Total Events:          {len(df)}")
    print(f"Unique Sessions:       {df['session_id'].nunique()}")
    print(f"Unique Traces:         {df['trace_id'].nunique()}")
    print(f"Unique Operations:     {df['operation_name'].nunique()}")
    print()

    # Text metrics
    if 'text_char_count' in df.columns:
        total_chars = df['text_char_count'].sum()
        total_words = df['text_word_count'].sum()
        total_bytes = df['text_byte_size'].sum()

        print("TEXT CAPACITY:")
        print(f"  Total Characters:    {total_chars:,}")
        print(f"  Total Words:         {total_words:,}")
        print(f"  Total Bytes:         {total_bytes:,}")
        print()

    # Image metrics
    if 'image_count' in df.columns and df['image_count'].notna().any():
        total_images = df['image_count'].sum()
        total_pixels = df['image_total_pixels'].sum()
        total_img_bytes = df['image_file_size_bytes'].sum()

        print("IMAGE CAPACITY:")
        print(f"  Total Images:        {int(total_images) if pd.notna(total_images) else 0}")
        print(f"  Total Pixels:        {int(total_pixels) if pd.notna(total_pixels) else 0:,}")
        print(f"  Total Bytes:         {int(total_img_bytes) if pd.notna(total_img_bytes) else 0:,}")
        print()

    # Performance
    if 'duration_ms' in df.columns:
        avg_duration = df['duration_ms'].mean()
        total_duration = df['duration_ms'].sum()

        print("PERFORMANCE:")
        print(f"  Avg Duration:        {avg_duration:.2f} ms")
        print(f"  Total Duration:      {total_duration:.2f} ms ({total_duration/1000:.2f} sec)")
        print()


def display_by_operation(df: pd.DataFrame):
    """Display metrics grouped by operation/node."""
    print("=" * 70)
    print("METRICS BY OPERATION / NODE")
    print("=" * 70)
    print()

    ops = df.groupby('operation_name').agg({
        'event_id': 'count',
        'text_char_count': 'sum',
        'text_word_count': 'sum',
        'duration_ms': 'mean'
    }).round(2)

    ops.columns = ['Count', 'Total Chars', 'Total Words', 'Avg Duration (ms)']
    ops = ops.sort_values('Count', ascending=False)

    print(ops.to_string())
    print()


def display_graph_topology(df: pd.DataFrame):
    """Display hierarchical graph topology."""
    print("=" * 70)
    print("GRAPH TOPOLOGY (Parent-Child Relationships)")
    print("=" * 70)
    print()

    # Group by session for clearer display
    for session_id in df['session_id'].unique():
        session_df = df[df['session_id'] == session_id].sort_values('timestamp')

        print(f"Session: {session_id}")
        print("-" * 60)

        # Build tree structure
        span_map = {}
        for _, row in session_df.iterrows():
            span_map[row['span_id']] = {
                'name': row['operation_name'],
                'parent': row['parent_span_id'],
                'chars': row.get('text_char_count', 0),
                'duration': row.get('duration_ms', 0)
            }

        # Find roots (no parent)
        roots = [sid for sid, info in span_map.items() if pd.isna(info['parent'])]

        # Display tree
        def print_tree(span_id, indent=0):
            if span_id not in span_map:
                return

            info = span_map[span_id]
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            chars_str = f"{int(info['chars'])} chars" if info['chars'] > 0 else ""
            duration_str = f"{info['duration']:.1f}ms" if info['duration'] > 0 else ""
            details = f"({chars_str}, {duration_str})".replace("(, ", "(").replace(", )", ")")

            print(f"{prefix}{info['name']} {details}")

            # Find children
            children = [sid for sid, sinfo in span_map.items() if sinfo['parent'] == span_id]
            for child in children:
                print_tree(child, indent + 1)

        for root in roots:
            print_tree(root)

        print()


def display_model_usage(df: pd.DataFrame):
    """Display models used (from custom_attributes)."""
    print("=" * 70)
    print("MODEL USAGE")
    print("=" * 70)
    print()

    if 'custom_attributes' not in df.columns:
        print("No model information found.")
        print()
        return

    models = defaultdict(int)
    for attrs_json in df['custom_attributes'].dropna():
        try:
            attrs = json.loads(attrs_json)
            if 'model' in attrs:
                models[attrs['model']] += 1
        except:
            pass

    if models:
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model:30s} {count:4d} calls")
    else:
        print("No model information found in custom_attributes.")

    print()


def display_timeline(df: pd.DataFrame):
    """Display event timeline."""
    print("=" * 70)
    print("EVENT TIMELINE (Most Recent)")
    print("=" * 70)
    print()

    timeline = df.sort_values('timestamp', ascending=False).head(20)

    for _, row in timeline.iterrows():
        timestamp = row['timestamp'] if isinstance(row['timestamp'], str) else row['timestamp'].strftime("%H:%M:%S")
        chars = f"{int(row['text_char_count'])} chars" if pd.notna(row.get('text_char_count')) else ""
        duration = f"{row['duration_ms']:.1f}ms" if pd.notna(row.get('duration_ms')) else ""

        print(f"  {timestamp} | {row['operation_name']:25s} | {chars:12s} | {duration}")

    print()


def export_summary_report(df: pd.DataFrame, output_file: str = "monitoring_report.txt"):
    """Export detailed report to text file."""
    with open(output_file, 'w') as f:
        f.write("LLMOps Monitoring Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Summary stats
        f.write(f"Total Events: {len(df)}\n")
        f.write(f"Total Text Characters: {df['text_char_count'].sum():,}\n")
        f.write(f"Total Text Words: {df['text_word_count'].sum():,}\n")
        f.write(f"Average Duration: {df['duration_ms'].mean():.2f} ms\n\n")

        # By operation
        f.write("Metrics by Operation:\n")
        ops = df.groupby('operation_name').agg({
            'event_id': 'count',
            'text_char_count': 'sum',
            'text_word_count': 'sum',
            'duration_ms': 'mean'
        }).round(2)
        f.write(ops.to_string())

    print(f"✓ Detailed report exported to: {output_file}\n")


def main():
    """Main analyzer function."""
    print()
    print("=" * 70)
    print("LLMOps Monitoring - Results Analyzer")
    print("=" * 70)
    print()

    # Load data
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_monitoring_data"
    df = load_monitoring_data(data_dir)

    # Display analysis
    display_summary(df)
    display_by_operation(df)
    display_graph_topology(df)
    display_model_usage(df)
    display_timeline(df)

    # Export report
    export_summary_report(df)

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nAnalysis interrupted by user")
    except Exception as e:
        logger.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
