#!/usr/bin/env python3
"""
Generate a small test dataset for validation.

This script generates a minimal dataset with a small time range.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src import OperatorReliabilityDatasetGenerator, DatasetConfig
    import logging
except ImportError as e:
    print(f"Error: Missing dependencies. Please install:")
    print(f"  pip3 install --user -r requirements.txt")
    print(f"\nMissing: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def main():
    """Generate a small test dataset."""
    logger.info("=" * 60)
    logger.info("Generating Small Test Dataset")
    logger.info("=" * 60)
    logger.info("")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Configure for small dataset with real API keys
    import os
    dune_key = os.getenv("DUNE_API_KEY", "6MNlQuZPLMwwDEmkRt63lLbXg2pJTcP7")
    etherscan_key = os.getenv("ETHERSCAN_API_KEY", "Y2X5Z21WJP6M9SP5UM6PWD7AHJZJDY6AEA")
    
    config = DatasetConfig(
        # API keys for free tier services
        dune_api_key=dune_key,
        etherscan_api_key=etherscan_key,
        use_dune=False,  # Try Etherscan first, then Dune if needed
        use_etherscan=True,  # Enable Etherscan API to fetch real data
        
        # Subgraph endpoints (fallback)
        eigenlayer_subgraph_url="https://gateway.thegraph.com/api/subgraphs/id/68g9WSC4QTUJmMpuSbgLNENrcYha4mPmXhWGCoupM7kB",
        avs_subgraph_url="https://gateway.thegraph.com/api/subgraphs/id/68g9WSC4QTUJmMpuSbgLNENrcYha4mPmXhWGCoupM7kB",
        
        # Small time range for testing
        time_window_days=7,  # Weekly windows
        lookback_days=30,  # Only 1 month lookback (smaller)
        prediction_horizon_days=7,
        
        # Thresholds
        min_stake_threshold=0.1,  # Lower threshold for more operators
        min_delegators=1,
        
        # Output
        output_dir=str(data_dir),
        output_filename="operator_reliability_dataset_small.csv",
        export_parquet=True,
        export_json=True,
        
        # Smaller batch size for testing
        batch_size=50,
        
        # Logging
        log_level="INFO",
        log_file="dataset_generation_small.log"
    )
    
    # Initialize generator
    logger.info("Initializing generator...")
    generator = OperatorReliabilityDatasetGenerator(config=config)
    
    # Small time range: last 2 weeks
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    
    logger.info("")
    logger.info(f"Time range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Lookback: {config.lookback_days} days")
    logger.info("")
    logger.info("Starting dataset generation...")
    logger.info("")
    
    try:
        # Generate dataset
        df = generator.generate_dataset(
            start_date=start_date,
            end_date=end_date
        )
        
        # Export
        if len(df) > 0:
            logger.info("")
            logger.info("Exporting dataset...")
            paths = generator.export_dataset(df, filename="operator_reliability_dataset_small")
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("Dataset Generated Successfully!")
            logger.info("=" * 60)
            logger.info(f"✓ Total rows: {len(df):,}")
            logger.info(f"✓ Total operators: {df['operator_id'].nunique():,}")
            logger.info(f"✓ Time windows: {df['time_window_start'].nunique():,}")
            logger.info(f"✓ Feature columns: {len(df.columns)}")
            logger.info("")
            logger.info("Label distribution:")
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                for label, count in label_counts.items():
                    pct = (count / len(df)) * 100
                    label_name = "High-Risk" if label == 1 else "Reliable"
                    logger.info(f"  {label_name} (label={label}): {count:,} ({pct:.1f}%)")
            logger.info("")
            logger.info("Output files:")
            for file_type, path in paths.items():
                logger.info(f"  - {path}")
            logger.info("")
            logger.info("Sample data:")
            logger.info("")
            print(df.head().to_string())
        else:
            logger.warning("⚠ No data generated!")
            logger.warning("This could mean:")
            logger.warning("  - Subgraph endpoints are not accessible")
            logger.warning("  - No data available for the time range")
            logger.warning("  - All operators filtered out by thresholds")
            logger.warning("")
            logger.warning("Check the log file: data/dataset_generation_small.log")
        
        logger.info("")
        logger.info("Done!")
        
    except KeyboardInterrupt:
        logger.info("")
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("")
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("")
        logger.error("Check the log file for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
