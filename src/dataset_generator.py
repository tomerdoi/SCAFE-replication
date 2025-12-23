"""
Main dataset generator for Operator Reliability / Risk Dataset.

Orchestrates data fetching, feature engineering, and label generation.
"""

import logging
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .config import DatasetConfig
from .subgraph_fetcher import SubgraphDataFetcher
from .public_api_fetcher import PublicAPIFetcher
from .feature_engineer import FeatureEngineer
from .label_generator import LabelGenerator

logger = logging.getLogger(__name__)


class OperatorReliabilityDatasetGenerator:
    """
    Main class for generating the Operator Reliability / Risk Dataset.
    
    This dataset is designed for machine learning tasks to predict
    whether an operator is "high-risk" or "reliable".
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize the dataset generator.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or DatasetConfig()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components - try public API first, fallback to subgraph
        try:
            self.fetcher = PublicAPIFetcher(
                api_key=self.config.api_key,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            self.use_public_api = True
            logger.info("Using Public API fetcher")
        except Exception as e:
            logger.warning(f"Public API not available, using subgraph: {e}")
            self.fetcher = SubgraphDataFetcher(
                eigenlayer_url=self.config.eigenlayer_subgraph_url,
                avs_url=self.config.avs_subgraph_url,
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            self.use_public_api = False
        
        self.feature_engineer = FeatureEngineer(
            min_stake_threshold=self.config.min_stake_threshold,
            min_delegators=self.config.min_delegators,
            gini_delegators=self.config.gini_calculation_delegators,
            include_advanced=self.config.include_advanced_features
        )
        
        self.label_generator = LabelGenerator(
            stake_drop_threshold=self.config.stake_drop_threshold,
            avs_exit_threshold_days=self.config.avs_exit_threshold_days,
            slashing_event_weight=self.config.slashing_event_weight
        )
        
        logger.info("Initialized OperatorReliabilityDatasetGenerator")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # File handler (if specified)
        handlers = [console_handler]
        if self.config.log_file:
            file_handler = logging.FileHandler(
                os.path.join(self.config.output_dir, self.config.log_file)
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            handlers=handlers,
            format=log_format,
            datefmt=date_format
        )
    
    def generate_time_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[tuple]:
        """
        Generate time windows for dataset generation.
        
        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            
        Returns:
            List of (window_start, window_end) tuples
        """
        windows = []
        current = start_date
        
        while current < end_date:
            window_end = current + timedelta(days=self.config.time_window_days)
            if window_end > end_date:
                window_end = end_date
            
            windows.append((current, window_end))
            current = window_end
        
        logger.info(f"Generated {len(windows)} time windows")
        return windows
    
    def fetch_operator_data(
        self,
        operator_id: str,
        time_window_start: datetime,
        time_window_end: datetime
    ) -> Dict[str, Any]:
        """
        Fetch all data for a single operator in a time window.
        
        Args:
            operator_id: Operator address
            time_window_start: Start of time window
            time_window_end: End of time window
            
        Returns:
            Dictionary containing all operator data
        """
        try:
            # Convert to timestamps
            window_start_ts = int(time_window_start.timestamp())
            window_end_ts = int(time_window_end.timestamp())
            lookback_start_ts = int(
                (time_window_start - timedelta(days=self.config.lookback_days)).timestamp()
            )
            
            # Fetch data
            delegations = self.fetcher.get_operator_delegations(
                operator_id=operator_id,
                timestamp_from=lookback_start_ts,
                timestamp_to=window_end_ts
            )
            
            avs_participations = self.fetcher.get_operator_avs_participation(
                operator_id=operator_id,
                timestamp_from=lookback_start_ts,
                timestamp_to=window_end_ts
            )
            
            stake_history = self.fetcher.get_operator_stake_history(
                operator_id=operator_id,
                timestamp_from=lookback_start_ts,
                timestamp_to=window_end_ts
            )
            
            top_delegators = self.fetcher.get_top_delegators(
                operator_id=operator_id,
                limit=self.config.gini_calculation_delegators
            )
            
            # Get current stake (from most recent history or default)
            current_stake = 0.0
            if stake_history:
                current_stake = float(stake_history[-1].get("totalStake", 0))
            
            return {
                "operator_id": operator_id,
                "current_stake": current_stake,
                "delegations": delegations,
                "avs_participations": avs_participations,
                "stake_history": stake_history,
                "top_delegators": top_delegators,
                "slashing_events": []  # Will be fetched separately for all operators
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for operator {operator_id}: {e}")
            raise
    
    def generate_dataset(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate the complete dataset.
        
        Args:
            start_date: Start date for dataset. If None, uses lookback_days from now.
            end_date: End date for dataset. If None, uses current date.
            
        Returns:
            DataFrame with features and labels
        """
        try:
            # Set default dates
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=self.config.lookback_days)
            
            logger.info(f"Generating dataset from {start_date} to {end_date}")
            
            # Generate time windows
            time_windows = self.generate_time_windows(start_date, end_date)
            
            # Fetch all operators
            logger.info("Fetching all operators...")
            if self.use_public_api:
                # Public API returns all operators at once
                all_operators = self.fetcher.get_all_operators()
                logger.info(f"Fetched {len(all_operators)} operators from public API")
            else:
                # Subgraph uses pagination
                all_operators = []
                skip = 0
                while True:
                    operators_batch = self.fetcher.get_all_operators(
                        skip=skip,
                        first=self.config.batch_size
                    )
                    if not operators_batch:
                        break
                    all_operators.extend(operators_batch)
                    skip += len(operators_batch)
                    logger.info(f"Fetched {len(all_operators)} operators so far...")
                    
                    if len(operators_batch) < self.config.batch_size:
                        break
            
            logger.info(f"Total operators fetched: {len(all_operators)}")
            
            # Fetch slashing events for all time windows
            logger.info("Fetching slashing events...")
            all_slashing_events = []
            for window_start, window_end in time_windows:
                window_start_ts = int(window_start.timestamp())
                window_end_ts = int((window_end + timedelta(
                    days=self.config.prediction_horizon_days
                )).timestamp())
                
                events = self.fetcher.get_slashing_events(
                    timestamp_from=window_start_ts,
                    timestamp_to=window_end_ts
                )
                all_slashing_events.extend(events)
            
            logger.info(f"Total slashing events fetched: {len(all_slashing_events)}")
            
            # Generate dataset rows
            dataset_rows = []
            prediction_horizon = timedelta(days=self.config.prediction_horizon_days)
            
            total_combinations = len(all_operators) * len(time_windows)
            processed = 0
            
            logger.info(f"Processing {total_combinations} operator-window combinations...")
            
            for operator in all_operators:
                # Handle different API response formats
                operator_id = (
                    operator.get("id") or 
                    operator.get("address") or 
                    operator.get("operator") or
                    operator.get("operatorAddress")
                )
                
                if not operator_id:
                    logger.warning(f"Skipping operator with no ID: {operator}")
                    continue
                
                for window_start, window_end in time_windows:
                    try:
                        # Fetch operator data
                        operator_data = self.fetch_operator_data(
                            operator_id=operator_id,
                            time_window_start=window_start,
                            time_window_end=window_end
                        )
                        
                        # Add slashing events
                        operator_data["slashing_events"] = all_slashing_events
                        
                        # Engineer features
                        features = self.feature_engineer.engineer_features(
                            operator_data=operator_data,
                            time_window_start=window_start,
                            time_window_end=window_end,
                            lookback_days=self.config.lookback_days
                        )
                        
                        if features is None:
                            continue  # Skipped due to thresholds
                        
                        # Generate label
                        label_info = self.label_generator.generate_label(
                            operator_id=operator_id,
                            operator_data=operator_data,
                            time_window_start=window_start,
                            time_window_end=window_end,
                            prediction_horizon=prediction_horizon
                        )
                        
                        # Combine features and label
                        row = {
                            **features,
                            "label": label_info["label"],
                            "is_high_risk": label_info["is_high_risk"],
                            "label_reasons": ",".join(label_info["label_reasons"]),
                            "label_confidence": label_info["label_confidence"]
                        }
                        
                        dataset_rows.append(row)
                        processed += 1
                        
                        if processed % 100 == 0:
                            logger.info(f"Processed {processed}/{total_combinations} combinations...")
                    
                    except Exception as e:
                        logger.warning(
                            f"Error processing operator {operator_id} "
                            f"for window {window_start}: {e}"
                        )
                        continue
            
            logger.info(f"Generated {len(dataset_rows)} dataset rows")
            
            # Create DataFrame
            df = pd.DataFrame(dataset_rows)
            
            if len(df) == 0:
                logger.warning("No data rows generated!")
                return df
            
            # Add metadata columns
            df["dataset_version"] = "1.0.0"
            df["generated_at"] = datetime.now().isoformat()
            df["config_time_window_days"] = self.config.time_window_days
            df["config_lookback_days"] = self.config.lookback_days
            df["config_prediction_horizon_days"] = self.config.prediction_horizon_days
            
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Label distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            raise
    
    def export_dataset(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export dataset to files.
        
        Args:
            df: DataFrame to export
            filename: Optional custom filename (without extension)
            
        Returns:
            Dictionary with paths to exported files
        """
        try:
            if filename is None:
                filename = self.config.output_filename.replace(".csv", "")
            
            output_paths = {}
            
            # Export CSV
            csv_path = os.path.join(self.config.output_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path
            logger.info(f"Exported CSV to {csv_path}")
            
            # Export Parquet (if enabled)
            if self.config.export_parquet:
                parquet_path = os.path.join(self.config.output_dir, f"{filename}.parquet")
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                output_paths["parquet"] = parquet_path
                logger.info(f"Exported Parquet to {parquet_path}")
            
            # Export JSON (if enabled)
            if self.config.export_json:
                json_path = os.path.join(self.config.output_dir, f"{filename}.json")
                df.to_json(json_path, orient="records", indent=2)
                output_paths["json"] = json_path
                logger.info(f"Exported JSON to {json_path}")
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            raise
    
    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        export: bool = True
    ) -> pd.DataFrame:
        """
        Run the complete dataset generation pipeline.
        
        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            export: Whether to export dataset to files
            
        Returns:
            Generated DataFrame
        """
        try:
            logger.info("Starting dataset generation pipeline...")
            
            # Generate dataset
            df = self.generate_dataset(start_date=start_date, end_date=end_date)
            
            # Export if requested
            if export and len(df) > 0:
                self.export_dataset(df)
            
            logger.info("Dataset generation pipeline completed successfully!")
            return df
            
        except Exception as e:
            logger.error(f"Error in dataset generation pipeline: {e}")
            raise
