"""
Configuration settings for the dataset generator.
"""

import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    
    # Subgraph endpoints
    # Note: The Graph hosted service was deprecated. Try decentralized network or Sidecar API
    # Decentralized network: https://gateway-arbitrum.network.thegraph.com/api/[api-key]/subgraphs/id/[subgraph-id]
    # Or use EigenLayer Sidecar API: https://sidecar-rpc.eigenlayer.xyz
    eigenlayer_subgraph_url: str = "https://gateway.thegraph.com/api/subgraphs/id/QmYz8fAdooq2ay7vhY3T9cN2K3uH8z6YXx8J7K8vN5pX9q"  # Placeholder - needs actual subgraph ID
    avs_subgraph_url: str = "https://gateway.thegraph.com/api/subgraphs/id/QmYz8fAdooq2ay7vhY3T9cN2K3uH8z6YXx8J7K8vN5pX9q"  # Placeholder
    
    # Time window settings
    time_window_days: int = 7  # Weekly aggregation
    lookback_days: int = 90  # Historical data to fetch
    prediction_horizon_days: int = 7  # Days ahead to predict risk
    
    # Feature engineering
    min_stake_threshold: float = 1.0  # Minimum stake in ETH to include operator
    min_delegators: int = 1  # Minimum number of delegators
    gini_calculation_delegators: int = 3  # Top N delegators for concentration
    
    # Label generation thresholds
    stake_drop_threshold: float = 0.20  # 20% drop indicates risk
    avs_exit_threshold_days: int = 30  # Exit from AVS within N days = risk
    slashing_event_weight: float = 1.0  # Weight for actual slashing events
    
    # Output settings
    output_dir: str = "data"
    output_filename: str = "operator_reliability_dataset.csv"
    export_parquet: bool = True
    export_json: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "dataset_generation.log"
    
    # API settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100  # Operators per batch
    api_key: Optional[str] = None  # API key for EigenLayer API (or set EIGENLAYER_API_KEY env var)
    dune_api_key: Optional[str] = None  # Dune Analytics API key (or set DUNE_API_KEY env var)
    etherscan_api_key: Optional[str] = None  # Etherscan API key (or set ETHERSCAN_API_KEY env var)
    use_dune: bool = False  # Use Dune API as primary data source (currently not working)
    use_etherscan: bool = True  # Use Etherscan as primary data source
    
    # Feature flags
    include_market_data: bool = False  # Optional: CoinGecko integration
    include_advanced_features: bool = True  # Entropy, volatility, etc.
    
    def __post_init__(self):
        """Validate configuration."""
        if self.time_window_days <= 0:
            raise ValueError("time_window_days must be positive")
        if self.lookback_days < self.time_window_days:
            raise ValueError("lookback_days must be >= time_window_days")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
