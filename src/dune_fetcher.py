"""
Dune Analytics API fetcher for EigenLayer operator data.

Uses Dune Analytics free tier API to fetch EigenLayer operator information.
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DuneAPIFetcher:
    """
    Fetches EigenLayer operator data from Dune Analytics API.
    
    API Documentation: https://docs.dune.com/api-reference/eigenlayer
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.dune.com",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Dune API fetcher.
        
        Args:
            api_key: Dune Analytics API key
            base_url: Base URL for Dune API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup session with retry strategy
        self.session = requests.Session()
        self.session.headers.update({
            "X-Dune-API-Key": self.api_key,
            "Content-Type": "application/json"
        })
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized DuneAPIFetcher with base URL: {base_url}")
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to Dune API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"Dune API request failed: {e.response.status_code} - {e.response.text[:200]}")
            raise
        except Exception as e:
            logger.error(f"Dune API request error: {e}")
            raise
    
    def get_all_operators(self, skip: int = 0, first: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch all operators from Dune Analytics.
        
        Args:
            skip: Number of records to skip (for pagination)
            first: Maximum number of operators to return
            
        Returns:
            List of operator dictionaries
        """
        # Dune API endpoint for operator metadata
        endpoint = "/api/v1/eigenlayer/operators/metadata"
        
        try:
            data = self._make_request(endpoint)
            
            # Dune API returns data in specific format
            if isinstance(data, list):
                operators = data
            elif isinstance(data, dict) and "data" in data:
                operators = data["data"]
            elif isinstance(data, dict) and "results" in data:
                operators = data["results"]
            else:
                operators = []
            
            logger.info(f"Fetched {len(operators)} operators from Dune API")
            return operators[skip:skip+first]
            
        except Exception as e:
            logger.error(f"Failed to fetch operators from Dune: {e}")
            return []
    
    def get_operator_metrics(self, operator_address: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific operator."""
        endpoint = f"/api/v1/eigenlayer/operators/{operator_address}/metrics"
        
        try:
            return self._make_request(endpoint)
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {operator_address}: {e}")
            return None
    
    # Implement methods expected by the dataset generator
    def get_operator_delegations(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on Dune API."""
        return []
    
    def get_operator_avs_participation(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on Dune API."""
        return []
    
    def get_slashing_events(self, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on Dune API."""
        return []
    
    def get_operator_stake_history(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on Dune API."""
        return []
    
    def get_top_delegators(self, operator_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on Dune API."""
        return []
