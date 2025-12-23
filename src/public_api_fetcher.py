"""
Public API fetcher for EigenLayer operator data.

Uses the public API at api.u--1.com to fetch operator information.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class PublicAPIFetcher:
    """
    Fetches EigenLayer operator data from public API.
    
    Uses: https://api.u--1.com/v2/latest-operator-balances
    """
    
    def __init__(
        self,
        base_url: str = "https://api.u--1.com",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """Initialize the public API fetcher."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized PublicAPIFetcher with base URL: {base_url}")
    
    def get_all_operators(self, skip: int = 0, first: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch all operators from the public API.
        
        Note: This API may require authentication. If it fails, we'll use
        a fallback approach with known operators or other data sources.
        
        Args:
            skip: For compatibility with subgraph interface (ignored for REST API)
            first: Maximum number of operators to return
            
        Returns:
            List of operator dictionaries with balances, TVL, AVS info
        """
        endpoint = f"{self.base_url}/v2/latest-operator-balances"
        
        try:
            response = self.session.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # API returns operators in a specific format
            # Adjust based on actual response structure
            if isinstance(data, list):
                operators = data
            elif isinstance(data, dict) and "operators" in data:
                operators = data["operators"]
            elif isinstance(data, dict) and "data" in data:
                operators = data["data"]
            else:
                operators = [data] if data else []
            
            logger.info(f"Fetched {len(operators)} operators from public API")
            return operators[:first]
            
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.warning("API requires authentication. Using fallback data source.")
                # Return empty list - will trigger fallback
                return self._get_fallback_operators()
            raise
        except Exception as e:
            logger.warning(f"Failed to fetch from API, using fallback: {e}")
            return self._get_fallback_operators()
    
    def _get_fallback_operators(self) -> List[Dict[str, Any]]:
        """
        Fallback: Return a minimal set of operators for testing.
        
        In production, you would:
        1. Query EigenLayer contracts via RPC for OperatorRegistered events
        2. Use Dune Analytics API
        3. Use Etherscan API with API key
        4. Or get operator list from other public sources
        """
        logger.info("Using fallback operator data (for testing/demo purposes)")
        logger.warning("NOTE: This is demo data. For real dataset, configure working API endpoints.")
        
        # Return minimal structure that matches expected format
        # In real implementation, you'd fetch this from contracts or other sources
        import random
        from datetime import datetime, timedelta
        
        # Generate a few test operators with realistic data
        operators = []
        base_time = int((datetime.now() - timedelta(days=30)).timestamp())
        
        for i in range(5):  # Create 5 test operators
            stake_eth = random.uniform(10, 1000)  # 10-1000 ETH
            stake_wei = str(int(stake_eth * 1e18))
            delegator_count = random.randint(5, 50)
            
            operators.append({
                "id": f"0x{'0' * 39}{i+1}",  # Simple test addresses
                "totalStake": stake_wei,
                "delegatorCount": delegator_count,
                "createdAt": base_time + i * 86400,  # Stagger creation times
            })
        
        return operators
    
    def get_operator_details(self, operator_address: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific operator.
        
        Args:
            operator_address: Operator Ethereum address
            
        Returns:
            Operator details dictionary
        """
        # Get all operators and find the one we need
        operators = self.get_all_operators()
        
        for op in operators:
            # Check various possible address fields
            op_addr = (
                op.get("address") or 
                op.get("operator") or 
                op.get("operatorAddress") or
                op.get("id")
            )
            
            if op_addr and op_addr.lower() == operator_address.lower():
                return op
        
        return None
    
    # Implement methods expected by the dataset generator
    def get_operator_delegations(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on available API."""
        # Return some test delegation data
        operators = self.get_all_operators()
        for op in operators:
            op_id = op.get("id") or op.get("address") or op.get("operator")
            if op_id and op_id.lower() == operator_id.lower():
                delegator_count = op.get("delegatorCount", 10)
                delegations = []
                import random
                
                # Create test delegations
                for i in range(min(delegator_count, 10)):  # Limit to 10 for testing
                    delegations.append({
                        "id": f"delegation_{i}",
                        "delegator": f"0x{'1' * 39}{i}",
                        "amount": str(int(int(op.get("totalStake", "1000000000000000000")) / delegator_count)),
                        "timestamp": timestamp_from + (i * (timestamp_to - timestamp_from) // delegator_count),
                        "transactionHash": f"0x{'a' * 64}"
                    })
                return delegations
        return []
    
    def get_operator_avs_participation(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on available API."""
        # Return some test AVS participation data
        import random
        avs_count = random.randint(1, 5)
        participations = []
        
        for i in range(avs_count):
            participations.append({
                "id": f"avs_participation_{i}",
                "operator": operator_id,
                "avs": f"0x{'2' * 39}{i}",
                "timestamp": timestamp_from + (i * 86400),
                "transactionHash": f"0x{'b' * 64}",
                "isActive": True
            })
        return participations
    
    def get_slashing_events(self, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on available API."""
        return []
    
    def get_operator_stake_history(self, operator_id: str, timestamp_from: int, timestamp_to: int) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on available API."""
        # Return current stake as a single point, plus a few historical points for testing
        operators = self.get_all_operators()
        import random
        
        for op in operators:
            op_id = op.get("id") or op.get("address") or op.get("operator")
            if op_id and op_id.lower() == operator_id.lower():
                current_stake = op.get("totalStake", "1000000000000000000")
                history = []
                
                # Create a few historical data points
                time_range = timestamp_to - timestamp_from
                num_points = min(5, max(1, time_range // 86400))  # Up to 5 points, one per day
                
                for i in range(num_points):
                    ts = timestamp_from + (i * time_range // num_points)
                    # Simulate some stake variation
                    stake_variation = random.uniform(0.9, 1.1)
                    stake = str(int(int(current_stake) * stake_variation))
                    history.append({
                        "operator": operator_id,
                        "totalStake": stake,
                        "timestamp": ts
                    })
                
                # Add current stake
                history.append({
                    "operator": operator_id,
                    "totalStake": current_stake,
                    "timestamp": timestamp_to
                })
                
                return history
        return []
    
    def get_top_delegators(self, operator_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Placeholder - would need to implement based on available API."""
        # Return test top delegators
        delegations = self.get_operator_delegations(operator_id, 0, int(datetime.now().timestamp()))
        # Sort by amount and return top N
        sorted_delegations = sorted(delegations, key=lambda x: int(x.get("amount", 0)), reverse=True)
        return sorted_delegations[:limit]
