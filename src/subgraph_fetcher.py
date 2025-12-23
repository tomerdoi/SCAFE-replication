"""
Subgraph data fetcher for EigenLayer and AVS data.

This module handles fetching on-chain data from The Graph subgraphs.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SubgraphDataFetcher:
    """
    Fetches data from EigenLayer and AVS subgraphs.
    
    Handles GraphQL queries, retries, and error handling.
    """
    
    def __init__(
        self,
        eigenlayer_url: str,
        avs_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the subgraph fetcher.
        
        Args:
            eigenlayer_url: URL for EigenLayer subgraph
            avs_url: URL for AVS subgraph
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.eigenlayer_url = eigenlayer_url
        self.avs_url = avs_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"Initialized SubgraphDataFetcher with EigenLayer: {eigenlayer_url}")
    
    def _execute_query(
        self,
        url: str,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query with error handling.
        
        Args:
            url: Subgraph endpoint URL
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Response data dictionary
            
        Raises:
            requests.RequestException: If request fails after retries
            ValueError: If response contains errors
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "errors" in data:
                error_msg = "; ".join([e.get("message", "Unknown error") for e in data["errors"]])
                logger.error(f"GraphQL errors: {error_msg}")
                raise ValueError(f"GraphQL query failed: {error_msg}")
            
            return data.get("data", {})
            
        except requests.Timeout as e:
            logger.error(f"Request timeout for {url}: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def get_all_operators(self, skip: int = 0, first: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch all operators from EigenLayer subgraph.
        
        Args:
            skip: Number of records to skip (for pagination)
            first: Number of records to fetch
            
        Returns:
            List of operator dictionaries
        """
        query = """
        query GetOperators($skip: Int!, $first: Int!) {
            operators(
                skip: $skip
                first: $first
                orderBy: totalStake
                orderDirection: desc
            ) {
                id
                totalStake
                delegatorCount
                createdAt
            }
        }
        """
        
        variables = {"skip": skip, "first": first}
        
        try:
            data = self._execute_query(self.eigenlayer_url, query, variables)
            operators = data.get("operators", [])
            logger.info(f"Fetched {len(operators)} operators (skip={skip})")
            return operators
        except Exception as e:
            logger.error(f"Failed to fetch operators: {e}")
            raise
    
    def get_operator_delegations(
        self,
        operator_id: str,
        timestamp_from: int,
        timestamp_to: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch delegations for a specific operator within a time range.
        
        Args:
            operator_id: Operator address
            timestamp_from: Start timestamp (Unix)
            timestamp_to: End timestamp (Unix)
            
        Returns:
            List of delegation dictionaries
        """
        query = """
        query GetDelegations($operator: String!, $timestampFrom: BigInt!, $timestampTo: BigInt!) {
            delegations(
                where: {
                    operator: $operator
                    timestamp_gte: $timestampFrom
                    timestamp_lte: $timestampTo
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                id
                delegator
                amount
                timestamp
                transactionHash
            }
        }
        """
        
        variables = {
            "operator": operator_id,
            "timestampFrom": timestamp_from,
            "timestampTo": timestamp_to
        }
        
        try:
            data = self._execute_query(self.eigenlayer_url, query, variables)
            delegations = data.get("delegations", [])
            logger.debug(f"Fetched {len(delegations)} delegations for operator {operator_id}")
            return delegations
        except Exception as e:
            logger.error(f"Failed to fetch delegations for {operator_id}: {e}")
            raise
    
    def get_operator_avs_participation(
        self,
        operator_id: str,
        timestamp_from: int,
        timestamp_to: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch AVS participation history for an operator.
        
        Args:
            operator_id: Operator address
            timestamp_from: Start timestamp (Unix)
            timestamp_to: End timestamp (Unix)
            
        Returns:
            List of AVS participation events
        """
        query = """
        query GetAVSParticipation($operator: String!, $timestampFrom: BigInt!, $timestampTo: BigInt!) {
            operatorAVSRegistrations(
                where: {
                    operator: $operator
                    timestamp_gte: $timestampFrom
                    timestamp_lte: $timestampTo
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                id
                operator
                avs
                timestamp
                transactionHash
                isActive
            }
        }
        """
        
        variables = {
            "operator": operator_id,
            "timestampFrom": timestamp_from,
            "timestampTo": timestamp_to
        }
        
        try:
            data = self._execute_query(self.avs_url, query, variables)
            participations = data.get("operatorAVSRegistrations", [])
            logger.debug(f"Fetched {len(participations)} AVS participations for operator {operator_id}")
            return participations
        except Exception as e:
            logger.error(f"Failed to fetch AVS participation for {operator_id}: {e}")
            raise
    
    def get_slashing_events(
        self,
        timestamp_from: int,
        timestamp_to: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch slashing events within a time range.
        
        Args:
            timestamp_from: Start timestamp (Unix)
            timestamp_to: End timestamp (Unix)
            
        Returns:
            List of slashing event dictionaries
        """
        query = """
        query GetSlashingEvents($timestampFrom: BigInt!, $timestampTo: BigInt!) {
            slashingEvents(
                where: {
                    timestamp_gte: $timestampFrom
                    timestamp_lte: $timestampTo
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                id
                operator
                avs
                amount
                timestamp
                transactionHash
                reason
            }
        }
        """
        
        variables = {
            "timestampFrom": timestamp_from,
            "timestampTo": timestamp_to
        }
        
        try:
            data = self._execute_query(self.eigenlayer_url, query, variables)
            events = data.get("slashingEvents", [])
            logger.info(f"Fetched {len(events)} slashing events")
            return events
        except Exception as e:
            logger.error(f"Failed to fetch slashing events: {e}")
            raise
    
    def get_operator_stake_history(
        self,
        operator_id: str,
        timestamp_from: int,
        timestamp_to: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical stake changes for an operator.
        
        Args:
            operator_id: Operator address
            timestamp_from: Start timestamp (Unix)
            timestamp_to: End timestamp (Unix)
            
        Returns:
            List of stake change events
        """
        query = """
        query GetStakeHistory($operator: String!, $timestampFrom: BigInt!, $timestampTo: BigInt!) {
            operatorStakeUpdates(
                where: {
                    operator: $operator
                    timestamp_gte: $timestampFrom
                    timestamp_lte: $timestampTo
                }
                orderBy: timestamp
                orderDirection: asc
            ) {
                id
                operator
                totalStake
                timestamp
                transactionHash
            }
        }
        """
        
        variables = {
            "operator": operator_id,
            "timestampFrom": timestamp_from,
            "timestampTo": timestamp_to
        }
        
        try:
            data = self._execute_query(self.eigenlayer_url, query, variables)
            updates = data.get("operatorStakeUpdates", [])
            logger.debug(f"Fetched {len(updates)} stake updates for operator {operator_id}")
            return updates
        except Exception as e:
            logger.error(f"Failed to fetch stake history for {operator_id}: {e}")
            raise
    
    def get_top_delegators(
        self,
        operator_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch top delegators by stake amount for an operator.
        
        Args:
            operator_id: Operator address
            limit: Number of top delegators to fetch
            
        Returns:
            List of delegator dictionaries with stake amounts
        """
        query = """
        query GetTopDelegators($operator: String!, $limit: Int!) {
            delegations(
                where: { operator: $operator }
                orderBy: amount
                orderDirection: desc
                first: $limit
            ) {
                id
                delegator
                amount
            }
        }
        """
        
        variables = {
            "operator": operator_id,
            "limit": limit
        }
        
        try:
            data = self._execute_query(self.eigenlayer_url, query, variables)
            delegators = data.get("delegations", [])
            logger.debug(f"Fetched top {len(delegators)} delegators for operator {operator_id}")
            return delegators
        except Exception as e:
            logger.error(f"Failed to fetch top delegators for {operator_id}: {e}")
            raise
