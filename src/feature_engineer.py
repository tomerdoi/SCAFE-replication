"""
Feature engineering module for operator reliability dataset.

Computes on-chain, behavioral, and economic features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features from raw subgraph data.
    
    Computes:
    - On-chain features (stake, delegators, concentration, AVS participation)
    - Behavioral features (churn, growth rates, AVS changes)
    - Economic features (slashable amounts, reward variance)
    """
    
    def __init__(
        self,
        min_stake_threshold: float = 1.0,
        min_delegators: int = 1,
        gini_delegators: int = 3,
        include_advanced: bool = True
    ):
        """
        Initialize feature engineer.
        
        Args:
            min_stake_threshold: Minimum stake in ETH to include
            min_delegators: Minimum number of delegators
            gini_delegators: Top N delegators for concentration calculation
            include_advanced: Whether to compute advanced features (entropy, volatility)
        """
        self.min_stake_threshold = min_stake_threshold
        self.min_delegators = min_delegators
        self.gini_delegators = gini_delegators
        self.include_advanced = include_advanced
        
        logger.info("Initialized FeatureEngineer")
    
    def compute_gini_coefficient(self, amounts: List[float]) -> float:
        """
        Compute Gini coefficient for stake concentration.
        
        Args:
            amounts: List of stake amounts
            
        Returns:
            Gini coefficient (0-1, higher = more concentrated)
        """
        if not amounts or len(amounts) == 0:
            return 0.0
        
        if len(amounts) == 1:
            return 1.0
        
        amounts = sorted(amounts, reverse=True)
        n = len(amounts)
        cumsum = np.cumsum(amounts)
        total = cumsum[-1]
        
        if total == 0:
            return 0.0
        
        # Gini = 1 - (2 * sum of cumulative proportions) / n
        indices = np.arange(1, n + 1)
        gini = 1 - (2 * np.sum(cumsum / total * indices)) / (n * (n + 1))
        
        return float(gini)
    
    def compute_entropy(self, amounts: List[float]) -> float:
        """
        Compute Shannon entropy for AVS diversity.
        
        Args:
            amounts: List of stake amounts per AVS
            
        Returns:
            Entropy value (higher = more diverse)
        """
        if not amounts or len(amounts) == 0:
            return 0.0
        
        total = sum(amounts)
        if total == 0:
            return 0.0
        
        proportions = [a / total for a in amounts if a > 0]
        if not proportions:
            return 0.0
        
        entropy = -sum(p * np.log2(p) for p in proportions)
        return float(entropy)
    
    def compute_stake_volatility(
        self,
        stake_history: List[Dict[str, Any]],
        window_days: int = 30
    ) -> float:
        """
        Compute stake volatility over a time window.
        
        Args:
            stake_history: List of stake update events
            window_days: Time window in days
            
        Returns:
            Coefficient of variation (std/mean) of stake
        """
        if not stake_history or len(stake_history) < 2:
            return 0.0
        
        # Extract stake values
        stakes = [float(event.get("totalStake", 0)) for event in stake_history]
        
        if len(stakes) < 2:
            return 0.0
        
        mean_stake = np.mean(stakes)
        if mean_stake == 0:
            return 0.0
        
        std_stake = np.std(stakes)
        volatility = std_stake / mean_stake if mean_stake > 0 else 0.0
        
        return float(volatility)
    
    def compute_on_chain_features(
        self,
        operator_id: str,
        current_stake: float,
        delegations: List[Dict[str, Any]],
        avs_participations: List[Dict[str, Any]],
        top_delegators: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Compute on-chain features for an operator.
        
        Args:
            operator_id: Operator address
            current_stake: Current total stake in ETH
            delegations: List of delegation events
            avs_participations: List of AVS participation events
            top_delegators: Optional list of top delegators with amounts
            
        Returns:
            Dictionary of on-chain features
        """
        try:
            features = {
                "operator_id": operator_id,
                "total_stake": float(current_stake),
                "num_delegators": len(set(d.get("delegator") for d in delegations)),
            }
            
            # Stake concentration (Gini coefficient)
            if top_delegators:
                amounts = [float(d.get("amount", 0)) for d in top_delegators]
                features["stake_gini"] = self.compute_gini_coefficient(amounts)
                features["stake_concentration_top3_pct"] = (
                    sum(amounts[:3]) / current_stake * 100
                    if current_stake > 0 else 0.0
                )
            else:
                # Fallback: compute from all delegations
                amounts = [float(d.get("amount", 0)) for d in delegations]
                features["stake_gini"] = self.compute_gini_coefficient(amounts)
                features["stake_concentration_top3_pct"] = 0.0
            
            # AVS features
            active_avss = [p for p in avs_participations if p.get("isActive", True)]
            features["avs_count"] = len(set(p.get("avs") for p in active_avss))
            
            if self.include_advanced and active_avss:
                # AVS diversity (entropy)
                avs_stakes = defaultdict(float)
                for p in active_avss:
                    avs_id = p.get("avs")
                    # Approximate: assume equal stake distribution per AVS
                    avs_stakes[avs_id] += current_stake / len(active_avss)
                
                features["avs_diversity_entropy"] = self.compute_entropy(
                    list(avs_stakes.values())
                )
            else:
                features["avs_diversity_entropy"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing on-chain features for {operator_id}: {e}")
            raise
    
    def compute_behavioral_features(
        self,
        operator_id: str,
        delegations: List[Dict[str, Any]],
        avs_participations: List[Dict[str, Any]],
        stake_history: List[Dict[str, Any]],
        time_window_start: datetime,
        time_window_end: datetime,
        lookback_days: int = 90
    ) -> Dict[str, float]:
        """
        Compute behavioral features over a time window.
        
        Args:
            operator_id: Operator address
            delegations: List of delegation events
            avs_participations: List of AVS participation events
            stake_history: List of stake update events
            time_window_start: Start of current time window
            time_window_end: End of current time window
            lookback_days: Days to look back for historical comparison
            
        Returns:
            Dictionary of behavioral features
        """
        try:
            features = {}
            
            # Time window boundaries
            window_start_ts = int(time_window_start.timestamp())
            window_end_ts = int(time_window_end.timestamp())
            lookback_start_ts = int((time_window_start - timedelta(days=lookback_days)).timestamp())
            
            # Delegation churn
            delegations_in_window = [
                d for d in delegations
                if window_start_ts <= int(d.get("timestamp", 0)) <= window_end_ts
            ]
            delegations_lookback = [
                d for d in delegations
                if lookback_start_ts <= int(d.get("timestamp", 0)) < window_start_ts
            ]
            
            delegators_current = set(d.get("delegator") for d in delegations_in_window)
            delegators_previous = set(d.get("delegator") for d in delegations_lookback)
            
            # Churn metrics
            new_delegators = len(delegators_current - delegators_previous)
            lost_delegators = len(delegators_previous - delegators_current)
            total_delegators_previous = len(delegators_previous) if delegators_previous else 1
            
            features["delegation_churn_rate"] = (
                lost_delegators / total_delegators_previous * 100
                if total_delegators_previous > 0 else 0.0
            )
            features["new_delegators_count"] = float(new_delegators)
            features["lost_delegators_count"] = float(lost_delegators)
            
            # Stake growth rate
            if stake_history:
                current_stake = float(stake_history[-1].get("totalStake", 0))
                lookback_stakes = [
                    float(s.get("totalStake", 0))
                    for s in stake_history
                    if lookback_start_ts <= int(s.get("timestamp", 0)) < window_start_ts
                ]
                
                if lookback_stakes:
                    previous_stake = lookback_stakes[-1]
                    if previous_stake > 0:
                        features["stake_growth_30d_pct"] = (
                            (current_stake - previous_stake) / previous_stake * 100
                        )
                    else:
                        features["stake_growth_30d_pct"] = 0.0
                else:
                    features["stake_growth_30d_pct"] = 0.0
            else:
                features["stake_growth_30d_pct"] = 0.0
            
            # AVS changes
            avs_changes = [
                p for p in avs_participations
                if window_start_ts <= int(p.get("timestamp", 0)) <= window_end_ts
            ]
            features["avs_switches_90d"] = len(avs_changes)
            
            # Time since last AVS opt-in
            opt_ins = [
                p for p in avs_participations
                if p.get("isActive", True) and int(p.get("timestamp", 0)) <= window_end_ts
            ]
            if opt_ins:
                last_opt_in_ts = max(int(p.get("timestamp", 0)) for p in opt_ins)
                last_opt_in_dt = datetime.fromtimestamp(last_opt_in_ts)
                days_since_opt_in = (time_window_end - last_opt_in_dt).days
                features["days_since_last_avs_optin"] = float(days_since_opt_in)
            else:
                features["days_since_last_avs_optin"] = 999.0  # Never opted in
            
            # Stake volatility (if advanced features enabled)
            if self.include_advanced and stake_history:
                features["stake_volatility_30d"] = self.compute_stake_volatility(
                    stake_history,
                    window_days=30
                )
            else:
                features["stake_volatility_30d"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing behavioral features for {operator_id}: {e}")
            raise
    
    def compute_economic_features(
        self,
        operator_id: str,
        current_stake: float,
        avs_participations: List[Dict[str, Any]],
        slashing_events: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute economic exposure features.
        
        Args:
            operator_id: Operator address
            current_stake: Current total stake
            avs_participations: List of AVS participation events
            slashing_events: List of slashing events (for this operator)
            
        Returns:
            Dictionary of economic features
        """
        try:
            features = {}
            
            # Max slashable amount (simplified: assume 10% per AVS, sum across AVSs)
            active_avss = [p for p in avs_participations if p.get("isActive", True)]
            num_avss = len(set(p.get("avs") for p in active_avss))
            
            # Conservative estimate: assume each AVS can slash up to 10% of stake
            # In practice, this would come from AVS-specific slashing parameters
            max_slash_per_avs = 0.10  # 10%
            features["max_slashable_amount"] = current_stake * max_slash_per_avs * num_avss
            features["max_slashable_pct"] = min(max_slash_per_avs * num_avss, 1.0) * 100
            
            # Historical slashing
            operator_slashings = [
                s for s in slashing_events
                if s.get("operator", "").lower() == operator_id.lower()
            ]
            features["historical_slashing_count"] = float(len(operator_slashings))
            features["historical_slashing_amount"] = sum(
                float(s.get("amount", 0)) for s in operator_slashings
            )
            
            # Reward rate variance (placeholder - would need actual reward data)
            # For now, use AVS count as proxy for reward diversity
            features["reward_diversity_proxy"] = float(num_avss)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing economic features for {operator_id}: {e}")
            raise
    
    def engineer_features(
        self,
        operator_data: Dict[str, Any],
        time_window_start: datetime,
        time_window_end: datetime,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Engineer all features for an operator in a time window.
        
        Args:
            operator_data: Dictionary containing all operator data
            time_window_start: Start of time window
            time_window_end: End of time window
            lookback_days: Days to look back for historical features
            
        Returns:
            Dictionary of all features
        """
        try:
            operator_id = operator_data["operator_id"]
            current_stake = operator_data.get("current_stake", 0.0)
            
            # Filter out operators below threshold
            if current_stake < self.min_stake_threshold:
                logger.debug(f"Skipping operator {operator_id}: stake below threshold")
                return None
            
            # Compute feature groups
            on_chain = self.compute_on_chain_features(
                operator_id=operator_id,
                current_stake=current_stake,
                delegations=operator_data.get("delegations", []),
                avs_participations=operator_data.get("avs_participations", []),
                top_delegators=operator_data.get("top_delegators")
            )
            
            behavioral = self.compute_behavioral_features(
                operator_id=operator_id,
                delegations=operator_data.get("delegations", []),
                avs_participations=operator_data.get("avs_participations", []),
                stake_history=operator_data.get("stake_history", []),
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                lookback_days=lookback_days
            )
            
            economic = self.compute_economic_features(
                operator_id=operator_id,
                current_stake=current_stake,
                avs_participations=operator_data.get("avs_participations", []),
                slashing_events=operator_data.get("slashing_events", [])
            )
            
            # Combine all features
            all_features = {
                **on_chain,
                **behavioral,
                **economic,
                "time_window_start": time_window_start.isoformat(),
                "time_window_end": time_window_end.isoformat(),
            }
            
            # Filter by minimum delegators
            if all_features.get("num_delegators", 0) < self.min_delegators:
                logger.debug(f"Skipping operator {operator_id}: delegators below threshold")
                return None
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error engineering features for operator: {e}")
            raise
