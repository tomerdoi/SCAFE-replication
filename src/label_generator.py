"""
Label generator for operator reliability classification.

Generates binary labels (high-risk vs reliable) using proxy indicators.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generates labels for operator reliability prediction.
    
    Uses proxy labels based on:
    - On-chain slashing events
    - AVS-reported penalties
    - Sustained stake drops
    - Operator exits from AVSs
    """
    
    def __init__(
        self,
        stake_drop_threshold: float = 0.20,
        avs_exit_threshold_days: int = 30,
        slashing_event_weight: float = 1.0
    ):
        """
        Initialize label generator.
        
        Args:
            stake_drop_threshold: Percentage drop in stake indicating risk (0-1)
            avs_exit_threshold_days: Days after which AVS exit indicates risk
            slashing_event_weight: Weight for actual slashing events
        """
        self.stake_drop_threshold = stake_drop_threshold
        self.avs_exit_threshold_days = avs_exit_threshold_days
        self.slashing_event_weight = slashing_event_weight
        
        logger.info("Initialized LabelGenerator")
    
    def check_slashing_events(
        self,
        operator_id: str,
        slashing_events: List[Dict[str, Any]],
        time_window_start: datetime,
        time_window_end: datetime,
        prediction_horizon: timedelta
    ) -> bool:
        """
        Check if operator experienced slashing in prediction window.
        
        Args:
            operator_id: Operator address
            slashing_events: List of slashing events
            time_window_start: Start of feature window
            time_window_end: End of feature window
            prediction_horizon: Time ahead to check for slashing
            
        Returns:
            True if slashing occurred in prediction window
        """
        try:
            window_end_ts = int(time_window_end.timestamp())
            horizon_end_ts = int((time_window_end + prediction_horizon).timestamp())
            
            operator_slashings = [
                s for s in slashing_events
                if s.get("operator", "").lower() == operator_id.lower()
                and window_end_ts < int(s.get("timestamp", 0)) <= horizon_end_ts
            ]
            
            return len(operator_slashings) > 0
            
        except Exception as e:
            logger.error(f"Error checking slashing events for {operator_id}: {e}")
            return False
    
    def check_stake_drop(
        self,
        stake_history: List[Dict[str, Any]],
        time_window_end: datetime,
        prediction_horizon: timedelta
    ) -> bool:
        """
        Check if operator experienced significant stake drop.
        
        Args:
            stake_history: List of stake update events
            time_window_end: End of feature window
            prediction_horizon: Time ahead to check for stake drop
            
        Returns:
            True if stake dropped significantly in prediction window
        """
        try:
            if not stake_history or len(stake_history) < 2:
                return False
            
            window_end_ts = int(time_window_end.timestamp())
            horizon_end_ts = int((time_window_end + prediction_horizon).timestamp())
            
            # Get stake at end of feature window
            window_stakes = [
                float(s.get("totalStake", 0))
                for s in stake_history
                if int(s.get("timestamp", 0)) <= window_end_ts
            ]
            
            if not window_stakes:
                return False
            
            stake_at_window_end = window_stakes[-1]
            
            # Get stake in prediction window
            horizon_stakes = [
                float(s.get("totalStake", 0))
                for s in stake_history
                if window_end_ts < int(s.get("timestamp", 0)) <= horizon_end_ts
            ]
            
            if not horizon_stakes:
                return False
            
            # Check for sustained drop
            min_stake_in_horizon = min(horizon_stakes)
            
            if stake_at_window_end > 0:
                drop_pct = (stake_at_window_end - min_stake_in_horizon) / stake_at_window_end
                return drop_pct >= self.stake_drop_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking stake drop: {e}")
            return False
    
    def check_avs_exit(
        self,
        operator_id: str,
        avs_participations: List[Dict[str, Any]],
        time_window_end: datetime,
        prediction_horizon: timedelta
    ) -> bool:
        """
        Check if operator exited from AVSs in prediction window.
        
        Args:
            operator_id: Operator address
            avs_participations: List of AVS participation events
            time_window_end: End of feature window
            prediction_horizon: Time ahead to check for exits
            
        Returns:
            True if operator exited from AVS within threshold
        """
        try:
            window_end_ts = int(time_window_end.timestamp())
            horizon_end_ts = int((time_window_end + prediction_horizon).timestamp())
            
            # Get AVSs active at end of window
            active_at_window_end = {
                p.get("avs")
                for p in avs_participations
                if p.get("operator", "").lower() == operator_id.lower()
                and p.get("isActive", True)
                and int(p.get("timestamp", 0)) <= window_end_ts
            }
            
            if not active_at_window_end:
                return False
            
            # Check for deactivations in prediction window
            deactivations = [
                p for p in avs_participations
                if p.get("operator", "").lower() == operator_id.lower()
                and not p.get("isActive", True)
                and window_end_ts < int(p.get("timestamp", 0)) <= horizon_end_ts
            ]
            
            if not deactivations:
                return False
            
            # Check if exit happened within threshold
            for deact in deactivations:
                deact_ts = int(deact.get("timestamp", 0))
                days_after_window = (datetime.fromtimestamp(deact_ts) - time_window_end).days
                
                if days_after_window <= self.avs_exit_threshold_days:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking AVS exit for {operator_id}: {e}")
            return False
    
    def generate_label(
        self,
        operator_id: str,
        operator_data: Dict[str, Any],
        time_window_start: datetime,
        time_window_end: datetime,
        prediction_horizon: timedelta
    ) -> Dict[str, Any]:
        """
        Generate label for an operator in a time window.
        
        Args:
            operator_id: Operator address
            operator_data: Dictionary containing operator data
            time_window_start: Start of feature window
            time_window_end: End of feature window
            prediction_horizon: Time ahead to predict risk
            
        Returns:
            Dictionary with label and label metadata
        """
        try:
            label_info = {
                "operator_id": operator_id,
                "is_high_risk": False,
                "label_reasons": [],
                "label_confidence": 0.0
            }
            
            # Check each risk indicator
            has_slashing = self.check_slashing_events(
                operator_id=operator_id,
                slashing_events=operator_data.get("slashing_events", []),
                time_window_start=time_window_start,
                time_window_end=time_window_end,
                prediction_horizon=prediction_horizon
            )
            
            has_stake_drop = self.check_stake_drop(
                stake_history=operator_data.get("stake_history", []),
                time_window_end=time_window_end,
                prediction_horizon=prediction_horizon
            )
            
            has_avs_exit = self.check_avs_exit(
                operator_id=operator_id,
                avs_participations=operator_data.get("avs_participations", []),
                time_window_end=time_window_end,
                prediction_horizon=prediction_horizon
            )
            
            # Combine indicators (OR logic: any indicator = high risk)
            if has_slashing:
                label_info["is_high_risk"] = True
                label_info["label_reasons"].append("slashing_event")
                label_info["label_confidence"] += self.slashing_event_weight
            
            if has_stake_drop:
                label_info["is_high_risk"] = True
                label_info["label_reasons"].append("stake_drop")
                label_info["label_confidence"] += 0.5
            
            if has_avs_exit:
                label_info["is_high_risk"] = True
                label_info["label_reasons"].append("avs_exit")
                label_info["label_confidence"] += 0.3
            
            # Normalize confidence to 0-1
            label_info["label_confidence"] = min(
                label_info["label_confidence"] / (self.slashing_event_weight + 0.5 + 0.3),
                1.0
            )
            
            # Convert to binary label
            label_info["label"] = 1 if label_info["is_high_risk"] else 0
            
            return label_info
            
        except Exception as e:
            logger.error(f"Error generating label for {operator_id}: {e}")
            raise
