import numpy as np
import pandas as pd
from typing import Dict, List
from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)

class RiskScoringService:
    
    @staticmethod
    def calculate_composite_risk_score(
        cashflow_forecast: np.ndarray,
        anomalies: List[Dict],
        invoice_risks: List[Dict],
        current_balance: float
    ) -> Dict[str, any]:
        
        logger.info("Calculating composite risk score")
        
        # Component 1: Cashflow Risk (40% weight)
        cashflow_risk = RiskScoringService._calculate_cashflow_risk(cashflow_forecast, current_balance)
        
        # Component 2: Anomaly Risk (30% weight)
        anomaly_risk = RiskScoringService._calculate_anomaly_risk(anomalies)
        
        # Component 3: Invoice Risk (30% weight)
        invoice_risk = RiskScoringService._calculate_invoice_risk(invoice_risks)
        
        # Weighted composite score
        composite_score = (
            0.40 * cashflow_risk +
            0.30 * anomaly_risk +
            0.30 * invoice_risk
        )
        
        # Ensure score is between 0-100
        composite_score = max(0, min(100, composite_score))
        
        # Determine risk level
        if composite_score < settings.RISK_THRESHOLD_LOW:
            risk_level = "LOW"
            risk_color = "green"
        elif composite_score < settings.RISK_THRESHOLD_HIGH:
            risk_level = "MEDIUM"
            risk_color = "yellow"
        else:
            risk_level = "HIGH"
            risk_color = "red"
        
        # Find critical day (day when balance goes negative)
        critical_day = RiskScoringService._find_critical_day(cashflow_forecast, current_balance)
        
        logger.info(f"Risk score calculated: {composite_score:.2f} ({risk_level})")
        
        return {
            'overall_score': round(composite_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'components': {
                'cashflow_risk': round(cashflow_risk, 2),
                'anomaly_risk': round(anomaly_risk, 2),
                'invoice_risk': round(invoice_risk, 2)
            },
            'critical_day': critical_day,
            'message': RiskScoringService._generate_risk_message(composite_score, critical_day)
        }
    
    @staticmethod
    def _calculate_cashflow_risk(forecast: np.ndarray, current_balance: float) -> float:
        cumulative_forecast = np.cumsum(forecast) + current_balance
        
        # Count days with negative balance
        negative_days = np.sum(cumulative_forecast < 0)
        
        # Minimum projected balance
        min_balance = np.min(cumulative_forecast)
        
        # Calculate risk based on negative days and severity
        if negative_days == 0:
            risk = 10.0
        else:
            # More negative days = higher risk
            day_risk = (negative_days / len(forecast)) * 50
            
            # More negative minimum = higher risk
            balance_risk = min(50, abs(min_balance) / 10000 * 50) if min_balance < 0 else 0
            
            risk = min(100, day_risk + balance_risk)
        
        return risk
    
    @staticmethod
    def _calculate_anomaly_risk(anomalies: List[Dict]) -> float:
        if not anomalies:
            return 5.0
        
        # Count critical anomalies
        critical_count = sum(1 for a in anomalies if a.get('severity', 0) > 0.7)
        medium_count = sum(1 for a in anomalies if 0.5 <= a.get('severity', 0) <= 0.7)
        
        # Calculate risk
        risk = min(100, (critical_count * 25) + (medium_count * 15))
        
        return max(5, risk)
    
    @staticmethod
    def _calculate_invoice_risk(invoice_risks: List[Dict]) -> float:
        if not invoice_risks:
            return 10.0
        
        # Calculate weighted risk based on invoice amounts and risk scores
        total_amount = sum(inv.get('amount', 0) for inv in invoice_risks)
        
        if total_amount == 0:
            return 10.0
        
        weighted_risk = sum(
            inv.get('risk_score', 0) * inv.get('amount', 0)
            for inv in invoice_risks
        ) / total_amount
        
        return min(100, weighted_risk)
    
    @staticmethod
    def _find_critical_day(forecast: np.ndarray, current_balance: float) -> int:
        cumulative = np.cumsum(forecast) + current_balance
        
        negative_indices = np.where(cumulative < 0)[0]
        
        if len(negative_indices) > 0:
            return int(negative_indices[0] + 1)  # +1 for 1-indexed days
        
        return -1  # No critical day found
    
    @staticmethod
    def _generate_risk_message(score: float, critical_day: int) -> str:
        if score < 30:
            if critical_day > 0:
                return f"Low risk with potential cash shortfall on day {critical_day}"
            return "Financial health is stable with low risk"
        
        elif score < 60:
            if critical_day > 0:
                return f"Moderate risk with liquidity issues expected on day {critical_day}"
            return "Moderate risk detected. Monitor cashflow closely"
        
        else:
            if critical_day > 0:
                return f"High risk of cash shortfall in next {critical_day} days"
            return "High financial risk. Immediate action recommended"
