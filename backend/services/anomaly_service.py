import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from backend.utils.logger import get_logger
from backend.utils.data_utils import detect_outliers_zscore

logger = get_logger(__name__)

class AnomalyDetectionService:
    
    @staticmethod
    def detect_anomalies(
        timeline: pd.DataFrame,
        expenses_df: pd.DataFrame
    ) -> List[Dict[str, any]]:
        
        logger.info("Starting anomaly detection")
        
        anomalies = []
        
        # 1. Expense spike detection
        expense_anomalies = AnomalyDetectionService._detect_expense_spikes(timeline, expenses_df)
        anomalies.extend(expense_anomalies)
        
        # 2. Sales drop detection
        sales_anomalies = AnomalyDetectionService._detect_sales_drops(timeline)
        anomalies.extend(sales_anomalies)
        
        # 3. Category-wise expense anomalies
        if 'category' in expenses_df.columns:
            category_anomalies = AnomalyDetectionService._detect_category_anomalies(expenses_df)
            anomalies.extend(category_anomalies)
        
        # 4. Cashflow volatility anomalies
        volatility_anomalies = AnomalyDetectionService._detect_volatility_anomalies(timeline)
        anomalies.extend(volatility_anomalies)
        
        # Sort by severity
        anomalies.sort(key=lambda x: x.get('severity', 0), reverse=True)
        
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        return anomalies
    
    @staticmethod
    def _detect_expense_spikes(timeline: pd.DataFrame, expenses_df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        # Detect outliers using Z-score
        expense_outliers = detect_outliers_zscore(timeline['expenses'], threshold=2.5)
        
        outlier_dates = timeline[expense_outliers]['date'].tolist()
        
        for date in outlier_dates[-5:]:  # Last 5 outliers
            date_data = timeline[timeline['date'] == date].iloc[0]
            mean_expense = timeline['expenses_rolling_mean_30d'].iloc[-1]
            
            if date_data['expenses'] > mean_expense:
                percentage_above = ((date_data['expenses'] - mean_expense) / mean_expense) * 100
                
                anomalies.append({
                    'type': 'expense_spike',
                    'category': 'Expenses',
                    'date': date.strftime('%Y-%m-%d'),
                    'severity': min(1.0, percentage_above / 100),
                    'description': f"{percentage_above:.1f}% above normal spending",
                    'amount': float(date_data['expenses']),
                    'expected_amount': float(mean_expense)
                })
        
        return anomalies
    
    @staticmethod
    def _detect_sales_drops(timeline: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        # Detect significant sales drops
        sales_outliers = detect_outliers_zscore(timeline['sales'], threshold=2.0)
        
        outlier_dates = timeline[sales_outliers]['date'].tolist()
        
        for date in outlier_dates[-3:]:  # Last 3 outliers
            date_data = timeline[timeline['date'] == date].iloc[0]
            mean_sales = timeline['sales_rolling_mean_30d'].iloc[-1]
            
            if date_data['sales'] < mean_sales * 0.5:  # Drop > 50%
                percentage_drop = ((mean_sales - date_data['sales']) / mean_sales) * 100
                
                anomalies.append({
                    'type': 'sales_drop',
                    'category': 'Sales',
                    'date': date.strftime('%Y-%m-%d'),
                    'severity': min(1.0, percentage_drop / 100),
                    'description': f"{percentage_drop:.1f}% below normal sales",
                    'amount': float(date_data['sales']),
                    'expected_amount': float(mean_sales)
                })
        
        return anomalies
    
    @staticmethod
    def _detect_category_anomalies(expenses_df: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        # Get recent 30 days and previous 30 days
        expenses_df = expenses_df.sort_values('date')
        recent_cutoff = expenses_df['date'].max() - pd.Timedelta(days=30)
        previous_cutoff = recent_cutoff - pd.Timedelta(days=30)
        
        recent_expenses = expenses_df[expenses_df['date'] > recent_cutoff]
        previous_expenses = expenses_df[
            (expenses_df['date'] > previous_cutoff) & 
            (expenses_df['date'] <= recent_cutoff)
        ]
        
        # Compare by category
        recent_by_category = recent_expenses.groupby('category')['amount'].sum()
        previous_by_category = previous_expenses.groupby('category')['amount'].sum()
        
        for category in recent_by_category.index:
            recent_amount = recent_by_category[category]
            previous_amount = previous_by_category.get(category, 0)
            
            if previous_amount > 0:
                change_pct = ((recent_amount - previous_amount) / previous_amount) * 100
                
                if abs(change_pct) > 30:  # 30% change threshold
                    anomalies.append({
                        'type': 'category_anomaly',
                        'category': category,
                        'severity': min(1.0, abs(change_pct) / 100),
                        'description': f"{abs(change_pct):.1f}% {'increase' if change_pct > 0 else 'decrease'}",
                        'amount': float(recent_amount),
                        'expected_amount': float(previous_amount),
                        'change_percentage': round(change_pct, 1)
                    })
        
        return anomalies
    
    @staticmethod
    def _detect_volatility_anomalies(timeline: pd.DataFrame) -> List[Dict]:
        anomalies = []
        
        # Check for sudden increase in volatility
        recent_volatility = timeline['sales_volatility'].tail(7).mean()
        historical_volatility = timeline['sales_volatility'].mean()
        
        if recent_volatility > historical_volatility * 1.5:
            anomalies.append({
                'type': 'volatility_spike',
                'category': 'Cashflow Pattern',
                'severity': min(1.0, recent_volatility / historical_volatility - 1),
                'description': 'Increased cashflow volatility detected',
                'amount': float(recent_volatility),
                'expected_amount': float(historical_volatility)
            })
        
        return anomalies
