import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from backend.utils.logger import get_logger
from backend.utils.data_utils import create_daily_timeline, fill_missing_dates, calculate_rolling_stats

logger = get_logger(__name__)

class FeatureEngineeringService:
    
    @staticmethod
    def create_features(sales_df: pd.DataFrame, expenses_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering")
        
        # Create daily timeline
        timeline = create_daily_timeline(sales_df, expenses_df)
        
        # Fill missing dates
        timeline = fill_missing_dates(timeline, 'date')
        
        # Calculate rolling statistics
        timeline = calculate_rolling_stats(timeline, 'sales', windows=[7, 14, 30])
        timeline = calculate_rolling_stats(timeline, 'expenses', windows=[7, 14, 30])
        timeline = calculate_rolling_stats(timeline, 'cashflow', windows=[7, 14, 30])
        
        # Add time-based features
        timeline['day_of_week'] = timeline['date'].dt.dayofweek
        timeline['day_of_month'] = timeline['date'].dt.day
        timeline['month'] = timeline['date'].dt.month
        timeline['quarter'] = timeline['date'].dt.quarter
        timeline['is_weekend'] = timeline['day_of_week'].isin([5, 6]).astype(int)
        timeline['is_month_end'] = (timeline['date'].dt.day >= 25).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            timeline[f'sales_lag_{lag}'] = timeline['sales'].shift(lag)
            timeline[f'expenses_lag_{lag}'] = timeline['expenses'].shift(lag)
            timeline[f'cashflow_lag_{lag}'] = timeline['cashflow'].shift(lag)
        
        # Fill NaN values from lag features
        timeline = timeline.fillna(0)
        
        # Sales/Expense ratio
        timeline['sales_expense_ratio'] = np.where(
            timeline['expenses'] > 0,
            timeline['sales'] / timeline['expenses'],
            0
        )
        
        # Volatility
        timeline['sales_volatility'] = timeline['sales'].rolling(window=30, min_periods=1).std()
        timeline['expense_volatility'] = timeline['expenses'].rolling(window=30, min_periods=1).std()
        
        # Trend indicators
        timeline['sales_trend'] = timeline['sales'] - timeline['sales_rolling_mean_30d']
        timeline['expense_trend'] = timeline['expenses'] - timeline['expenses_rolling_mean_30d']
        
        logger.info(f"Feature engineering completed. Total features: {len(timeline.columns)}")
        
        return timeline
    
    @staticmethod
    def prepare_sequence_data(df: pd.DataFrame, lookback: int = 60, forecast_horizon: int = 30) -> Dict:
        logger.info(f"Preparing sequence data with lookback={lookback}, horizon={forecast_horizon}")
        
        # Select relevant features for modeling
        feature_columns = [
            'sales', 'expenses', 'cashflow', 'cumulative_cashflow',
            'sales_rolling_mean_7d', 'sales_rolling_mean_14d', 'sales_rolling_mean_30d',
            'expenses_rolling_mean_7d', 'expenses_rolling_mean_14d', 'expenses_rolling_mean_30d',
            'day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_month_end',
            'sales_expense_ratio'
        ]
        
        # Ensure all required columns exist
        available_columns = [col for col in feature_columns if col in df.columns]
        
        data = df[available_columns].values
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        sequences = []
        targets = []
        
        for i in range(len(data_scaled) - lookback - forecast_horizon + 1):
            sequences.append(data_scaled[i:i+lookback])
            # Target is the cashflow for next 'forecast_horizon' days
            target_idx = df.columns.get_loc('cashflow')
            targets.append(data[i+lookback:i+lookback+forecast_horizon, target_idx])
        
        logger.info(f"Created {len(sequences)} sequences")
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets),
            'scaler': scaler,
            'feature_columns': available_columns
        }
    
    @staticmethod
    def extract_expense_categories(expenses_df: pd.DataFrame) -> Dict[str, float]:
        if 'category' not in expenses_df.columns:
            return {'General': float(expenses_df['amount'].sum())}
        
        category_totals = expenses_df.groupby('category')['amount'].sum().to_dict()
        return {k: float(v) for k, v in category_totals.items()}
