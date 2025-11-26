import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def parse_date_column(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        invalid_dates = df[date_column].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid dates in column {date_column}")
        
        df = df.dropna(subset=[date_column])
        return df
    except Exception as e:
        logger.error(f"Error parsing date column: {str(e)}")
        raise

def create_daily_timeline(sales_df: pd.DataFrame, expenses_df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Aggregate sales by day
        sales_daily = sales_df.groupby('date')['amount'].sum().reset_index()
        sales_daily.columns = ['date', 'sales']
        
        # Aggregate expenses by day
        expenses_daily = expenses_df.groupby('date')['amount'].sum().reset_index()
        expenses_daily.columns = ['date', 'expenses']
        
        # Merge on date
        timeline = pd.merge(sales_daily, expenses_daily, on='date', how='outer')
        timeline = timeline.fillna(0)
        
        # Sort by date
        timeline = timeline.sort_values('date').reset_index(drop=True)
        
        # Calculate daily cashflow
        timeline['cashflow'] = timeline['sales'] - timeline['expenses']
        timeline['cumulative_cashflow'] = timeline['cashflow'].cumsum()
        
        logger.info(f"Created daily timeline with {len(timeline)} days")
        return timeline
    
    except Exception as e:
        logger.error(f"Error creating daily timeline: {str(e)}")
        raise

def fill_missing_dates(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    try:
        df = df.sort_values(date_column).reset_index(drop=True)
        
        min_date = df[date_column].min()
        max_date = df[date_column].max()
        
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        full_df = pd.DataFrame({date_column: all_dates})
        merged_df = pd.merge(full_df, df, on=date_column, how='left')
        
        # Forward fill for missing values
        numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)
        
        logger.info(f"Filled missing dates. Total days: {len(merged_df)}")
        return merged_df
    
    except Exception as e:
        logger.error(f"Error filling missing dates: {str(e)}")
        raise

def calculate_rolling_stats(df: pd.DataFrame, column: str, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    try:
        for window in windows:
            df[f'{column}_rolling_mean_{window}d'] = df[column].rolling(window=window, min_periods=1).mean()
            df[f'{column}_rolling_std_{window}d'] = df[column].rolling(window=window, min_periods=1).std()
        
        return df
    except Exception as e:
        logger.error(f"Error calculating rolling stats: {str(e)}")
        raise

def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    try:
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise
