import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)

class ForecastService:
    
    @staticmethod
    def simple_forecast(historical_data: pd.DataFrame, forecast_days: int = 30) -> Dict[str, any]:
        logger.info(f"Generating simple forecast for {forecast_days} days")
        
        # Use exponential moving average for forecast
        sales_ema = historical_data['sales'].ewm(span=14, adjust=False).mean()
        expenses_ema = historical_data['expenses'].ewm(span=14, adjust=False).mean()
        
        # Last known values
        last_sales = sales_ema.iloc[-1]
        last_expenses = expenses_ema.iloc[-1]
        
        # Calculate trend
        recent_sales = historical_data['sales'].tail(30).values
        recent_expenses = historical_data['expenses'].tail(30).values
        
        sales_trend = np.polyfit(range(len(recent_sales)), recent_sales, 1)[0]
        expense_trend = np.polyfit(range(len(recent_expenses)), recent_expenses, 1)[0]
        
        # Generate forecast
        forecast_sales = []
        forecast_expenses = []
        forecast_cashflow = []
        
        for day in range(1, forecast_days + 1):
            # Add trend and some randomness
            sales_pred = last_sales + (sales_trend * day) + np.random.normal(0, last_sales * 0.1)
            expense_pred = last_expenses + (expense_trend * day) + np.random.normal(0, last_expenses * 0.1)
            
            # Ensure non-negative
            sales_pred = max(0, sales_pred)
            expense_pred = max(0, expense_pred)
            
            forecast_sales.append(sales_pred)
            forecast_expenses.append(expense_pred)
            forecast_cashflow.append(sales_pred - expense_pred)
        
        # Calculate cumulative cashflow
        current_balance = historical_data['cumulative_cashflow'].iloc[-1]
        cumulative_forecast = np.cumsum(forecast_cashflow) + current_balance
        
        # Find minimum balance
        min_balance = np.min(cumulative_forecast)
        min_balance_day = np.argmin(cumulative_forecast) + 1
        
        logger.info(f"Forecast completed. Min balance: {min_balance:.2f} on day {min_balance_day}")
        
        return {
            'forecast_days': forecast_days,
            'daily_forecast': [
                {
                    'day': i + 1,
                    'sales': round(forecast_sales[i], 2),
                    'expenses': round(forecast_expenses[i], 2),
                    'cashflow': round(forecast_cashflow[i], 2),
                    'cumulative_balance': round(cumulative_forecast[i], 2)
                }
                for i in range(forecast_days)
            ],
            'summary': {
                'total_sales': round(sum(forecast_sales), 2),
                'total_expenses': round(sum(forecast_expenses), 2),
                'net_cashflow': round(sum(forecast_cashflow), 2),
                'min_balance': round(min_balance, 2),
                'min_balance_day': min_balance_day,
                'starting_balance': round(current_balance, 2),
                'ending_balance': round(cumulative_forecast[-1], 2)
            }
        }
    
    @staticmethod
    def gru_forecast_placeholder(sequence_data: Dict, forecast_days: int = 30) -> np.ndarray:
        logger.info("GRU forecast placeholder - using simple method")
        
        # In production, this would use a trained GRU model
        # For now, return simple forecast
        last_values = sequence_data['sequences'][-1, -1, :]
        
        # Simple persistence forecast with decay
        forecast = []
        for i in range(forecast_days):
            decay_factor = 0.95 ** i
            pred = last_values[0] * decay_factor  # Assuming first feature is cashflow
            forecast.append(pred)
        
        return np.array(forecast)
