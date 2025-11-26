import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from backend.utils.logger import get_logger
from backend.utils.data_utils import parse_date_column

logger = get_logger(__name__)

class DataIngestService:
    
    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    @staticmethod
    def validate_sales_data(df: pd.DataFrame) -> Dict[str, any]:
        required_columns = ['date', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to auto-detect columns
            df = DataIngestService._auto_detect_columns(df, 'sales')
        
        # Validate data types
        df = parse_date_column(df, 'date')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Remove negative amounts
        negative_count = (df['amount'] < 0).sum()
        if negative_count > 0:
            logger.warning(f"Removing {negative_count} negative sales amounts")
            df = df[df['amount'] >= 0]
        
        logger.info(f"Sales data validated: {len(df)} records")
        
        return {
            'dataframe': df,
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'total_amount': float(df['amount'].sum())
        }
    
    @staticmethod
    def validate_expense_data(df: pd.DataFrame) -> Dict[str, any]:
        required_columns = ['date', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to auto-detect columns
            df = DataIngestService._auto_detect_columns(df, 'expense')
        
        # Add category if not present
        if 'category' not in df.columns:
            df['category'] = 'General'
            logger.info("Added default category column")
        
        # Validate data types
        df = parse_date_column(df, 'date')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
        
        # Ensure amounts are positive
        df['amount'] = df['amount'].abs()
        
        logger.info(f"Expense data validated: {len(df)} records")
        
        # Category breakdown
        category_breakdown = df.groupby('category')['amount'].sum().to_dict()
        
        return {
            'dataframe': df,
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'total_amount': float(df['amount'].sum()),
            'categories': category_breakdown
        }
    
    @staticmethod
    def _auto_detect_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        logger.info(f"Attempting to auto-detect columns for {data_type} data")
        
        # Common date column names
        date_candidates = ['date', 'Date', 'DATE', 'transaction_date', 'txn_date', 'time', 'timestamp']
        
        # Common amount column names
        amount_candidates = ['amount', 'Amount', 'AMOUNT', 'value', 'total', 'price', 'sales', 'expense']
        
        # Common category column names
        category_candidates = ['category', 'Category', 'CATEGORY', 'type', 'description', 'desc']
        
        # Find date column
        date_col = None
        for col in df.columns:
            if col.lower() in [c.lower() for c in date_candidates]:
                date_col = col
                break
        
        if date_col:
            df = df.rename(columns={date_col: 'date'})
            logger.info(f"Detected date column: {date_col}")
        else:
            raise ValueError("Could not detect date column. Please ensure your CSV has a 'date' column.")
        
        # Find amount column
        amount_col = None
        for col in df.columns:
            if col.lower() in [c.lower() for c in amount_candidates] and col != 'date':
                amount_col = col
                break
        
        if amount_col:
            df = df.rename(columns={amount_col: 'amount'})
            logger.info(f"Detected amount column: {amount_col}")
        else:
            raise ValueError("Could not detect amount column. Please ensure your CSV has an 'amount' column.")
        
        # Find category column (optional for expenses)
        if data_type == 'expense':
            category_col = None
            for col in df.columns:
                if col.lower() in [c.lower() for c in category_candidates] and col not in ['date', 'amount']:
                    category_col = col
                    break
            
            if category_col:
                df = df.rename(columns={category_col: 'category'})
                logger.info(f"Detected category column: {category_col}")
        
        return df
    
    @staticmethod
    def parse_bank_sms(sms_text: str) -> pd.DataFrame:
        logger.info("Parsing bank SMS transactions")
        
        # This is a simplified version - in production, use regex patterns for different banks
        transactions = []
        lines = sms_text.strip().split('\n')
        
        for line in lines:
            # Simple parsing logic - customize based on actual SMS format
            if 'debited' in line.lower() or 'credited' in line.lower():
                # Extract transaction details using basic parsing
                # In production, use proper regex patterns
                pass
        
        df = pd.DataFrame(transactions)
        logger.info(f"Parsed {len(df)} transactions from SMS")
        
        return df
