import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class InvoiceRiskService:
    
    @staticmethod
    def generate_sample_invoices(amount_range: tuple = (5000, 25000)) -> List[Dict]:
        logger.info("Generating sample invoice data")
        
        customers = ['S. Traders', 'V-Mart', 'Global Corp', 'Tech Solutions', 'Metro Supplies']
        
        invoices = []
        base_date = datetime.now()
        
        for i in range(5):
            invoice_date = base_date - timedelta(days=np.random.randint(15, 60))
            due_days = np.random.randint(1, 20)
            amount = np.random.uniform(*amount_range)
            
            # Calculate risk based on due days and amount
            risk_score = InvoiceRiskService._calculate_invoice_risk(due_days, amount)
            
            invoices.append({
                'invoice_id': f'INV{1020 + i}',
                'customer': customers[i % len(customers)],
                'amount': round(amount, 2),
                'issue_date': invoice_date.strftime('%Y-%m-%d'),
                'due_in_days': due_days,
                'risk_score': risk_score,
                'risk_level': InvoiceRiskService._get_risk_level(risk_score)
            })
        
        # Sort by risk score descending
        invoices.sort(key=lambda x: x['risk_score'], reverse=True)
        
        logger.info(f"Generated {len(invoices)} sample invoices")
        
        return invoices
    
    @staticmethod
    def _calculate_invoice_risk(days_until_due: int, amount: float) -> float:
        # Base risk on urgency
        if days_until_due <= 3:
            time_risk = 80
        elif days_until_due <= 7:
            time_risk = 60
        elif days_until_due <= 14:
            time_risk = 40
        else:
            time_risk = 20
        
        # Amount factor (higher amounts = higher risk impact)
        amount_factor = min(20, (amount / 10000) * 10)
        
        # Add some randomness for customer history simulation
        history_factor = np.random.uniform(-10, 10)
        
        risk_score = time_risk + amount_factor + history_factor
        
        return max(0, min(100, risk_score))
    
    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        if risk_score >= 70:
            return 'high'
        elif risk_score >= 40:
            return 'medium'
        else:
            return 'low'
    
    @staticmethod
    def get_overdue_invoices(invoices: List[Dict]) -> List[Dict]:
        high_risk = [inv for inv in invoices if inv['risk_score'] >= 70]
        logger.info(f"Found {len(high_risk)} high-risk invoices")
        return high_risk[:3]  # Top 3
