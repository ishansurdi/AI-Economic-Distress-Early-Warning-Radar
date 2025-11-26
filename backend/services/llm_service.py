import os
from typing import Dict, List
from openai import OpenAI
from anthropic import Anthropic
from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)

class LLMService:
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients if API keys are available
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your-openai-api-key":
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        if settings.ANTHROPIC_API_KEY and settings.ANTHROPIC_API_KEY != "your-anthropic-api-key":
            try:
                self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
    
    def generate_recommendations(
        self,
        risk_score: float,
        cashflow_summary: Dict,
        anomalies: List[Dict],
        invoice_data: List[Dict]
    ) -> List[Dict[str, str]]:
        
        logger.info("Generating AI recommendations")
        
        # Build context from analysis
        context = self._build_context(risk_score, cashflow_summary, anomalies, invoice_data)
        
        # Try to use LLM if available
        if self.openai_client:
            return self._generate_with_openai(context)
        elif self.anthropic_client:
            return self._generate_with_anthropic(context)
        else:
            # Fallback to rule-based recommendations
            logger.info("Using rule-based recommendations (no LLM API key)")
            return self._generate_rule_based(risk_score, cashflow_summary, anomalies, invoice_data)
    
    def _build_context(
        self,
        risk_score: float,
        cashflow_summary: Dict,
        anomalies: List[Dict],
        invoice_data: List[Dict]
    ) -> str:
        
        context = f"""Financial Analysis Summary:
        
Risk Score: {risk_score}/100

Cashflow Forecast (30 days):
- Total Sales: ₹{cashflow_summary.get('total_sales', 0):,.2f}
- Total Expenses: ₹{cashflow_summary.get('total_expenses', 0):,.2f}
- Net Cashflow: ₹{cashflow_summary.get('net_cashflow', 0):,.2f}
- Minimum Balance: ₹{cashflow_summary.get('min_balance', 0):,.2f} on day {cashflow_summary.get('min_balance_day', 0)}

Anomalies Detected: {len(anomalies)}
"""
        
        if anomalies:
            context += "\nTop Anomalies:\n"
            for ano in anomalies[:3]:
                context += f"- {ano.get('type')}: {ano.get('description')}\n"
        
        if invoice_data:
            high_risk_invoices = [inv for inv in invoice_data if inv.get('risk_score', 0) >= 70]
            context += f"\nHigh-Risk Invoices: {len(high_risk_invoices)}\n"
        
        return context
    
    def _generate_with_openai(self, context: str) -> List[Dict[str, str]]:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial advisor helping a small business. Provide 4 specific, actionable recommendations based on the analysis. Each recommendation should be 1-2 sentences."
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nProvide 4 actionable recommendations."
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            recommendations_text = response.choices[0].message.content
            return self._parse_recommendations(recommendations_text)
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_rule_based(0, {}, [], [])
    
    def _generate_with_anthropic(self, context: str) -> List[Dict[str, str]]:
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"{context}\n\nAs a financial advisor, provide 4 specific actionable recommendations."
                    }
                ]
            )
            
            recommendations_text = message.content[0].text
            return self._parse_recommendations(recommendations_text)
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._generate_rule_based(0, {}, [], [])
    
    def _parse_recommendations(self, text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        recommendations = []
        for line in lines:
            # Remove numbering
            clean_line = line.lstrip('0123456789.-)• ')
            if len(clean_line) > 20:  # Valid recommendation
                recommendations.append({
                    'title': clean_line[:50] + '...' if len(clean_line) > 50 else clean_line,
                    'description': clean_line
                })
        
        # Ensure exactly 4 recommendations
        while len(recommendations) < 4:
            recommendations.extend(self._generate_rule_based(0, {}, [], []))
        
        return recommendations[:4]
    
    def _generate_rule_based(
        self,
        risk_score: float,
        cashflow_summary: Dict,
        anomalies: List[Dict],
        invoice_data: List[Dict]
    ) -> List[Dict[str, str]]:
        
        recommendations = []
        
        # Cashflow-based recommendations
        min_balance = cashflow_summary.get('min_balance', 0)
        if min_balance < 0:
            recommendations.append({
                'title': 'Secure short-term credit line',
                'description': f'Your projected minimum balance is ₹{min_balance:,.2f}. Arrange credit facility or negotiate payment terms.'
            })
        
        # Anomaly-based recommendations
        if len(anomalies) > 0:
            expense_anomalies = [a for a in anomalies if a.get('type') == 'expense_spike']
            if expense_anomalies:
                recommendations.append({
                    'title': 'Review recent expense spikes',
                    'description': f'Detected {len(expense_anomalies)} unusual expense patterns. Review and optimize spending categories.'
                })
        
        # Invoice-based recommendations
        if invoice_data:
            high_risk = [inv for inv in invoice_data if inv.get('risk_score', 0) >= 70]
            if high_risk:
                recommendations.append({
                    'title': 'Follow up on pending invoices',
                    'description': f'{len(high_risk)} invoices require immediate attention. Implement automated payment reminders.'
                })
        
        # General recommendations
        default_recommendations = [
            {
                'title': 'Build emergency reserve',
                'description': 'Maintain 3-6 months of operating expenses as cash reserve for business continuity.'
            },
            {
                'title': 'Diversify revenue streams',
                'description': 'Reduce dependency on single customers or products to minimize revenue volatility.'
            },
            {
                'title': 'Automate expense tracking',
                'description': 'Implement real-time expense monitoring to catch anomalies early and control costs.'
            },
            {
                'title': 'Negotiate better payment terms',
                'description': 'Work with suppliers for extended payment terms and customers for advance payments.'
            }
        ]
        
        # Fill up to 4 recommendations
        for rec in default_recommendations:
            if len(recommendations) >= 4:
                break
            if rec not in recommendations:
                recommendations.append(rec)
        
        return recommendations[:4]
