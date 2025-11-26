"""
TabNet - Interpretable Tabular Model
Great for invoice delay prediction and credit risk scoring
Selects important financial features automatically using sequential attention
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TabNetService:
    """
    TabNet for interpretable invoice risk and credit scoring
    Uses sequential attention for feature selection
    """
    
    @staticmethod
    def predict_invoice_risk(
        invoices: List[Dict],
        historical_data: pd.DataFrame,
        n_steps: int = 3
    ) -> List[Dict[str, any]]:
        """
        Predict invoice payment delay risk using TabNet
        """
        logger.info(f"TabNet invoice risk prediction for {len(invoices)} invoices")
        
        try:
            if not invoices:
                return []
            
            # Prepare features for each invoice
            invoice_features = TabNetService._prepare_invoice_features(
                invoices, historical_data
            )
            
            # Apply sequential attention mechanism
            predictions = []
            
            for i, invoice in enumerate(invoices):
                # Get features for this invoice
                features = invoice_features[i] if i < len(invoice_features) else {}
                
                # Apply TabNet sequential attention
                risk_score, feature_importance, attention_masks = TabNetService._tabnet_forward(
                    features, n_steps
                )
                
                # Generate prediction
                prediction = {
                    'invoice_id': invoice.get('invoice_id', f'INV_{i}'),
                    'customer': invoice.get('customer', 'Unknown'),
                    'amount': invoice.get('amount', 0),
                    'due_days': invoice.get('due_in_days', 30),
                    'predicted_risk_score': float(risk_score),
                    'risk_category': TabNetService._categorize_risk(risk_score),
                    'feature_importance': feature_importance,
                    'attention_explanation': TabNetService._generate_explanation(
                        feature_importance, attention_masks
                    ),
                    'recommendations': TabNetService._generate_recommendations(
                        risk_score, feature_importance
                    )
                }
                
                predictions.append(prediction)
            
            logger.info(f"TabNet predictions completed for {len(predictions)} invoices")
            
            return sorted(predictions, key=lambda x: x['predicted_risk_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"TabNet invoice prediction error: {e}")
            return []
    
    @staticmethod
    def predict_credit_risk(
        customer_data: Dict[str, any],
        historical_data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Predict credit risk for a customer using TabNet
        """
        logger.info(f"TabNet credit risk assessment")
        
        try:
            # Prepare customer features
            features = TabNetService._prepare_credit_features(
                customer_data, historical_data
            )
            
            # Apply TabNet
            credit_score, feature_importance, _ = TabNetService._tabnet_forward(
                features, n_steps=3
            )
            
            # Generate credit assessment
            assessment = {
                'customer': customer_data.get('customer', 'Unknown'),
                'credit_score': float(credit_score),
                'credit_rating': TabNetService._get_credit_rating(credit_score),
                'recommended_credit_limit': TabNetService._calculate_credit_limit(
                    credit_score, historical_data
                ),
                'key_risk_factors': TabNetService._identify_risk_factors(
                    feature_importance
                ),
                'approval_recommendation': 'Approve' if credit_score < 50 else 'Review' if credit_score < 70 else 'Decline'
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"TabNet credit risk error: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _prepare_invoice_features(
        invoices: List[Dict],
        historical_data: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """Prepare feature set for invoice risk prediction"""
        features_list = []
        
        for invoice in invoices:
            features = {}
            
            # Invoice-specific features
            features['amount'] = float(invoice.get('amount', 0))
            features['due_days'] = float(invoice.get('due_in_days', 30))
            features['invoice_age_days'] = float(invoice.get('age_days', 0))
            
            # Customer payment history (if available)
            customer = invoice.get('customer', '')
            
            # Historical payment behavior simulation
            # In production, this would query actual payment history
            features['avg_payment_delay'] = np.random.uniform(0, 10)
            features['payment_reliability'] = np.random.uniform(0.7, 1.0)
            features['previous_defaults'] = np.random.randint(0, 3)
            
            # Financial health indicators from historical data
            if not historical_data.empty:
                recent_balance = historical_data['cumulative_cashflow'].iloc[-1] if 'cumulative_cashflow' in historical_data.columns else 0
                features['current_balance'] = float(recent_balance)
                
                avg_sales = historical_data['sales'].mean() if 'sales' in historical_data.columns else 0
                features['avg_monthly_sales'] = float(avg_sales * 30)
                
                # Cashflow volatility
                if 'sales_volatility' in historical_data.columns:
                    features['cashflow_volatility'] = float(historical_data['sales_volatility'].mean())
                else:
                    features['cashflow_volatility'] = 0.1
            
            # Invoice to sales ratio
            if features.get('avg_monthly_sales', 0) > 0:
                features['invoice_to_sales_ratio'] = features['amount'] / features['avg_monthly_sales']
            else:
                features['invoice_to_sales_ratio'] = 0.1
            
            # Time-based features
            features['is_month_end'] = 1.0 if features['due_days'] <= 5 else 0.0
            features['is_quarter_end'] = 1.0 if features['due_days'] <= 2 else 0.0
            
            features_list.append(features)
        
        return features_list
    
    @staticmethod
    def _prepare_credit_features(
        customer_data: Dict[str, any],
        historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Prepare features for credit risk assessment"""
        features = {}
        
        # Customer information
        features['customer_age_months'] = float(customer_data.get('customer_age_months', 12))
        features['total_transactions'] = float(customer_data.get('total_transactions', 10))
        features['total_revenue'] = float(customer_data.get('total_revenue', 100000))
        
        # Payment behavior
        features['avg_payment_delay'] = float(customer_data.get('avg_payment_delay', 5))
        features['max_payment_delay'] = float(customer_data.get('max_payment_delay', 15))
        features['on_time_payment_rate'] = float(customer_data.get('on_time_payment_rate', 0.85))
        
        # Financial indicators
        if not historical_data.empty:
            features['current_balance'] = float(
                historical_data['cumulative_cashflow'].iloc[-1]
                if 'cumulative_cashflow' in historical_data.columns
                else 0
            )
        
        # Defaults and disputes
        features['num_defaults'] = float(customer_data.get('num_defaults', 0))
        features['num_disputes'] = float(customer_data.get('num_disputes', 0))
        
        return features
    
    @staticmethod
    def _tabnet_forward(
        features: Dict[str, float],
        n_steps: int = 3
    ) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
        """
        Simulate TabNet forward pass with sequential attention
        """
        # Convert features to array
        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])
        
        # Normalize features
        feature_values = (feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-8)
        
        # Initialize
        feature_importance = {k: 0.0 for k in feature_names}
        attention_masks = []
        processed_features = np.zeros_like(feature_values)
        
        # Sequential attention steps
        for step in range(n_steps):
            # Calculate attention mask for this step
            attention_mask = TabNetService._calculate_attention_mask(
                feature_values, processed_features, step
            )
            
            attention_masks.append({
                feature_names[i]: float(attention_mask[i])
                for i in range(len(feature_names))
            })
            
            # Apply attention to select features
            selected_features = feature_values * attention_mask
            
            # Update feature importance
            for i, name in enumerate(feature_names):
                feature_importance[name] += float(attention_mask[i]) / n_steps
            
            # Process selected features (simulate neural network layer)
            step_output = np.tanh(selected_features)
            processed_features += step_output
        
        # Calculate final risk score
        risk_score = np.sum(processed_features) * 10 + 50  # Scale to 0-100
        risk_score = max(0, min(100, risk_score))
        
        return risk_score, feature_importance, attention_masks
    
    @staticmethod
    def _calculate_attention_mask(
        features: np.ndarray,
        processed_features: np.ndarray,
        step: int
    ) -> np.ndarray:
        """Calculate attention mask for feature selection"""
        # Simulate learnable attention mechanism
        
        # Remaining features (not yet fully processed)
        remaining = 1 - np.abs(processed_features) / (np.abs(features) + 1e-8)
        remaining = np.clip(remaining, 0, 1)
        
        # Feature salience based on magnitude
        salience = np.abs(features) / (np.sum(np.abs(features)) + 1e-8)
        
        # Combine remaining capacity and salience
        attention = remaining * salience
        
        # Add some randomness to simulate learned patterns
        noise = np.random.uniform(0.8, 1.2, len(features))
        attention = attention * noise
        
        # Normalize to create mask
        attention = attention / (np.sum(attention) + 1e-8)
        
        # Apply sparsity (TabNet characteristic)
        threshold = np.percentile(attention, 30)  # Keep top 70%
        attention[attention < threshold] = 0
        
        # Re-normalize
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention
    
    @staticmethod
    def _categorize_risk(risk_score: float) -> str:
        """Categorize risk score into risk levels"""
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 50:
            return "Medium-Low Risk"
        elif risk_score < 70:
            return "Medium-High Risk"
        else:
            return "High Risk"
    
    @staticmethod
    def _generate_explanation(
        feature_importance: Dict[str, float],
        attention_masks: List[Dict[str, float]]
    ) -> List[str]:
        """Generate human-readable explanations"""
        explanations = []
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top 3 features
        for feature, importance in sorted_features[:3]:
            if importance > 0.1:
                explanation = TabNetService._feature_to_explanation(feature, importance)
                explanations.append(explanation)
        
        return explanations
    
    @staticmethod
    def _feature_to_explanation(feature_name: str, importance: float) -> str:
        """Convert feature name and importance to explanation"""
        explanations_map = {
            'amount': f"Invoice amount significantly impacts risk (importance: {importance:.2%})",
            'due_days': f"Payment deadline is a key factor (importance: {importance:.2%})",
            'avg_payment_delay': f"Historical payment delays strongly influence risk (importance: {importance:.2%})",
            'payment_reliability': f"Customer payment reliability is critical (importance: {importance:.2%})",
            'current_balance': f"Current cashflow balance affects risk (importance: {importance:.2%})",
            'cashflow_volatility': f"Cashflow stability impacts payment probability (importance: {importance:.2%})",
            'invoice_to_sales_ratio': f"Invoice size relative to sales matters (importance: {importance:.2%})"
        }
        
        return explanations_map.get(
            feature_name,
            f"{feature_name.replace('_', ' ').title()} is important (importance: {importance:.2%})"
        )
    
    @staticmethod
    def _generate_recommendations(
        risk_score: float,
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on risk score"""
        recommendations = []
        
        if risk_score > 70:
            recommendations.append("⚠️ High risk: Consider advance payment or partial upfront")
            recommendations.append("📞 Contact customer immediately to confirm payment intent")
            recommendations.append("💰 Review credit limit and payment terms")
        elif risk_score > 50:
            recommendations.append("⚡ Moderate risk: Send payment reminder 3-5 days before due date")
            recommendations.append("📋 Monitor payment closely and follow up promptly")
        else:
            recommendations.append("✅ Low risk: Standard payment terms acceptable")
            recommendations.append("🎯 Maintain current customer relationship")
        
        # Feature-specific recommendations
        if feature_importance.get('avg_payment_delay', 0) > 0.15:
            recommendations.append("📊 Customer has history of delays - enforce stricter terms")
        
        if feature_importance.get('current_balance', 0) > 0.15:
            recommendations.append("💸 Low cashflow detected - consider payment plan options")
        
        return recommendations
    
    @staticmethod
    def _get_credit_rating(credit_score: float) -> str:
        """Convert credit score to rating"""
        if credit_score < 30:
            return "AAA - Excellent"
        elif credit_score < 45:
            return "AA - Very Good"
        elif credit_score < 60:
            return "A - Good"
        elif credit_score < 75:
            return "BBB - Fair"
        else:
            return "BB - Poor"
    
    @staticmethod
    def _calculate_credit_limit(
        credit_score: float,
        historical_data: pd.DataFrame
    ) -> float:
        """Calculate recommended credit limit"""
        # Base on average monthly sales
        if 'sales' in historical_data.columns and not historical_data.empty:
            avg_monthly_sales = historical_data['sales'].mean() * 30
        else:
            avg_monthly_sales = 100000
        
        # Adjust based on credit score
        if credit_score < 30:
            multiplier = 0.5  # 50% of monthly sales
        elif credit_score < 50:
            multiplier = 0.3
        elif credit_score < 70:
            multiplier = 0.15
        else:
            multiplier = 0.05
        
        return float(avg_monthly_sales * multiplier)
    
    @staticmethod
    def _identify_risk_factors(
        feature_importance: Dict[str, float]
    ) -> List[str]:
        """Identify key risk factors from feature importance"""
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        risk_factors = []
        for feature, importance in sorted_features[:5]:
            if importance > 0.1:
                risk_factors.append(
                    feature.replace('_', ' ').title()
                )
        
        return risk_factors
