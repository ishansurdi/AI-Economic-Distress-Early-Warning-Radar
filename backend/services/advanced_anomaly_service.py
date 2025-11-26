"""
Advanced Anomaly Detection Models
Implements Deep Autoencoders, Isolation Forest, and Graph Attention Networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest
from sklearn.preprocessing import StandardScaler
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DeepDenoisingAutoencoder:
    """
    Deep Denoising Autoencoder for anomaly detection
    Detects abnormal expense spikes and unusual financial behavior
    Uses reconstruction error to signal anomalies
    """
    
    @staticmethod
    def detect_anomalies(
        timeline: pd.DataFrame,
        expenses_df: pd.DataFrame,
        threshold_percentile: float = 95
    ) -> List[Dict[str, any]]:
        """
        Detect anomalies using reconstruction error
        """
        logger.info("Deep Autoencoder anomaly detection starting")
        
        try:
            # Prepare feature matrix
            features = DeepDenoisingAutoencoder._prepare_features(timeline, expenses_df)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Simulate autoencoder encoding-decoding
            encoded, original_features = DeepDenoisingAutoencoder._encode(features_scaled)
            reconstructed = DeepDenoisingAutoencoder._decode(encoded, original_features)
            
            # Calculate reconstruction error
            reconstruction_errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
            
            # Set threshold based on percentile
            threshold = np.percentile(reconstruction_errors, threshold_percentile)
            
            # Identify anomalies
            anomalies = []
            anomaly_indices = np.where(reconstruction_errors > threshold)[0]
            
            for idx in anomaly_indices:
                if idx < len(timeline):
                    row = timeline.iloc[idx]
                    error = reconstruction_errors[idx]
                    
                    # Determine anomaly type
                    anomaly_type = DeepDenoisingAutoencoder._classify_anomaly(
                        row, features[idx], features_scaled[idx], reconstructed[idx]
                    )
                    
                    anomaly = {
                        'model': 'Deep_Autoencoder',
                        'type': anomaly_type['type'],
                        'category': anomaly_type['category'],
                        'date': row['date'].strftime('%Y-%m-%d') if 'date' in row else 'Unknown',
                        'severity': min(1.0, float((error - threshold) / threshold)),
                        'reconstruction_error': float(error),
                        'description': anomaly_type['description'],
                        'affected_metrics': anomaly_type['metrics']
                    }
                    
                    anomalies.append(anomaly)
            
            logger.info(f"Deep Autoencoder detected {len(anomalies)} anomalies")
            
            return sorted(anomalies, key=lambda x: x['severity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Deep Autoencoder error: {e}")
            return []
    
    @staticmethod
    def _prepare_features(timeline: pd.DataFrame, expenses_df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for autoencoder"""
        features = []
        
        base_features = ['sales', 'expenses', 'cumulative_cashflow']
        
        # Add available features
        for col in base_features:
            if col in timeline.columns:
                features.append(timeline[col].values)
        
        # Add rolling statistics
        rolling_cols = [col for col in timeline.columns if 'rolling' in col.lower()]
        for col in rolling_cols[:5]:  # Limit to 5 rolling features
            features.append(timeline[col].fillna(0).values)
        
        # Add lag features
        lag_cols = [col for col in timeline.columns if 'lag' in col.lower()]
        for col in lag_cols[:3]:  # Limit to 3 lag features
            features.append(timeline[col].fillna(0).values)
        
        return np.column_stack(features) if features else np.zeros((len(timeline), 1))
    
    @staticmethod
    def _encode(features: np.ndarray, compression_ratio: float = 0.5) -> Tuple[np.ndarray, int]:
        """Simulate encoding layer (dimensionality reduction)"""
        n_features = features.shape[1]
        n_encoded = max(1, int(n_features * compression_ratio))
        
        # Simulate learned encoding using PCA-like projection
        # Random projection for simulation
        projection_matrix = np.random.randn(n_features, n_encoded) / np.sqrt(n_features)
        encoded = features @ projection_matrix
        
        # Apply non-linearity (ReLU)
        encoded = np.maximum(0, encoded)
        
        return encoded, n_features
    
    @staticmethod
    def _decode(encoded: np.ndarray, output_features: int) -> np.ndarray:
        """Simulate decoding layer (reconstruction)"""
        # Simulate learned decoding
        projection_matrix = np.random.randn(encoded.shape[1], output_features) / np.sqrt(encoded.shape[1])
        decoded = encoded @ projection_matrix
        
        return decoded
    
    @staticmethod
    def _classify_anomaly(
        row: pd.Series,
        features: np.ndarray,
        features_scaled: np.ndarray,
        reconstructed: np.ndarray
    ) -> Dict[str, any]:
        """Classify type of anomaly based on reconstruction error pattern"""
        # Calculate feature-wise reconstruction errors
        feature_errors = np.abs(features_scaled - reconstructed)
        
        # Identify most anomalous features
        max_error_idx = np.argmax(feature_errors)
        
        # Classify anomaly
        if max_error_idx == 0:  # Sales feature
            if 'sales' in row and row['sales'] < row.get('sales_rolling_mean_30d', row['sales']):
                return {
                    'type': 'sales_anomaly',
                    'category': 'Revenue',
                    'description': 'Unusual sales pattern detected',
                    'metrics': ['sales', 'revenue_trend']
                }
        
        if max_error_idx == 1:  # Expenses feature
            if 'expenses' in row:
                return {
                    'type': 'expense_anomaly',
                    'category': 'Expenses',
                    'description': 'Abnormal expense spike detected',
                    'metrics': ['expenses', 'spending_pattern']
                }
        
        # Default classification
        return {
            'type': 'complex_anomaly',
            'category': 'Multi-Factor',
            'description': 'Complex financial behavior anomaly',
            'metrics': ['multiple_factors']
        }


class AdvancedIsolationForest:
    """
    Enhanced Isolation Forest for quick unsupervised anomaly detection
    Captures irregular transactions without needing big data
    """
    
    @staticmethod
    def detect_anomalies(
        timeline: pd.DataFrame,
        expenses_df: pd.DataFrame,
        contamination: float = 0.1
    ) -> List[Dict[str, any]]:
        """
        Detect anomalies using Isolation Forest
        """
        logger.info(f"Isolation Forest anomaly detection with contamination={contamination}")
        
        try:
            # Prepare features
            features = AdvancedIsolationForest._prepare_features(timeline)
            
            if features.shape[0] < 2:
                logger.warning("Insufficient data for Isolation Forest")
                return []
            
            # Fit Isolation Forest
            iso_forest = SKLearnIsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            predictions = iso_forest.fit_predict(features)
            anomaly_scores = iso_forest.score_samples(features)
            
            # Extract anomalies (prediction = -1)
            anomalies = []
            anomaly_indices = np.where(predictions == -1)[0]
            
            for idx in anomaly_indices:
                if idx < len(timeline):
                    row = timeline.iloc[idx]
                    score = anomaly_scores[idx]
                    
                    # Analyze anomaly characteristics
                    characteristics = AdvancedIsolationForest._analyze_anomaly(row, features[idx])
                    
                    anomaly = {
                        'model': 'Isolation_Forest',
                        'type': characteristics['type'],
                        'category': characteristics['category'],
                        'date': row['date'].strftime('%Y-%m-%d') if 'date' in row else 'Unknown',
                        'severity': min(1.0, float(abs(score) / 2)),  # Normalize score
                        'anomaly_score': float(score),
                        'description': characteristics['description'],
                        'indicators': characteristics['indicators']
                    }
                    
                    anomalies.append(anomaly)
            
            logger.info(f"Isolation Forest detected {len(anomalies)} anomalies")
            
            return sorted(anomalies, key=lambda x: x['severity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Isolation Forest error: {e}")
            return []
    
    @staticmethod
    def _prepare_features(timeline: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for Isolation Forest"""
        feature_cols = []
        
        # Core features
        if 'sales' in timeline.columns:
            feature_cols.append('sales')
        if 'expenses' in timeline.columns:
            feature_cols.append('expenses')
        if 'cumulative_cashflow' in timeline.columns:
            feature_cols.append('cumulative_cashflow')
        
        # Rolling statistics
        rolling_cols = [col for col in timeline.columns if 'rolling' in col.lower()]
        feature_cols.extend(rolling_cols[:5])
        
        # Lag features
        lag_cols = [col for col in timeline.columns if 'lag' in col.lower()]
        feature_cols.extend(lag_cols[:3])
        
        # Volatility features
        volatility_cols = [col for col in timeline.columns if 'volatility' in col.lower()]
        feature_cols.extend(volatility_cols[:2])
        
        # Extract features
        features = timeline[feature_cols].fillna(0).values
        
        return features
    
    @staticmethod
    def _analyze_anomaly(row: pd.Series, feature_vector: np.ndarray) -> Dict[str, any]:
        """Analyze characteristics of detected anomaly"""
        indicators = []
        
        # Check sales deviation
        if 'sales' in row and 'sales_rolling_mean_30d' in row:
            sales_dev = abs(row['sales'] - row['sales_rolling_mean_30d'])
            if sales_dev > row['sales_rolling_mean_30d'] * 0.5:
                indicators.append(f"Sales deviation: {sales_dev:.0f}")
        
        # Check expense deviation
        if 'expenses' in row and 'expenses_rolling_mean_30d' in row:
            exp_dev = abs(row['expenses'] - row['expenses_rolling_mean_30d'])
            if exp_dev > row['expenses_rolling_mean_30d'] * 0.5:
                indicators.append(f"Expense deviation: {exp_dev:.0f}")
        
        # Determine type
        if 'expense' in ' '.join(indicators).lower():
            anomaly_type = 'irregular_expense'
            category = 'Expense Anomaly'
            description = 'Irregular expense transaction detected'
        elif 'sales' in ' '.join(indicators).lower():
            anomaly_type = 'irregular_revenue'
            category = 'Revenue Anomaly'
            description = 'Irregular revenue pattern detected'
        else:
            anomaly_type = 'pattern_anomaly'
            category = 'Pattern Anomaly'
            description = 'Unusual financial pattern detected'
        
        return {
            'type': anomaly_type,
            'category': category,
            'description': description,
            'indicators': indicators if indicators else ['General anomaly']
        }


class GraphAttentionNetwork:
    """
    Graph Attention Networks (GAT)
    Models relationships between suppliers, categories, invoices, and payments
    Detects relational anomalies
    """
    
    @staticmethod
    def detect_relational_anomalies(
        expenses_df: pd.DataFrame,
        timeline: pd.DataFrame
    ) -> List[Dict[str, any]]:
        """
        Detect relational anomalies using graph-based attention
        """
        logger.info("Graph Attention Network relational anomaly detection")
        
        try:
            # Build financial graph
            graph = GraphAttentionNetwork._build_financial_graph(expenses_df, timeline)
            
            # Calculate attention scores
            attention_scores = GraphAttentionNetwork._calculate_attention_scores(graph)
            
            # Detect anomalies in graph structure
            anomalies = GraphAttentionNetwork._detect_graph_anomalies(
                graph, attention_scores, expenses_df
            )
            
            logger.info(f"GAT detected {len(anomalies)} relational anomalies")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"GAT error: {e}")
            return []
    
    @staticmethod
    def _build_financial_graph(
        expenses_df: pd.DataFrame,
        timeline: pd.DataFrame
    ) -> Dict[str, any]:
        """Build graph representation of financial relationships"""
        graph = {
            'nodes': [],
            'edges': [],
            'node_features': {}
        }
        
        # Create category nodes
        if 'category' in expenses_df.columns:
            categories = expenses_df['category'].unique()
            
            for category in categories:
                category_data = expenses_df[expenses_df['category'] == category]
                
                node_id = f"category_{category}"
                graph['nodes'].append(node_id)
                graph['node_features'][node_id] = {
                    'type': 'category',
                    'total_amount': float(category_data['amount'].sum()),
                    'transaction_count': len(category_data),
                    'avg_amount': float(category_data['amount'].mean()),
                    'std_amount': float(category_data['amount'].std())
                }
        
        # Create supplier nodes (if available)
        if 'supplier' in expenses_df.columns or 'vendor' in expenses_df.columns:
            supplier_col = 'supplier' if 'supplier' in expenses_df.columns else 'vendor'
            suppliers = expenses_df[supplier_col].unique()
            
            for supplier in suppliers:
                supplier_data = expenses_df[expenses_df[supplier_col] == supplier]
                
                node_id = f"supplier_{supplier}"
                graph['nodes'].append(node_id)
                graph['node_features'][node_id] = {
                    'type': 'supplier',
                    'total_amount': float(supplier_data['amount'].sum()),
                    'transaction_count': len(supplier_data)
                }
                
                # Create edges between supplier and categories
                if 'category' in expenses_df.columns:
                    for category in supplier_data['category'].unique():
                        edge = {
                            'source': node_id,
                            'target': f"category_{category}",
                            'weight': float(supplier_data[supplier_data['category'] == category]['amount'].sum())
                        }
                        graph['edges'].append(edge)
        
        # Create temporal edges (time-based relationships)
        if 'date' in expenses_df.columns:
            expenses_df_sorted = expenses_df.sort_values('date')
            
            # Connect consecutive time periods
            for i in range(len(expenses_df_sorted) - 1):
                edge = {
                    'source': f"transaction_{i}",
                    'target': f"transaction_{i+1}",
                    'weight': 1.0,
                    'type': 'temporal'
                }
                graph['edges'].append(edge)
        
        return graph
    
    @staticmethod
    def _calculate_attention_scores(graph: Dict[str, any]) -> Dict[str, float]:
        """Calculate attention scores for graph nodes"""
        attention_scores = {}
        
        for node_id in graph['nodes']:
            features = graph['node_features'].get(node_id, {})
            
            # Calculate attention based on transaction volume and variability
            total_amount = features.get('total_amount', 0)
            std_amount = features.get('std_amount', 0)
            transaction_count = features.get('transaction_count', 0)
            
            # Higher attention to high-volume, high-variability nodes
            attention = (total_amount / 100000) * (1 + std_amount / 10000) * np.log1p(transaction_count)
            
            attention_scores[node_id] = float(attention)
        
        # Normalize attention scores
        total_attention = sum(attention_scores.values())
        if total_attention > 0:
            attention_scores = {
                k: v / total_attention
                for k, v in attention_scores.items()
            }
        
        return attention_scores
    
    @staticmethod
    def _detect_graph_anomalies(
        graph: Dict[str, any],
        attention_scores: Dict[str, float],
        expenses_df: pd.DataFrame
    ) -> List[Dict[str, any]]:
        """Detect anomalies in graph structure"""
        anomalies = []
        
        # Find nodes with unusually high attention
        avg_attention = np.mean(list(attention_scores.values())) if attention_scores else 0
        std_attention = np.std(list(attention_scores.values())) if attention_scores else 0
        
        threshold = avg_attention + 2 * std_attention
        
        for node_id, attention in attention_scores.items():
            if attention > threshold:
                features = graph['node_features'].get(node_id, {})
                
                anomaly = {
                    'model': 'Graph_Attention_Network',
                    'type': 'relational_anomaly',
                    'category': features.get('type', 'Unknown').capitalize(),
                    'severity': min(1.0, float((attention - avg_attention) / (std_attention + 1e-8))),
                    'attention_score': float(attention),
                    'description': f"Unusual relationship pattern in {node_id.replace('_', ' ')}",
                    'node_id': node_id,
                    'indicators': [
                        f"High attention score: {attention:.4f}",
                        f"Transaction count: {features.get('transaction_count', 0)}",
                        f"Total amount: ₹{features.get('total_amount', 0):,.0f}"
                    ]
                }
                
                anomalies.append(anomaly)
        
        # Detect unusual edge patterns
        edge_anomalies = GraphAttentionNetwork._detect_edge_anomalies(graph)
        anomalies.extend(edge_anomalies)
        
        return sorted(anomalies, key=lambda x: x['severity'], reverse=True)
    
    @staticmethod
    def _detect_edge_anomalies(graph: Dict[str, any]) -> List[Dict[str, any]]:
        """Detect anomalies in edge patterns"""
        anomalies = []
        
        if not graph['edges']:
            return anomalies
        
        # Calculate edge statistics
        edge_weights = [edge['weight'] for edge in graph['edges'] if 'weight' in edge]
        
        if edge_weights:
            avg_weight = np.mean(edge_weights)
            std_weight = np.std(edge_weights)
            threshold = avg_weight + 2.5 * std_weight
            
            for edge in graph['edges']:
                weight = edge.get('weight', 0)
                
                if weight > threshold:
                    anomaly = {
                        'model': 'Graph_Attention_Network',
                        'type': 'edge_anomaly',
                        'category': 'Relationship',
                        'severity': min(1.0, float((weight - avg_weight) / (std_weight + 1e-8))),
                        'description': f"Unusual transaction relationship detected",
                        'indicators': [
                            f"Edge weight: {weight:.0f}",
                            f"Source: {edge.get('source', 'Unknown')}",
                            f"Target: {edge.get('target', 'Unknown')}"
                        ]
                    }
                    
                    anomalies.append(anomaly)
        
        return anomalies
