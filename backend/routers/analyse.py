from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import pandas as pd

from backend.utils.logger import get_logger
from backend.config.settings import settings
from backend.services.ingest_service import DataIngestService
from backend.services.feature_engineering import FeatureEngineeringService
from backend.services.forecast_service import ForecastService
from backend.services.anomaly_service import AnomalyDetectionService
from backend.services.invoice_service import InvoiceRiskService
from backend.services.risk_service import RiskScoringService

# Advanced ML models
from backend.services.advanced_forecast_service import (
    DAGRUForecastService,
    TemporalFusionTransformerService,
    NBEATSService,
    DeepARService
)
from backend.services.advanced_anomaly_service import (
    DeepDenoisingAutoencoder,
    AdvancedIsolationForest,
    GraphAttentionNetwork
)
from backend.services.tabnet_service import TabNetService
from backend.services.ensemble_meta_model import EnsembleMetaRiskModel

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analysis"])


class AnalyseRequest(BaseModel):
    sales_file: Optional[str] = None
    expense_file: Optional[str] = None
    use_latest: bool = True


@router.post("/analyse")
async def analyse_data(request: AnalyseRequest):
    
    logger.info("Starting financial analysis")
    
    try:
        # Load data files
        if request.use_latest:
            # Find latest processed files
            sales_files = list(settings.PROCESSED_DIR.glob("processed_*sales*.csv"))
            expense_files = list(settings.PROCESSED_DIR.glob("processed_*expense*.csv"))
            
            if not sales_files and not expense_files:
                raise HTTPException(status_code=404, detail="No processed data files found. Please upload data first.")
            
            sales_path = max(sales_files, key=lambda f: f.stat().st_mtime) if sales_files else None
            expense_path = max(expense_files, key=lambda f: f.stat().st_mtime) if expense_files else None
        else:
            sales_path = Path(request.sales_file) if request.sales_file else None
            expense_path = Path(request.expense_file) if request.expense_file else None
        
        # Load DataFrames
        sales_df = pd.read_csv(sales_path) if sales_path else None
        expense_df = pd.read_csv(expense_path) if expense_path else None
        
        if sales_df is not None:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
        if expense_df is not None:
            expense_df['date'] = pd.to_datetime(expense_df['date'])
        
        logger.info(f"Loaded sales: {len(sales_df) if sales_df is not None else 0} records, expenses: {len(expense_df) if expense_df is not None else 0} records")
        
        # Feature engineering - pass sales and expense dataframes
        timeline_with_features = FeatureEngineeringService.create_features(
            sales_df if sales_df is not None else pd.DataFrame(),
            expense_df if expense_df is not None else pd.DataFrame()
        )
        
        logger.info(f"Created timeline with {len(timeline_with_features)} days and {len(timeline_with_features.columns)} features")
        
        # 1. Cashflow Forecast
        forecast_result = ForecastService.simple_forecast(timeline_with_features, forecast_days=settings.FORECAST_DAYS)
        
        # 2. Anomaly Detection
        anomalies = AnomalyDetectionService.detect_anomalies(timeline_with_features, expense_df if expense_df is not None else pd.DataFrame())
        
        # 3. Invoice Risk Analysis
        invoices = InvoiceRiskService.generate_sample_invoices()
        
        # 4. Risk Scoring
        current_balance = timeline_with_features['cumulative_cashflow'].iloc[-1]
        forecast_array = [d['cashflow'] for d in forecast_result['daily_forecast']]
        
        risk_analysis = RiskScoringService.calculate_composite_risk_score(
            cashflow_forecast=forecast_array,
            anomalies=anomalies,
            invoice_risks=invoices,
            current_balance=current_balance
        )
        
        logger.info(f"Analysis completed. Risk Score: {risk_analysis['overall_score']}")
        
        # Convert numpy types to Python native types for JSON serialization
        import numpy as np
        
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        response_data = {
            "status": "success",
            "analysis": {
                "risk": risk_analysis,
                "forecast": forecast_result,
                "anomalies": anomalies[:10],  # Top 10 anomalies
                "invoices": invoices[:5],  # Top 5 risky invoices
                "summary": {
                    "data_points": int(len(timeline_with_features)),
                    "date_range": {
                        "start": timeline_with_features['date'].min().strftime('%Y-%m-%d'),
                        "end": timeline_with_features['date'].max().strftime('%Y-%m-%d')
                    },
                    "total_sales": float(timeline_with_features['sales'].sum()),
                    "total_expenses": float(timeline_with_features['expenses'].sum()),
                    "current_balance": float(current_balance),
                    "avg_daily_sales": float(timeline_with_features['sales'].mean()),
                    "avg_daily_expenses": float(timeline_with_features['expenses'].mean()),
                    "daily_data": [
                        {
                            "date": row['date'].strftime('%Y-%m-%d'),
                            "sales": float(row['sales']),
                            "expenses": float(row['expenses']),
                            "balance": float(row['cumulative_cashflow'])
                        }
                        for _, row in timeline_with_features.tail(30).iterrows()
                    ],
                    "expense_by_category": (
                        expense_df.groupby('category')['amount'].sum().to_dict()
                        if expense_df is not None and 'category' in expense_df.columns
                        else {}
                    )
                }
            }
        }
        
        # Convert all numpy types
        return convert_to_native(response_data)
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="Required data files not found")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/analyse/status")
async def get_analysis_status():
    
    try:
        # Check for processed files
        processed_files = list(settings.PROCESSED_DIR.glob("processed_*.csv"))
        
        return {
            "status": "success",
            "data_available": len(processed_files) > 0,
            "processed_files": len(processed_files),
            "ready_for_analysis": len(processed_files) >= 1
        }
    
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyse/advanced")
async def analyse_data_advanced(request: AnalyseRequest):
    """
    Advanced analysis using all 9 ML/AI techniques:
    - DA-GRU, TFT, N-BEATS, DeepAR for forecasting
    - Deep Autoencoder, Isolation Forest, GAT for anomaly detection
    - TabNet for invoice risk
    - Ensemble Meta-Model for final E-Risk Score
    """
    
    logger.info("Starting ADVANCED financial analysis with 9 ML models")
    
    try:
        # Load data files (same as regular analysis)
        if request.use_latest:
            sales_files = list(settings.PROCESSED_DIR.glob("processed_*sales*.csv"))
            expense_files = list(settings.PROCESSED_DIR.glob("processed_*expense*.csv"))
            
            if not sales_files and not expense_files:
                raise HTTPException(status_code=404, detail="No processed data files found. Please upload data first.")
            
            sales_path = max(sales_files, key=lambda f: f.stat().st_mtime) if sales_files else None
            expense_path = max(expense_files, key=lambda f: f.stat().st_mtime) if expense_files else None
        else:
            sales_path = Path(request.sales_file) if request.sales_file else None
            expense_path = Path(request.expense_file) if request.expense_file else None
        
        # Load DataFrames
        sales_df = pd.read_csv(sales_path) if sales_path else None
        expense_df = pd.read_csv(expense_path) if expense_path else None
        
        if sales_df is not None:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
        if expense_df is not None:
            expense_df['date'] = pd.to_datetime(expense_df['date'])
        
        logger.info(f"Loaded sales: {len(sales_df) if sales_df is not None else 0} records, expenses: {len(expense_df) if expense_df is not None else 0} records")
        
        # Feature engineering
        timeline_with_features = FeatureEngineeringService.create_features(
            sales_df if sales_df is not None else pd.DataFrame(),
            expense_df if expense_df is not None else pd.DataFrame()
        )
        
        logger.info(f"Created timeline with {len(timeline_with_features)} days and {len(timeline_with_features.columns)} features")
        
        # ===== TRADITIONAL MODELS (for baseline) =====
        traditional_forecast = ForecastService.simple_forecast(timeline_with_features, forecast_days=30)
        traditional_anomalies = AnomalyDetectionService.detect_anomalies(timeline_with_features, expense_df if expense_df is not None else pd.DataFrame())
        traditional_invoices = InvoiceRiskService.generate_sample_invoices()
        
        current_balance = timeline_with_features['cumulative_cashflow'].iloc[-1]
        forecast_array = [d['cashflow'] for d in traditional_forecast['daily_forecast']]
        
        traditional_risk = RiskScoringService.calculate_composite_risk_score(
            cashflow_forecast=forecast_array,
            anomalies=traditional_anomalies,
            invoice_risks=traditional_invoices,
            current_balance=current_balance
        )
        
        # ===== ADVANCED FORECASTING MODELS =====
        logger.info("Running advanced forecasting models...")
        
        # 1. DA-GRU (Dual Attention GRU)
        dagru_forecast = DAGRUForecastService.forecast(timeline_with_features, forecast_days=30)
        
        # 2. Temporal Fusion Transformer
        tft_forecast = TemporalFusionTransformerService.forecast(timeline_with_features, forecast_days=30)
        
        # 3. N-BEATS
        nbeats_forecast = NBEATSService.forecast(timeline_with_features, forecast_days=30)
        
        # 4. DeepAR (Probabilistic)
        deepar_forecast = DeepARService.forecast(timeline_with_features, forecast_days=30)
        
        # ===== ADVANCED ANOMALY DETECTION =====
        logger.info("Running advanced anomaly detection models...")
        
        # 5. Deep Denoising Autoencoder
        autoencoder_anomalies = DeepDenoisingAutoencoder.detect_anomalies(
            timeline_with_features,
            expense_df if expense_df is not None else pd.DataFrame()
        )
        
        # 6. Isolation Forest
        isolation_forest_anomalies = AdvancedIsolationForest.detect_anomalies(
            timeline_with_features,
            expense_df if expense_df is not None else pd.DataFrame()
        )
        
        # 7. Graph Attention Networks
        gat_anomalies = GraphAttentionNetwork.detect_relational_anomalies(
            expense_df if expense_df is not None else pd.DataFrame(),
            timeline_with_features
        )
        
        # ===== TABNET INVOICE RISK =====
        logger.info("Running TabNet invoice risk prediction...")
        
        # 8. TabNet for invoice risk
        tabnet_invoice_risks = TabNetService.predict_invoice_risk(
            traditional_invoices,
            timeline_with_features
        )
        
        # ===== ENSEMBLE META-MODEL =====
        logger.info("Calculating Ensemble Meta-Risk Score...")
        
        # 9. Gradient-Boosted Ensemble Meta-Risk Model
        ensemble_risk = EnsembleMetaRiskModel.calculate_ensemble_risk_score(
            dagru_forecast=dagru_forecast,
            tft_forecast=tft_forecast,
            nbeats_forecast=nbeats_forecast,
            deepar_forecast=deepar_forecast,
            autoencoder_anomalies=autoencoder_anomalies,
            isolation_forest_anomalies=isolation_forest_anomalies,
            gat_anomalies=gat_anomalies,
            tabnet_invoice_risks=tabnet_invoice_risks,
            traditional_risk=traditional_risk,
            historical_data=timeline_with_features
        )
        
        logger.info(f"Advanced Analysis completed. E-Risk Score: {ensemble_risk.get('e_risk_score', 0)}")
        
        # Convert numpy types to Python native types
        import numpy as np
        
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Prepare comprehensive response
        response_data = {
            "status": "success",
            "analysis_type": "advanced",
            "models_used": [
                "DA-GRU", "Temporal Fusion Transformer", "N-BEATS", "DeepAR",
                "Deep Autoencoder", "Isolation Forest", "Graph Attention Network",
                "TabNet", "Ensemble Meta-Model"
            ],
            "ensemble_risk": ensemble_risk,
            "traditional_risk": traditional_risk,
            "advanced_forecasts": {
                "dagru": dagru_forecast,
                "tft": tft_forecast,
                "nbeats": nbeats_forecast,
                "deepar": deepar_forecast,
                "traditional": traditional_forecast
            },
            "advanced_anomalies": {
                "deep_autoencoder": autoencoder_anomalies[:5],
                "isolation_forest": isolation_forest_anomalies[:5],
                "graph_attention": gat_anomalies[:5],
                "traditional": traditional_anomalies[:5]
            },
            "invoice_analysis": {
                "tabnet_predictions": tabnet_invoice_risks[:5],
                "traditional_risks": traditional_invoices[:5]
            },
            "summary": {
                "data_points": int(len(timeline_with_features)),
                "date_range": {
                    "start": timeline_with_features['date'].min().strftime('%Y-%m-%d'),
                    "end": timeline_with_features['date'].max().strftime('%Y-%m-%d')
                },
                "total_sales": float(timeline_with_features['sales'].sum()),
                "total_expenses": float(timeline_with_features['expenses'].sum()),
                "current_balance": float(current_balance),
                "avg_daily_sales": float(timeline_with_features['sales'].mean()),
                "avg_daily_expenses": float(timeline_with_features['expenses'].mean()),
                "daily_data": [
                    {
                        "date": row['date'].strftime('%Y-%m-%d'),
                        "sales": float(row['sales']),
                        "expenses": float(row['expenses']),
                        "balance": float(row['cumulative_cashflow'])
                    }
                    for _, row in timeline_with_features.tail(30).iterrows()
                ],
                "expense_by_category": (
                    expense_df.groupby('category')['amount'].sum().to_dict()
                    if expense_df is not None and 'category' in expense_df.columns
                    else {}
                )
            }
        }
        
        # Convert all numpy types
        return convert_to_native(response_data)
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="Required data files not found")
    
    except Exception as e:
        logger.error(f"Advanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced analysis failed: {str(e)}")
