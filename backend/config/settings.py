import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "E-DERA API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    PROCESSED_DIR: Path = BASE_DIR / "storage" / "processed"
    MODELS_DIR: Path = BASE_DIR / "trained_models"
    
    # API Configuration
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:5500", "*"]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".csv", ".pdf"}
    
    # Model Configuration
    FORECAST_DAYS: int = 30
    MIN_DATA_POINTS: int = 60
    RISK_THRESHOLD_LOW: float = 30.0
    RISK_THRESHOLD_HIGH: float = 60.0
    
    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_PROVIDER: str = "openai"  # or "anthropic"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_MAX_TOKENS: int = 500
    
    # Feature Engineering
    ANOMALY_THRESHOLD: float = 2.5  # Standard deviations
    INVOICE_RISK_FEATURES: list = ["days_overdue", "amount", "customer_history"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
