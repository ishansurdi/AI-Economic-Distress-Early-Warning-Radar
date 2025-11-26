import sys
from loguru import logger
from pathlib import Path

# Remove default logger
logger.remove()

# Console logger
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# File logger
log_path = Path(__file__).resolve().parent.parent / "logs"
log_path.mkdir(exist_ok=True)

logger.add(
    log_path / "edera_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    level="DEBUG"
)

def get_logger(name: str):
    return logger.bind(name=name)
