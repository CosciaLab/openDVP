# src/opendvp/logger.py
from loguru import logger
import sys

# Remove default Loguru handler
logger.remove()

# Add custom handler
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SS}</green> | <level>{level}</level> | {message}",
    level="INFO",
)

# Optional: expose logger for import
__all__ = ["logger"]