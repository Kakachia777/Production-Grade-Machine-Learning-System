import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and structured output."""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors
        self.fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        } if use_colors else {
            level: self.fmt for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, "extra"):
            log_record.update(record.extra)
            
        return json.dumps(log_record)

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_json: bool = False,
    use_colors: bool = True
) -> None:
    """
    Setup logging configuration with both console and file handlers.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
        use_json: Whether to use JSON formatting for logs
        use_colors: Whether to use colors in console output
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_json:
        console_formatter = JSONFormatter()
    else:
        console_formatter = CustomFormatter(use_colors=use_colors)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        if use_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""
    
    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        if self.extra:
            extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs

def get_logger(name: str, **kwargs) -> LoggerAdapter:
    """
    Get a logger instance with additional context.
    
    Args:
        name: Logger name
        **kwargs: Additional context to add to all log messages
        
    Returns:
        LoggerAdapter instance
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, kwargs) 