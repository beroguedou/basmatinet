import os
import sys
import logging
from pathlib import Path

# Configure location for logs
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = Path(BASE_DIR, 'logs')
# Logger
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'minimal': {'format': '%(message)s'},
        'detailed': {
            'format': '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'minimal',
            'level': logging.DEBUG,
        },
        'info': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': Path(LOGS_DIR, 'info.log'),
            'maxBytes': 10485760,  # 1 MB
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.INFO,
        },
        'error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': Path(LOGS_DIR, 'error.log'),
            'maxBytes': 10485760,  # 1 MB
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.ERROR,
        },
    },
    'loggers': {
        'root': {
            'handlers': ['console', 'info', 'error'],
            'level': logging.DEBUG,
            'propagate': True,
        },
    },
}
