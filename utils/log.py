import logging.config
import os
from pathlib import Path


LOGGING = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(levelname)-8s]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        } 
    },
    'filters': {

    },
    'handlers': {
        'file': {
            'level': 'INFO', 
            'class': 'logging.FileHandler',
            'filename': str(Path(os.path.abspath(__file__)).parent.parent) + '/train.log'

        },
        'console': {
            'level': 'INFO',
            'formatter': 'default',
            'class': 'logging.StreamHandler'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING)
