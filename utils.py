import logging
from datetime import datetime
import os
import yaml

LOG_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = 'logs'


def setup_logger(name='LLMKGC', process_name=""):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = f'LLMKGC_{process_name}_{LOG_TIMESTAMP}.log'
    log_path = os.path.join(LOG_DIR, log_file)

    os.makedirs(LOG_DIR, exist_ok=True)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)
    return logger

def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config