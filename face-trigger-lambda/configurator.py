import os
import json
import logging.config
import logging


def setup_logging(
    default_path='/home/aws_cam/lambda_test/face-trigger-lambda/face-trigger-lambda/config/log_config.json',
    default_level=logging.DEBUG,
    env_key='LOG_CFG'
):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def setup_config(config_file_path='/home/aws_cam/lambda_test/face-trigger-lambda/face-trigger-lambda/config/config.json'):
    """
    Setup various parameters required for face-recognition

    """

    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            data = json.load(f)
            logging.info("Configuration parameters are: {}".format(data))

            return data

    else:
        logging.warn(
            "Configuration file does not exist! Using default configuration. (Defined in-program)")
