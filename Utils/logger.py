import os
import logging
from datetime import datetime

def get_logger(name=None, console=True, file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger = config_logger(logger, console, file, name)
    return logger
    
def config_logger(logger, console:bool, file:bool, name:str):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s')
    
    if console == True:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file == True:
        now = datetime.now()
        file_handler = logging.FileHandler(os.path.join("/System_Integration/Output/log", f'{name} {now}.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger