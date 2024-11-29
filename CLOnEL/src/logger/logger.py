# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:09
# @Author  : Rui Hua
# @Email   : 
# @File    : logger.py
# @Software: PyCharm
import logging
import os
import time

def setup_logger():
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"logs/{current_time}.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")

    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    return logger


logger = setup_logger()