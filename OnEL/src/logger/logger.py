# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:09
# @Author  : Rui Hua
# @Email   : 
# @File    : logger.py
# @Software: PyCharm
import logging
import os



def setup_logger(log_file):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    _logger = logging.getLogger()
    return _logger


