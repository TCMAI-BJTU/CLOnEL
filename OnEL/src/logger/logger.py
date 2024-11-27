# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:09
# @Author  : Rui Hua
# @Email   : 
# @File    : logger.py
# @Software: PyCharm
import logging
import os
import time



def setup_logger(log_file):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 记录到文件
            logging.StreamHandler()  # 记录到控制台
        ]
    )

    # 创建日志对象
    _logger = logging.getLogger()  # 初始化后赋值给 _logger
    return _logger


