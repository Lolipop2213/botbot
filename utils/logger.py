# utils/logger.py
"""
Модуль для настройки логирования.
"""
import logging
import os
from datetime import datetime

# Убедимся, что папка для логов существует
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Настройка логгера с выводом в файл и консоль.
    
    Args:
        name (str): Имя логгера.
        log_file (str, optional): Имя файла лога. Если None, используется имя логгера + .log.
        level: Уровень логирования.
        
    Returns:
        logging.Logger: Настроенный логгер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Избегаем добавления повторяющихся обработчиков
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Обработчик для файла
    if log_file is None:
        log_file = f"{name}.log"
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file), encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Предотвращаем логи от propagating к root logger
    logger.propagate = False
    
    return logger

# Глобальный логгер для мониторинга
monitoring_logger = setup_logger("monitoring_bot")

# Глобальный логгер для пайплайна
pipeline_logger = setup_logger("pipeline")