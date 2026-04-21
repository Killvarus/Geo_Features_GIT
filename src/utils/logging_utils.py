"""
Утилиты логирования проекта.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Создаёт логгер с файловым хендлером и без дублирования обработчиков."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_null_logger(name: str) -> logging.Logger:
    """Логгер без вывода в консоль, безопасный по умолчанию."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger
