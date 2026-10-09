"""
Утилиты логирования проекта.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'


def logger_name_for_file(base: str, log_file: Optional[Path]) -> str:
    """Уникальное имя логгера, чтобы разные конфиги не писали в один файл."""
    if log_file is None:
        return base
    resolved = Path(log_file).resolve().as_posix().replace(':', '').replace('/', '.')
    return f'{base}.{resolved}'


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Создаёт логгер с файловым хендлером.

    Если тот же logger уже открыт на другой файл — переключает FileHandler.
    Раньше повторный вызов с тем же name и новым путём молча писал в первый файл.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if log_file is None:
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    desired = log_file.resolve()

    for handler in list(logger.handlers):
        if isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            continue
        if isinstance(handler, logging.FileHandler):
            try:
                current = Path(handler.baseFilename).resolve()
            except OSError:
                current = None
            if current == desired:
                return logger
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(file_handler)
    return logger


def close_logger_file_handlers(logger: logging.Logger) -> None:
    """Закрывает FileHandler'ы, чтобы на Windows можно было удалить лог-файл."""
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()


def get_null_logger(name: str) -> logging.Logger:
    """Логгер без вывода в консоль, безопасный по умолчанию."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


_ORIG_PRINT = print


def configure_stdio() -> None:
    """UTF-8 stdout/stderr, чтобы кириллица не ломала pipe на Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        except (OSError, ValueError, AttributeError):
            pass


def safe_print(*args, **kwargs) -> None:
    """print, который не роняет процесс при мёртвом stdout (Errno 22 на Windows)."""
    kwargs.setdefault('flush', True)
    try:
        _ORIG_PRINT(*args, **kwargs)
    except (OSError, ValueError):
        return


def install_safe_print() -> None:
    """Подменяет builtins.print на safe_print для длинных прогонов."""
    import builtins
    builtins.print = safe_print
