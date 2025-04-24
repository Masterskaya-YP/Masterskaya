"""
chat_analyzer - Пакет для анализа данных чатов из JSON-файлов

Основные возможности:
- Анализ активности пользователей
- Визуализация временных паттернов
- Сетевой анализ взаимодействий
- Генерация отчетов в PDF
"""

from .chat_analyzer import ChatAnalyzer

__version__ = '0.1.0'
__author__ = 'Ваше имя или название организации'
__all__ = ['ChatAnalyzer']

# Инициализация логгера при импорте пакета
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f'Инициализация пакета chat_analyzer версии {__version__}')