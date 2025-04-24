# Chat Analyzer

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Анализатор чатов - инструмент для визуализации и анализа данных из JSON-экспортов чатов. Создает подробные отчеты в PDF с графиками и статистикой.

## Возможности

- 📊 **Анализ активности пользователей**: Топ-20 самых активных участников
- ⏱ **Временные паттерны**: Активность по часам, дням недели и их комбинациям
- 📝 **Анализ текста**: Облако слов, частотность значимых слов
- 🕸 **Сетевой анализ**: Визуализация взаимодействий между участниками
- 📈 **Дополнительная аналитика**: Длина сообщений, динамика активности
- 📂 **Экспорт результатов**: PDF-отчеты и интерактивные HTML-визуализации

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/ваш-username/chat-analyzer.git
cd chat-analyzer
```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
## Использование

**Запуск из командной строки**
```bash
python -m chat_analyzer.chat_analyzer
```
## Программное использование

```bash
from chat_analyzer import ChatAnalyzer
analyzer = ChatAnalyzer()
analyzer.analyze_chat("path/to/your/chat.json")
```

## Пример вывода
**После обработки в папке output будут созданы:**

- report_<chat_name>.pdf - полный отчет с графиками

- network_<chat_name>.html - интерактивная визуализация сети

**Поддерживаемые форматы данных**
Программа работает с JSON-файлами в формате экспорта Telegram. Пример структуры:
```bash

json
{
  "name": "Название чата",
  "messages": [
    {
      "id": 123,
      "date": "2023-01-01T12:00:00",
      "from": "Имя пользователя",
      "text": "Текст сообщения"
    }
  ]
}
```

**Требования**
- Python 3.7+

**Зависимости:** pandas, matplotlib, seaborn, networkx, pyvis, wordcloud