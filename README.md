# Анализатор чатов Telegram

Проект для анализа чатов Telegram с использованием методов NLP и сетевого анализа.

## Возможности

- Анализ активности пользователей
- Визуализация временных паттернов активности
- Сетевой анализ взаимодействий между участниками
- Кластеризация тем сообщений с помощью BERTopic
- Генерация тем и ключевых слов с использованием LLM (Llama3, YandexGPT)
- Сравнительный анализ нескольких чатов

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone [https://github.com/yourusername/chat-analyzer.git](https://github.com/Masterskaya-YP/Masterskaya
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Установите Ollama (для работы с локальными LLM):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3:8b
   ```

## Использование

1. Поместите JSON-файлы с экспортированными чатами Telegram в папку input_data

2. Запустите анализатор:
   ```bash
   python EDA/chat_analyzer.py
   ```
3. Для тематического анализа запустите:
   ```bash
   python main_topics/main.py
   ```
4. Для разметки диалогов и получению тем и информаци:
   ```bash
   python dialog/main.py
   ```
5. Результаты будут сохранены в папке output:

   - PDF-отчеты с визуализациями

   - Excel-файлы с кластеризованными темами

   - HTML-файлы с интерактивными сетями взаимодействий

## Структура проекта:

```bash
Masterskaya/
├── EDA/
│   ├──EDA.py
│   ├──README.md
│   └──requirements.txt
├── dialog/
│   ├── research/
│   │     ├── README.md
│   │     └──clusters.ipynb
│   ├── README.md  
│   ├── main.py
│   └── requirements.txt  
├── input_data/          # Входные JSON-файлы чатов
├── main_topics/
│   ├──search/
│   │  ├── data/
│   │  └── bertopic_add_LLM_localy.ipynb
│   ├──src/module/
│   │  ├── __init__.py
│   │  ├── bertopic_setup.py
│   │  └── dict_to_dataframe_parser.py
│   ├──README.md
│   └── main.py
├── output/             # Результаты анализа
├── bot.py
└── README.md           # Этот файл
```

## Требования
1. Python 3.8+

2. Основные зависимости:

- pandas, numpy, matplotlib, seaborn

- networkx, pyvis

- transformers, bertopic, llama-cpp

- ollama (для работы с локальными LLM)

## Примеры отчетов

1. Активность пользователей - Топ-20 самых активных участников

2. Временные паттерны - Графики активности по часам и дням недели

3. Сетевые графики - Визуализация взаимодействий между участниками

4. Тематические кластеры - Основные темы обсуждений с ключевыми словами

5. Сравнение чатов - Сравнительные таблицы и графики для нескольких чатов    
