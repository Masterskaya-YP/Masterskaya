## ⚙️ Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Masterskaya-YP/Masterskaya.git
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.\.venv\Scripts\activate   # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```
Например: pip install json pandas numpy networkx ollama tqdm

4. Установка на сервере ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

6. Запускаем ollama
```bash
ollama serve
```

7. Загружаем модель. Примерно 4ГБ
```bash
ollama pull llama3:8b
```

8. Скопируйте в папку input_data файл result.json (только такое имя файла) с выгрузкой. Папка output должна быть пуста.

9. Запустите основной файл для диалогов:
```bash
python dialog/main.py
```

9. При запуске программа выделяет диалоги и их может быть 100-300 шт, молбщие и маленькие. По умолчанию LLM определяет тему и другую информацию у первых 20, согласно рейтингу. Топ-20 пожно поменять с помощью команды. Например обрабатываем Топ-3:
```bash
python dialog/main.py --top 30
```

📈 Результаты работы
После выполнения скрипта будут созданы файлы в папке output:
* df_clast.xlsx файл со всемы сообщениями, размечен по кластерам диалогов
* clusters.xlsx файл с темами диалогов и информацией.
