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

4. Скопируйте в папку input_data файл result.json (только такое имя файла) с выгрузкой. Папка output должна быть пуста.

5. Запустите основной файл:
```bash
python dialog/main.py
```
📈 Результаты работы
После выполнения скрипта будут созданы файлы в папке output:
* df_clast.xlsx файл со всемы сообщениями, размечен по кластерам диалогов
* clusters.xlsx файл с темами диалогов и информацией.
