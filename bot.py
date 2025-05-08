from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import os
import asyncio
import subprocess
import sys
from aiogram.types import FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
import shutil
import zipfile

API_TOKEN = '7933511249:AAFYQ_cbX6io6vvTQZI6S-0iZjquF0ILGHA'
UPLOAD_FOLDER = 'input_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Создаем класс для хранения состояний
class DialogStates(StatesGroup):
    waiting_for_top_dialog = State()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    """Обработчик команды /start с сообщением об ожидании загрузки файла"""
    await message.answer(
        "Привет! Пожалуйста, загрузите файл result.json для продолжения работы. Используйте скрепку, чтобы прикрепить файл"
    )

@dp.message(lambda message: message.document and message.document.file_name == 'result.json')
async def handle_json_file(message: types.Message):
    """Обработка загруженного JSON-файла"""
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    
    # Скачиваем файл
    downloaded_file = await bot.download_file(file_path)
    
    # Сохраняем файл
    save_path = os.path.join(UPLOAD_FOLDER, 'result.json')
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file.read())
    
    await message.answer("Файл result.json успешно загружен!")
    await show_file_buttons(message)  # Показываем file кнопки

@dp.message(lambda message: message.document and message.document.file_name != 'result.json')
async def handle_wrong_file(message: types.Message):
    """Обработка неправильного файла"""
    await message.answer("Пожалуйста, загрузите именно файл result.json")

async def show_file_buttons(message: types.Message):
    """Показывает file кнопки"""
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Очистка папки с отчетами"))
    builder.add(types.KeyboardButton(text="Скачать папку с отчетами"))
    builder.add(types.KeyboardButton(text="Выбор обработки файла result.json"))
    builder.adjust(3)
    await message.answer(
        "Выберите действие:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчики file кнопок
@dp.message(lambda message: message.text == "Выбор обработки файла result.json")
async def button1_handler(message: types.Message):
    await show_main_buttons(message)  # Показываем основные кнопки

@dp.message(lambda message: message.text == "Очистка папки с отчетами")
async def clear_reports_folder(message: types.Message):
    """Очистка папки output"""
    try:
        # Удаляем все содержимое папки output
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Ошибка при удалении {file_path}: {e}')

        await message.answer("Папка с отчетами успешно очищена!")
    except Exception as e:
        await message.answer(f"Ошибка при очистке папки: {str(e)}")
    
@dp.message(lambda message: message.text == "Скачать папку с отчетами")
async def download_reports_folder(message: types.Message):
    """Архивирует и отправляет папку output"""
    try:
        if not os.listdir(OUTPUT_FOLDER):
            await message.answer("Папка с отчетами пуста!")
            return

        # Создаем временный zip-архив
        zip_filename = "reports.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
        
        # Создаем архив
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(OUTPUT_FOLDER):
                for file in files:
                    if file != zip_filename:  # Исключаем сам архив
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, OUTPUT_FOLDER)
                        zipf.write(file_path, arcname)

        # Отправляем архив пользователю
        zip_file = FSInputFile(zip_path)
        await message.answer_document(zip_file, caption="Архив с отчетами")
        
        # Удаляем временный архив
        os.remove(zip_path)
        
    except Exception as e:
        await message.answer(f"Ошибка при создании архива: {str(e)}")

async def show_main_buttons(message: types.Message):
    """Показывает три основные кнопки"""
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="EDA — Исследовательский анализ данных"))
    builder.add(types.KeyboardButton(text="Диалоги — Поиск ключевых тем"))
    builder.add(types.KeyboardButton(text="Кластеризация — Группировка схожих элементов"))
    builder.adjust(3)
    await message.answer(
        "Выберите действие:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчики основных кнопок
@dp.message(lambda message: message.text == "EDA — Исследовательский анализ данных")
async def handle_eda(message: types.Message):
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Это может занять некоторое время. Пожалуйста, подождите...")
    await run_eda_script(message)
    await wait_msg.delete()
    await show_file_buttons(message)

# Функция для запуска скрипта EDA
async def run_eda_script(message: types.Message):
    try:
        script_path = os.path.join("EDA", "EDA.py")
        
        if not os.path.exists(script_path):
            return await message.answer("❌ Файл скрипта не найден")
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            await message.answer(f"✅ Скрипт выполнен")
        else:
            error_msg = result.stderr if result.stderr else "Неизвестная ошибка"
            await message.answer(f"❌ Ошибка выполнения:\n{error_msg[:1000]}")
            
    except Exception as e:
        await message.answer(f"❌ Системная ошибка: {str(e)}")

# Обработчик кнопки "Диалоги"
@dp.message(lambda message: message.text == "Диалоги — Поиск ключевых тем")
async def handle_dialogs(message: types.Message, state: FSMContext):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="top_dialog = 20"))
    builder.add(types.KeyboardButton(text="введите значение top_dialog"))
    builder.adjust(2)
    await message.answer(
        "Выберите вариант:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчик кнопки "введите значение top_dialog"
@dp.message(lambda message: message.text == "введите значение top_dialog")
async def ask_for_top_dialog(message: types.Message, state: FSMContext):
    await message.answer("Пожалуйста, введите целое число для top_dialog:")
    await state.set_state(DialogStates.waiting_for_top_dialog)

# Обработчик введенного значения
@dp.message(DialogStates.waiting_for_top_dialog)
async def process_top_dialog(message: types.Message, state: FSMContext):
    try:
        top_dialog = int(message.text)
        await state.update_data(top_dialog=top_dialog)
        await state.clear()
        wait_msg = await message.answer("⏳ Выполняется обработка файла. Это может занять некоторое время. Пожалуйста, подождите...")
        await run_dialog_script(message, top_dialog)
        await wait_msg.delete()
        await show_file_buttons(message)
    except ValueError:
        await message.answer("Пожалуйста, введите целое число!")
        return

# Обработчик кнопки "top_dialog = 20"
@dp.message(lambda message: message.text == "top_dialog = 20")
async def handle_top_dialog_20(message: types.Message):
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Это может занять некоторое время. Пожалуйста, подождите...")
    await run_dialog_script(message, 20)
    await wait_msg.delete()
    await show_file_buttons(message)

# Функция для запуска скрипта с параметром
async def run_dialog_script(message: types.Message, top_dialog: int):
    try:
        script_path = os.path.join("dialog", "main.py")
        
        if not os.path.exists(script_path):
            return await message.answer("❌ Файл скрипта не найден")
        
        # Запускаем скрипт с передачей параметра
        result = subprocess.run(
            [sys.executable, script_path, str(top_dialog)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            await message.answer(f"✅ Скрипт выполнен с top_dialog={top_dialog}")
        else:
            error_msg = result.stderr if result.stderr else "Неизвестная ошибка"
            await message.answer(f"❌ Ошибка выполнения:\n{error_msg[:1000]}")
            
    except Exception as e:
        await message.answer(f"❌ Системная ошибка: {str(e)}")

@dp.message(lambda message: message.text == "Кластеризация — Группировка схожих элементов")
async def handle_topics(message: types.Message):
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Это может занять некоторое время. Пожалуйста, подождите...")
    await run_topics_script(message)
    await wait_msg.delete()
    await show_file_buttons(message)

# Функция для запуска скрипта topics
async def run_topics_script(message: types.Message):
    try:
        script_path = os.path.join("main_topics", "main.py")
        
        if not os.path.exists(script_path):
            return await message.answer("❌ Файл скрипта не найден")
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            await message.answer(f"✅ Скрипт выполнен")
        else:
            error_msg = result.stderr if result.stderr else "Неизвестная ошибка"
            await message.answer(f"❌ Ошибка выполнения:\n{error_msg[:1000]}")
            
    except Exception as e:
        await message.answer(f"❌ Системная ошибка: {str(e)}")

async def main():
    print("""
    #############################################
    # Бот запущен.                              #
    # Напишите ему в Телеграм в личку:         #
    # @DigestHelpBot                           #
    #############################################
    """)
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен")