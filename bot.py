import logging
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

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

API_TOKEN = '7933511249:AAFYQ_cbX6io6vvTQZI6S-0iZjquF0ILGHA'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input_data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



# UPLOAD_FOLDER = 'input_data'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# OUTPUT_FOLDER = 'output'
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

class DialogStates(StatesGroup):
    waiting_for_top_dialog = State()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    logger.info(f"Пользователь {message.from_user.id} начал работу с ботом.")
    await message.answer(
        "Привет! Пожалуйста, загрузите файл result.json для продолжения работы. Используйте скрепку, чтобы прикрепить файл"
    )

@dp.message(lambda message: message.document and message.document.file_name == 'result.json')
async def handle_json_file(message: types.Message):
    logger.info(f"Получен файл result.json от пользователя {message.from_user.id}")
    try:
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path

        downloaded_file = await bot.download_file(file_path)
        save_path = os.path.join(UPLOAD_FOLDER, 'result.json')
        logger.info(f"Пользователь загрузил json в {save_path}.")
        with open(save_path, 'wb') as new_file:
            new_file.write(downloaded_file.read())

        logger.info("Файл result.json успешно сохранён.")
        await message.answer("Файл result.json успешно загружен!")
        await show_file_buttons(message)
    except Exception as e:
        logger.exception("Ошибка при обработке файла result.json")
        await message.answer(f"❌ Ошибка при загрузке файла: {str(e)}")

@dp.message(lambda message: message.document and message.document.file_name != 'result.json')
async def handle_wrong_file(message: types.Message):
    logger.warning(f"Пользователь {message.from_user.id} загрузил неверный файл: {message.document.file_name}")
    await message.answer("Пожалуйста, загрузите именно файл result.json")

async def show_file_buttons(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Очистка папки с отчетами"))
    builder.add(types.KeyboardButton(text="Скачать папку с отчетами"))
    builder.add(types.KeyboardButton(text="Обработка файла result.json"))
    builder.adjust(3)
    await message.answer("Выберите действие:", reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message(lambda message: message.text == "Обработка файла result.json")
async def button1_handler(message: types.Message):
    logger.info(f"Пользователь {message.from_user.id} выбрал обработку result.json")
    await show_main_buttons(message)

@dp.message(lambda message: message.text == "Очистка папки с отчетами")
async def clear_reports_folder(message: types.Message):
    try:
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        logger.info("Папка с отчетами очищена")
        await message.answer("Папка с отчетами успешно очищена!")
    except Exception as e:
        logger.exception("Ошибка при очистке папки с отчетами")
        await message.answer(f"Ошибка при очистке папки: {str(e)}")

@dp.message(lambda message: message.text == "Скачать папку с отчетами")
async def download_reports_folder(message: types.Message):
    try:
        if not os.listdir(OUTPUT_FOLDER):
            await message.answer("Папка с отчетами пуста!")
            return
        zip_filename = "reports.zip"
        zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(OUTPUT_FOLDER):
                for file in files:
                    if file != zip_filename:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, OUTPUT_FOLDER)
                        zipf.write(file_path, arcname)
        logger.info(f"Пользователь {message.from_user.id} скачал архив с отчетами")
        zip_file = FSInputFile(zip_path)
        await message.answer_document(zip_file, caption="Архив с отчетами")
        os.remove(zip_path)
    except Exception as e:
        logger.exception("Ошибка при скачивании архива")
        await message.answer(f"Ошибка при создании архива: {str(e)}")

async def show_main_buttons(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="EDA — Исследовательский анализ данных"))
    # builder.add(types.KeyboardButton(text="Диалоги — Поиск ключевых тем"))
    builder.add(types.KeyboardButton(text="Кластеризация — Группировка схожих элементов"))
    builder.adjust(3)
    await message.answer("Выберите действие:", reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message(lambda message: message.text == "EDA — Исследовательский анализ данных")
async def handle_eda(message: types.Message):
    logger.info(f"Запуск EDA от пользователя {message.from_user.id}")
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Пожалуйста, подождите (примерно 2 минуты)...")
    await run_eda_script(message)
    await wait_msg.delete()
    await show_file_buttons(message)

async def run_eda_script(message: types.Message):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'input_data')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        script_path = os.path.join(BASE_DIR, "EDA/EDA.py")
        if not os.path.exists(script_path):
            logger.error("Файл EDA.py не найден")
            return await message.answer("❌ Файл скрипта не найден")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            logger.info("EDA выполнен успешно")
            await message.answer("✅ Скрипт выполнен")
        else:
            logger.error("Ошибка при выполнении EDA:\n" + result.stderr[:500])
            await message.answer(f"❌ Ошибка выполнения:\n{result.stderr[:1000]}")
    except Exception as e:
        logger.exception("Системная ошибка при запуске EDA")
        await message.answer(f"❌ Системная ошибка: {str(e)}")

@dp.message(lambda message: message.text == "Диалоги — Поиск ключевых тем")
async def handle_dialogs(message: types.Message, state: FSMContext):
    logger.info(f"Пользователь {message.from_user.id} выбрал анализ диалогов")
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="top_dialog = 20"))
    builder.add(types.KeyboardButton(text="введите значение top_dialog"))
    builder.adjust(2)
    await message.answer("Выберите вариант:", reply_markup=builder.as_markup(resize_keyboard=True))

@dp.message(lambda message: message.text == "введите значение top_dialog")
async def ask_for_top_dialog(message: types.Message, state: FSMContext):
    await message.answer("Пожалуйста, введите целое число для top_dialog:")
    await state.set_state(DialogStates.waiting_for_top_dialog)

@dp.message(DialogStates.waiting_for_top_dialog)
async def process_top_dialog(message: types.Message, state: FSMContext):
    try:
        top_dialog = int(message.text)
        await state.update_data(top_dialog=top_dialog)
        await state.clear()
        logger.info(f"Введено значение top_dialog={top_dialog} пользователем {message.from_user.id}")
        wait_msg = await message.answer("⏳ Выполняется обработка файла. Пожалуйста, подождите...")
        await run_dialog_script(message, top_dialog)
        await wait_msg.delete()
        await show_file_buttons(message)
    except ValueError:
        logger.warning("Некорректный ввод top_dialog")
        await message.answer("Пожалуйста, введите целое число!")

@dp.message(lambda message: message.text == "top_dialog = 20")
async def handle_top_dialog_20(message: types.Message):
    logger.info("Запуск диалогового анализа с top_dialog=20")
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Пожалуйста, подождите...")
    await run_dialog_script(message, 20)
    await wait_msg.delete()
    await show_file_buttons(message)

async def run_dialog_script(message: types.Message, top_dialog: int):
    try:
        script_path = os.path.join("dialog", "main.py")
        if not os.path.exists(script_path):
            logger.error("Файл dialog/main.py не найден")
            return await message.answer("❌ Файл скрипта не найден")
        result = subprocess.run([sys.executable, script_path, str(top_dialog)],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            logger.info(f"Скрипт диалога выполнен с top_dialog={top_dialog}")
            await message.answer(f"✅ Скрипт выполнен с top_dialog={top_dialog}")
        else:
            logger.error("Ошибка выполнения скрипта диалога:\n" + result.stderr[:500])
            await message.answer(f"❌ Ошибка выполнения:\n{result.stderr[:1000]}")
    except Exception as e:
        logger.exception("Системная ошибка при запуске анализа диалогов")
        await message.answer(f"❌ Системная ошибка: {str(e)}")

@dp.message(lambda message: message.text == "Кластеризация — Группировка схожих элементов")
async def handle_topics(message: types.Message):
    logger.info("Запуск кластеризации")
    wait_msg = await message.answer("⏳ Выполняется обработка файла. Пожалуйста, подождите (примерно 15 минут)...")
    await run_topics_script(message)
    await wait_msg.delete()
    await show_file_buttons(message)
    
import asyncio
import os
import sys
from aiogram import types

async def run_topics_script_new(message: types.Message):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(BASE_DIR, "main_topics/main.py")

        if not os.path.exists(script_path):
            logger.error("Файл кластеризации не найден")
            return await message.answer("❌ Файл скрипта не найден")

        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await message.answer("⏳ Запуск скрипта кластеризации...")

        async def stream_output(stream, prefix):
            buffer = ""
            async for line in stream:
                decoded_line = line.decode().strip()
                buffer += decoded_line + "\n"
                # Отправка по 1000 символов или по новой строке
                if len(buffer) > 1000 or "\n" in decoded_line:
                    await message.answer(f"{prefix} <pre>{buffer.strip()[:4000]}</pre>", parse_mode='HTML')
                    buffer = ""
            if buffer:
                await message.answer(f"{prefix} <pre>{buffer.strip()[:4000]}</pre>", parse_mode='HTML')

        await asyncio.gather(
            stream_output(process.stdout, "📄 stdout:"),
            stream_output(process.stderr, "⚠️ stderr:")
        )

        return_code = await process.wait()

        if return_code == 0:
            logger.info("Скрипт кластеризации успешно выполнен")
            await message.answer("✅ Скрипт выполнен")
        else:
            logger.error(f"Скрипт завершился с кодом {return_code}")
            await message.answer(f"❌ Скрипт завершился с кодом {return_code}")

    except Exception as e:
        logger.exception("Системная ошибка при запуске кластеризации")
        await message.answer(f"❌ Системная ошибка: {str(e)}")


async def run_topics_script(message: types.Message):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(BASE_DIR, "main_topics/main.py")
        # script_path = os.path.join("main_topics", "main.py")
        if not os.path.exists(script_path):
            logger.error("Файл кластеризации не найден")
            return await message.answer("❌ Файл скрипта не найден")
        result = subprocess.run([sys.executable, script_path], 
                                capture_output=True, 
                                text=True, 
                                encoding='utf-8', 
                                errors='replace')
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0:
            logger.info("Скрипт выполнен успешно")
            if stdout:
                await message.answer(f"✅ Скрипт выполнен:\n<pre>{stdout[:4000]}</pre>", parse_mode='HTML')
            else:
                await message.answer("✅ Скрипт выполнен без вывода")
        else:
            logger.error("Скрипт завершился с ошибкой:\n" + stderr[:500])
            await message.answer(f"❌ Ошибка:\n<pre>{stderr[:4000]}</pre>", parse_mode='HTML')


        if result.stdout:
            logger.info(f"stdout: {result.stdout[:1000]}")
        if result.returncode == 0:
            logger.info("Скрипт кластеризации успешно выполнен")
            await message.answer("✅ Скрипт выполнен")
        else:
            logger.error("Ошибка выполнения скрипта кластеризации:\n" + result.stderr[:500])
            await message.answer(f"❌ Ошибка выполнения:\n{result.stderr[:1000]}")
    except Exception as e:
        logger.exception("Системная ошибка при запуске кластеризации")
        await message.answer(f"❌ Системная ошибка: {str(e)}")

async def main():
    logger.info("Бот запущен и ожидает входящих сообщений")
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
        logger.info("Бот остановлен вручную")
