from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder

# Токен вашего бота (замените на свой)
API_TOKEN = '7933511249:AAFYQ_cbX6io6vvTQZI6S-0iZjquF0ILGHA'

# Инициализация бота с настройками по умолчанию
bot = Bot(
    token=API_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

# Обработчик команды /start
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Кнопка 1"))
    builder.add(types.KeyboardButton(text="Кнопка 2"))
    builder.add(types.KeyboardButton(text="Кнопка 3"))
    builder.adjust(3)  # 3 кнопки в ряд

    await message.answer(
        "Привет! Это стартовое сообщение. Выберите одну из кнопок:",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчики кнопок
@dp.message(lambda message: message.text == "Кнопка 1")
async def button1_handler(message: types.Message):
    await message.answer("Вы нажали Кнопку 1!")

@dp.message(lambda message: message.text == "Кнопка 2")
async def button2_handler(message: types.Message):
    await message.answer("Вы нажали Кнопку 2!")

@dp.message(lambda message: message.text == "Кнопка 3")
async def button3_handler(message: types.Message):
    await message.answer("Вы нажали Кнопку 3!")

# Запуск бота
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
    import asyncio
    asyncio.run(main())