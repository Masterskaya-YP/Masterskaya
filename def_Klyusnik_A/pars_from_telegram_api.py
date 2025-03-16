from telethon import TelegramClient
import pandas as pd

api_id = 'YOUR_API_ID'
api_hash = 'YOUR_API_HASH'
phone_number = 'YOUR_PHONE_NUMBER'

# Создание клиента
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    await client.start()
    
    # Получение списка чатов и каналов

    async for dialog in client.iter_dialogs():
        print(dialog.name, dialog.id)

    # Извлечение сообщений из 18 активных чатов/каналов
    messages = []
    chat_ids = []  # Список для хранения ID чатов

    async for dialog in client.iter_dialogs():
        if len(messages) >= 18:  # Ограничение на 18 чатов
            break
        if dialog.is_group or dialog.is_channel:
            chat_ids.append(dialog.id)
            async for message in client.iter_messages(dialog.id, limit=100):  # Извлечение последних 100 сообщений
                messages.append({
                    'chat_id': dialog.id,
                    'chat_name': dialog.name,
                    'message': message.message,
                    'date': message.date
                })

    # Сохранение сообщений в DataFrame
    df = pd.DataFrame(messages)
    df.to_csv('Masterskaya\def_Klyusnik_A\download\messages.csv', index=False)  # Сохранение в CSV файл

with client:
    client.loop.run_until_complete(main())