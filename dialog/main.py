import json
import pandas as pd
import os
import numpy as np
import networkx as nx
import ollama
import time
from tqdm import tqdm
import sys
import time

# постоянная для вычесления временного порога близости сообщений
# суть: 41й перцентиль среди временных распределений в диалоге
# 41 найдено практически и интуитивно
PERC = 41 
FOLDER_OUTPUT = 'output'
start = time.perf_counter()

if __name__ == "__main__":
    top_dialog = int(sys.argv[1]) if len(sys.argv) > 1 else 20  # значение по умолчанию 20

# определим путь
path_json = os.getcwd() + '/input_data/result.json'
with open(path_json, 'r', encoding='utf-8') as file:
    chat_data = json.load(file)

# Извлекаем сообщения, а также название чата(в JSON-файл один чат) и chat_id
messages = chat_data.get('messages', [])
chat_name = chat_data.get('name')
chat_id = chat_data.get('id')
# парсинг
parsed_messages = []
for message in messages:
    if message.get('type') == 'message':
        text = message['text']
        reactions = message.get('reactions', [])
        reaction_count = sum(reaction.get('count', 0) for reaction in reactions) if reactions else 0
        parsed_messages.append({
            'message_id': message['id'],
            'date': message['date'],
            'chat_name': chat_name,
            'chat_id': chat_id,
            'sender_id': message.get('from_id'),
            'from': message.get('from', 'Unknown'),
            'reply_to_message_id': message.get('reply_to_message_id'),
            'text': text,
            'reaction_count': reaction_count
        })
# Создаем датафрейм
df = pd.DataFrame(parsed_messages)
df['date'] = pd.to_datetime(df['date'])

# вычеслим 41й перцентиль на списке временных итервалов всех сообщений
pl = []
for i in range(len(df)-1):
   time_diff = df.loc[i+1, 'date']-df.loc[i, 'date']
   seconds = time_diff.total_seconds()
   pl.append(seconds)
percentile_int = int(np.percentile(pl, PERC))
# выразим в минутах
th_minute = percentile_int/60

# соберем кластеры
x = []
for i in range(len(df)-1, -1, -1):
  if not pd.isna(df.loc[i, 'reply_to_message_id']):
    x.append([df.loc[i, 'message_id'], df.loc[i, 'reply_to_message_id']])
for i in range(len(df)-1):
  if df.loc[i+1, 'date']-df.loc[i, 'date'] < pd.Timedelta(minutes=th_minute):
    x.append([df.loc[i, 'message_id'], df.loc[i+1, 'message_id']])
para = pd.DataFrame(x, columns=['one','two'])
G = nx.Graph()
for _, row in para.iterrows():
    G.add_edge(row['one'], row['two'])
connected_components = list(nx.connected_components(G))
result = [sorted(list(component)) for component in connected_components]
result.sort(key=lambda x: x[0])
sorted_result = sorted(result, key=lambda x: len(x), reverse=True)
df_clast = df[['message_id', 'text']]
df_clast = df_clast.copy()
df_clast.loc[:, 'n_claster'] = np.nan
# Создаем словарь для быстрого поиска кластера по message_id
cluster_mapping = {}
for cluster_idx, cluster in enumerate(sorted_result):
    for message_id in cluster:
        cluster_mapping[float(message_id)] = cluster_idx  # преобразуем np.float64 в float

# Заполняем столбец n_claster
df_clast.loc[:, 'n_claster'] = df_clast['message_id'].map(cluster_mapping)

# Если какие-то message_id не найдены в кластерах, они останутся NaN
# Создаем и добавляем столбец с размерами кластеров
cluster_sizes = {i: len(cluster) for i, cluster in enumerate(sorted_result)}
df_clast.loc[:, 'cluster_size'] = df_clast['n_claster'].map(cluster_sizes)
# Считаем вхождения каждого ID в обоих столбцах
counts_one = para['one'].value_counts()
counts_two = para['two'].value_counts()

# Объединяем счетчики (складываем значения для одинаковых ID)
total_counts = counts_one.add(counts_two, fill_value=0).astype(int)

# Добавляем новый столбец в исходный датафрейм
para['connection_count'] = para['one'].map(total_counts)
# Добавляем счетчик связей в df_clast
df_clast.loc[:,'connection_count'] = df_clast['message_id'].map(total_counts)
df_clast_filtred = df_clast.query('connection_count > 2')
clusters = df_clast_filtred.groupby('n_claster')['text'].apply(list).reset_index()

# для уточнения рейтинга кластеров добавим реакции
df_clast['reaction_count'] = df['reaction_count']
# Создаем словарь для хранения суммы реакций по кластерам
cluster_reactions = df_clast.groupby('n_claster')['reaction_count'].sum().to_dict()

# Добавляем столбец с рейтингом кластера
df_clast['cluster_rating'] = df_clast.apply(
    lambda row: row['cluster_size'] + cluster_reactions.get(row['n_claster'], 0) 
    if pd.notna(row['n_claster']) else np.nan,
    axis=1
)
# Создаем словарь с рейтингами кластеров из df_clast
cluster_ratings = df_clast.groupby('n_claster')['cluster_rating'].first().to_dict()

# Добавляем столбец cluster_rating в clusters
clusters['cluster_rating'] = clusters['n_claster'].map(cluster_ratings)

# Заполняем NaN нулями (если кластер есть в clusters, но нет в df_clast)
clusters['cluster_rating'] = clusters['cluster_rating'].fillna(0)
# сортируем темы по рейтингу
clusters = clusters.sort_values('cluster_rating', ascending=False)


# Определим функции для получения темы от LLM через батчи
def process_batch(batch_messages):
    """Обрабатывает батч из 5 сообщений и возвращает список тем"""
    batch_topics = []
    for messages in batch_messages:
        if isinstance(messages, list):
            processed_text = ' '.join([str(item) if not isinstance(item, dict) else item.get('text', '') 
                                    for item in messages])
        else:
            processed_text = str(messages)

        prompt = f"""
        Проанализируй следующие сообщения из одного кластера и определи основную тему диалога.
        Сообщения должны иметь общую тематику. Напиши только основную тему. Без пояснений.

        Сообщения:
        {processed_text}

        Основная тема диалога:
        """
        try:
            response = ollama.generate(
                model="llama3:8b",  # Квантованная версия
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_ctx': 2048  # Уменьшенный контекст
                }
            )          
            topic = response['response'].strip().strip('"').split('\n')[0]
            batch_topics.append(topic)
        except Exception as e:
            print(f"\nОшибка при обработке кластера: {e}")
            batch_topics.append("Не определена")
    
    return batch_topics

# Функция для батч-обработки
def process_in_batches(df, batch_size=5):
    """Обрабатывает DataFrame батчами"""
    all_topics = []
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Обработка кластеров"):
        batch = df['text'].iloc[i:i+batch_size].tolist()
        batch_topics = process_batch(batch)
        all_topics.extend(batch_topics)
        time.sleep(1)  # небольшая пауза между батчами
        
    return all_topics

# Определим функции для получения  ключевых слов от LLM через батчи
def process_batch_words(batch_messages):
    """Обрабатывает батч из 5 сообщений и возвращает списки ключевых слов"""
    batch_topics = []
    for messages in batch_messages:
        if isinstance(messages, list):
            processed_text = ' '.join([str(item) if not isinstance(item, dict) else item.get('text', '') 
                                    for item in messages])
        else:
            processed_text = str(messages)

        prompt = f"""
        [Инструкция]
        Анализируй сообщения и строго верни ровно 10 ключевых слов через запятую.
        Требования:
        1. Ровно 10 слов/фраз
        2. Только существительные или словосочетания (2-3 слова)
        3. Разделитель: запятая
        4. Без точек, кавычек, номеров и пояснений
        5. Если слов меньше 10 - дополни список общими словами из темы

        [Пример]
        маркетинг, реклама, Яндекс Директ, настройка, объявления, аудитория, таргетинг, бюджет, конверсия, CTR

        [Сообщения]
        {processed_text}

        [Ответ]
        """
        
        try:
            response = ollama.generate(
                model="llama3:8b",
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_ctx': 2048  # Уменьшенный контекст
                }
            )
            topic = response['response'].strip().strip('"').split('\n')[0]
            batch_topics.append(topic)
        except Exception as e:
            print(f"\nОшибка при обработке кластера: {e}")
            batch_topics.append("Не определена")
    
    return batch_topics
    
 
# Функция для батч-обработки # убрал tqdm
def process_in_batches_words(df, batch_size=5):
    """Обрабатывает DataFrame батчами"""
    all_topics = []
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Обработка кластеров"):
        batch = df['text'].iloc[i:i+batch_size].tolist()
        batch_topics = process_batch_words(batch)
        all_topics.extend(batch_topics)
        time.sleep(1)  # небольшая пауза между батчами
        
    return all_topics

# Определим функции для получения главной мысли от LLM через батчи
def process_batch_idea(batch_messages):
    """Обрабатывает батч из 5 сообщений и возвращает списки ключевых слов"""
    batch_topics = []
    for messages in batch_messages:
        if isinstance(messages, list):
            processed_text = ' '.join([str(item) if not isinstance(item, dict) else item.get('text', '') 
                                    for item in messages])
        else:
            processed_text = str(messages)

        prompt = f"""
        [Инструкция]
        Проанализируй следующие сообщения из чата и выдели главную мысль этого диалога, 
        напиши главную мысль на руском языке в виде 1-3 предложений. 
        Предложения должны быть:
        - на русском языке
        - без ковычек
        - без пояснений и дополнительного текста

        [Сообщения для анализа]
        {processed_text}

            """
        
        try:
            response = ollama.generate(
                model="llama3:8b",
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_ctx': 2048  # Уменьшенный контекст
                }
            )
            topic = response['response'].strip().strip('"').split('\n')[0]
            batch_topics.append(topic)
        except Exception as e:
            print(f"\nОшибка при обработке кластера: {e}")
            batch_topics.append("Не определена")
    
    return batch_topics
    
# Функция для батч-обработки
def process_in_batches_idea(df, batch_size=5):
    """Обрабатывает DataFrame батчами"""
    all_topics = []
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Обработка кластеров"):
        batch = df['text'].iloc[i:i+batch_size].tolist()
        batch_topics = process_batch_idea(batch)
        all_topics.extend(batch_topics)
        time.sleep(1)  # небольшая пауза между батчами
        
    return all_topics

# Получаем темы от LLM
clusters['topic'] = "не обработано"  # Сначала заполняем все значения
clusters.iloc[:top_dialog, clusters.columns.get_loc('topic')] = process_in_batches(clusters.head(top_dialog))

# Получаем ключевые слова от LLM
clusters['words'] = "не обработано"  # Сначала заполняем все значения
clusters.iloc[:top_dialog, clusters.columns.get_loc('words')] = process_in_batches_words(clusters.head(top_dialog))

# Получаем главную мысль от LLM
clusters['idea'] = "не обработано"  # Сначала заполняем все значения
clusters.iloc[:top_dialog, clusters.columns.get_loc('idea')] = process_in_batches_idea(clusters.head(top_dialog))

# Получаем абсолютный путь к родительскому каталогу проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
path_dir_output = os.path.join(parent_dir, FOLDER_OUTPUT)
# Создаем папку (если не существует)
os.makedirs(path_dir_output, exist_ok=True)

# Экспорт DataFrame в Excel файл
df_clast.to_excel('output/df_clast.xlsx', index=False, engine='openpyxl')
print("Файл успешно сохранён: df_clast.xlsx")

# перед сохраниением удалим ненужный столбец с текстами
clusters = clusters.drop(['text'], axis=1)

# Экспорт DataFrame в Excel файл
clusters.to_excel('output/clusters.xlsx', index=False, engine='openpyxl')
print("Файл успешно сохранён: clusters.xlsx")

end = time.perf_counter()
print(f"Время выполнения: {end - start:.2f} сек")