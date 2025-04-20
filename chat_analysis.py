# Парсинг json файла
import json
import pandas as pd
import numpy as np
import os
import logging
import warnings
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from PyPDF2 import PdfWriter
from matplotlib.backends.backend_pdf import PdfPages

# Настройки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
mpl.rcParams.update(mpl.rcParamsDefault)
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

def load_and_process_json(file_path):
    """Загрузка и обработка JSON файла"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    df.rename(columns={'id': 'id_first'}, inplace=True)
    messages_df = pd.json_normalize(df['messages'])
    result_df = pd.concat([df.drop(columns=['messages']), messages_df], axis=1)
    return result_df

def preprocess_df(df):
    """Предобработка данных"""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    if 'text' in df.columns:
        try:
            df['text'] = df['text'].apply(ast.literal_eval)
            df['text_clean'] = df['text'].apply(lambda x: x[0]['text'] if x else '')
        except:
            df['text_clean'] = df['text'].fillna('')
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['date_only'] = df['date'].dt.date
    return df

def analyze_active_users(df, chat_name, pdf_pages):
    """Анализ активных пользователей"""
    user_activity = df['from'].value_counts().head(20)
    
    fig = plt.figure(figsize=(12, 8))
    user_activity.plot(kind='barh')
    plt.title(f'Топ-20 активных пользователей в {chat_name}')
    plt.xlabel('Количество сообщений')
    plt.ylabel('Пользователь')
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close()
    
    return user_activity

def analyze_time_patterns(df, chat_name, pdf_pages):
    """Анализ временных паттернов"""
    # Активность по часам
    hourly_activity = df['hour'].value_counts().sort_index()
    
    fig = plt.figure(figsize=(12, 6))
    hourly_activity.plot(kind='bar')
    plt.title(f'Активность по часам в {chat_name}')
    plt.xlabel('Час дня')
    plt.ylabel('Количество сообщений')
    plt.xticks(rotation=0)
    pdf_pages.savefig(fig)
    plt.close()
    
    # Активность по дням недели
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_activity = df['day_of_week'].value_counts().reindex(days_order)
    
    fig = plt.figure(figsize=(12, 6))
    daily_activity.plot(kind='bar')
    plt.title(f'Активность по дням недели в {chat_name}')
    plt.xlabel('День недели')
    plt.ylabel('Количество сообщений')
    plt.xticks(rotation=45)
    pdf_pages.savefig(fig)
    plt.close()
    
    # Heatmap активности
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack().reindex(days_order)
    fig = plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="g")
    plt.title(f'Активность по дням недели и часам в {chat_name}')
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close()
    
    return {'hourly_activity': hourly_activity, 'daily_activity': daily_activity}

def analyze_text_content(df, chat_name, pdf_pages):
    """Анализ текстового содержания"""
    if 'text_clean' not in df.columns:
        return None
    
    def clean_text(text):
        if not isinstance(text, str):
            return ''
        text = re.sub(r'(type|castom_emoji|bot_command|\\n|\[|\]|\{|\}|\(|\))', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [word for word in text.split() 
                if len(word) > 5 and not word.isdigit() 
                and not word.startswith(('http', 'www'))]
        return ' '.join(words)
    
    df['cleaned_text'] = df['text_clean'].apply(clean_text)
    custom_stopwords = {'который', 'которые', 'когда', 'потому', 'очень', 'может', 'будет', 'этого', 'этот'}
    
    # Облако слов
    all_text = ' '.join(df['cleaned_text'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         stopwords=custom_stopwords, collocations=False).generate(all_text)
    
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Облако слов в {chat_name}')
    plt.axis('off')
    pdf_pages.savefig(fig)
    plt.close()
    
    # Топ-20 слов
    words = [word for text in df['cleaned_text'].dropna() 
             for word in text.split() if word not in custom_stopwords]
    
    word_counts = Counter(words).most_common(20)
    
    if word_counts:
        fig = plt.figure(figsize=(12, 6))
        pd.DataFrame(word_counts, columns=['word', 'count']).plot(
            x='word', y='count', kind='bar', color='teal')
        plt.title(f'Топ-20 значимых слов в {chat_name}')
        plt.xlabel('Слово')
        plt.ylabel('Частота')
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close()
    
    return word_counts

def analyze_network(df, chat_name, pdf_pages):
    """Сетевой анализ"""
    if 'reply_to_message_id' not in df.columns or 'from_id' not in df.columns:
        return None
    
    G = nx.DiGraph()
    user_message_count = df['from'].value_counts().to_dict()
    
    for user, count in user_message_count.items():
        G.add_node(user, size=count)
    
    replies = df[df['reply_to_message_id'].notna()]
    for _, row in replies.iterrows():
        original_msg = df[df['id'] == row['reply_to_message_id']]
        if not original_msg.empty:
            source = original_msg.iloc[0]['from']
            target = row['from']
            G.add_edge(source, target, weight=G.get_edge_data(source, target, {}).get('weight', 0) + 1)
    
    if len(G.nodes()) == 0:
        return None

    # Визуализация графа
    fig = plt.figure(figsize=(18, 14))
    plt.title(f'Структура взаимодействий в чате {chat_name}', pad=20, fontsize=14)
    
    node_sizes = [np.log(G.nodes[n]['size'])*800 for n in G.nodes()]
    edge_widths = [0.3 + G[u][v]['weight']*0.7 for u,v in G.edges()]
    pos = nx.spring_layout(G, k=0.7, iterations=100, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#1f78b4', alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#666666', alpha=0.5)
    
    degrees = [d for n,d in G.degree(weight='weight')]
    if degrees:
        threshold = np.percentile(degrees, 75)
        important_nodes = [n for n in G.nodes() if G.degree(n, weight='weight') > threshold]
        labels = {n: n for n in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    plt.axis('off')
    pdf_pages.savefig(fig)
    plt.close()
    
    return G

def additional_analysis(df, chat_name, pdf_pages):
    """Дополнительный анализ"""
    if 'text_clean' in df.columns:
        df['msg_length'] = df['text_clean'].str.len()
        fig = plt.figure(figsize=(12, 6))
        df['msg_length'].hist(bins=50)
        plt.title(f'Распределение длины сообщений в {chat_name}')
        plt.xlabel('Длина сообщения (символы)')
        plt.ylabel('Количество')
        pdf_pages.savefig(fig)
        plt.close()
    
    if 'date' in df.columns:
        daily_counts = df.resample('D', on='date').size()
        fig = plt.figure(figsize=(14, 6))
        daily_counts.plot()
        plt.title(f'Динамика сообщений в {chat_name}')
        plt.xlabel('Дата')
        plt.ylabel('Количество сообщений')
        plt.grid(True)
        pdf_pages.savefig(fig)
        plt.close()

def main():
    # Загрузка данных
    df_1 = load_and_process_json('./data/дата/result.json')
    df_2 = load_and_process_json('./data/менеджмент/result.json')
    df_3 = load_and_process_json('./data/маркетинг/result.json')
    
    # Предобработка
    df_1 = preprocess_df(df_1)
    df_2 = preprocess_df(df_2)
    df_3 = preprocess_df(df_3)
    
    # Создание PDF отчета
    with PdfPages('chat_analysis_report.pdf') as pdf_pages:
        # Анализ для каждого чата
        for df, chat_name in [(df_1, "DATA PRACTICUM"), (df_2, "MANAGEMENT ALUMNI"), (df_3, "MARKETING CHAT")]:
            # Добавляем страницу с названием чата
            fig = plt.figure(figsize=(11, 8.5))
            plt.text(0.5, 0.5, f"Анализ чата: {chat_name}", 
                    ha='center', va='center', fontsize=20)
            plt.axis('off')
            pdf_pages.savefig(fig)
            plt.close()
            
            # Выполняем анализы
            analyze_active_users(df, chat_name, pdf_pages)
            analyze_time_patterns(df, chat_name, pdf_pages)
            analyze_text_content(df, chat_name, pdf_pages)
            analyze_network(df, chat_name, pdf_pages)
            additional_analysis(df, chat_name, pdf_pages)

if __name__ == "__main__":
    main()