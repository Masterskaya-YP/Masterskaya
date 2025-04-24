# chat_analyzer.py
import json
import logging
import os
import warnings
import re
import ast  # Добавлен импорт ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
from pyvis.network import Network
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib import cm

# Настройки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
mpl.rcParams.update(mpl.rcParamsDefault)
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

class ChatAnalyzer:
    def __init__(self):
        self.output_dir = Path("./output")  # Изменено на ./output
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_process_json(self, file_path):
        """Загрузка и обработка JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data)
        df.rename(columns={'id': 'id_first'}, inplace=True)
        messages_df = pd.json_normalize(df['messages'])
        return pd.concat([df.drop(columns=['messages']), messages_df], axis=1)
    
    def preprocess_df(self, df):
        """Предобработка данных"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if 'text' in df.columns:
            try:
                df['text'] = df['text'].apply(ast.literal_eval)  # Теперь ast доступен
                df['text_clean'] = df['text'].apply(lambda x: x[0]['text'] if x else '')
            except:
                df['text_clean'] = df['text'].fillna('')
        
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        df['date_only'] = df['date'].dt.date
        return df


    def analyze_active_users(self, df, chat_name, pdf_pages):
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
    
    def analyze_time_patterns(self, df, chat_name, pdf_pages):
        """Анализ временных паттернов"""
        hourly_activity = df['hour'].value_counts().sort_index()
        
        fig = plt.figure(figsize=(12, 6))
        hourly_activity.plot(kind='bar')
        plt.title(f'Активность по часам в {chat_name}')
        plt.xlabel('Час дня')
        plt.ylabel('Количество сообщений')
        plt.xticks(rotation=0)
        pdf_pages.savefig(fig)
        plt.close()
        
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
        
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack().reindex(days_order)
        fig = plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="g")
        plt.title(f'Активность по дням недели и часам в {chat_name}')
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close()
        
        return {'hourly_activity': hourly_activity, 'daily_activity': daily_activity}
    
    def analyze_text_content(self, df, chat_name, pdf_pages):
        """Анализ текстового содержания"""
        if 'text_clean' not in df.columns:
            logger.warning(f"В чате {chat_name} нет данных для анализа текста")
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
        
        all_text = ' '.join(df['cleaned_text'].dropna())
        
        if not all_text.strip():
            logger.warning(f"В чате {chat_name} нет текста для построения облака слов после очистки")
            return None
        
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                stopwords=custom_stopwords, collocations=False).generate(all_text)
            
            fig = plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Облако слов в {chat_name}')
            plt.axis('off')
            pdf_pages.savefig(fig)
            plt.close()
        except ValueError as e:
            logger.warning(f"Не удалось создать облако слов для {chat_name}: {str(e)}")
            return None
        
        words = [word for text in df['cleaned_text'].dropna() 
                 for word in text.split() if word not in custom_stopwords]
        
        if not words:
            logger.warning(f"В чате {chat_name} нет слов для анализа после фильтрации")
            return None
        
        word_counts = Counter(words).most_common(20)
        
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
    
    def analyze_network(self, df, chat_name, pdf_pages):
        """Сетевой анализ с визуализацией кластеров"""
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

        # 1. Основной граф взаимодействий
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
        
        # 2. Интерактивная визуализация
        net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
        pagerank = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        
        for node in G.nodes():
            net.add_node(
                node, 
                size=np.log(G.nodes[node]['size']) * 5,
                title=f"{node}\nСообщений: {G.nodes[node]['size']}\nPageRank: {pagerank[node]:.3f}\nBetweenness: {betweenness[node]:.3f}",
                group=int(pagerank[node] * 100))
        
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], width=edge[2]['weight']*0.5)
        
        html_path = self.output_dir / f"network_{chat_name}.html"
        net.save_graph(str(html_path))
        logger.info(f"Интерактивная визуализация сохранена в {html_path}")
        
        return G
    
    def additional_analysis(self, df, chat_name, pdf_pages):
        """Дополнительный анализ"""
        if 'text_clean' in df.columns:
            df['msg_length'] = df['text_clean'].str.len()
            avg_length = df['msg_length'].mean()
            
            fig = plt.figure(figsize=(12, 6))
            df['msg_length'].hist(bins=50)
            plt.title(f'Распределение длины сообщений в {chat_name}\nСредняя длина: {avg_length:.1f} символов')
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
            
            rolling_avg = daily_counts.rolling(window=7).mean()
            
            fig = plt.figure(figsize=(14, 6))
            rolling_avg.plot()
            plt.title(f'Скользящее среднее (7 дней) активности в {chat_name}')
            plt.xlabel('Дата')
            plt.ylabel('Количество сообщений')
            plt.grid(True)
            pdf_pages.savefig(fig)
            plt.close()
    
    def analyze_chat(self, json_path):
        """Основной метод анализа чата"""
        try:
            chat_name = Path(json_path).stem
            logger.info(f"Анализ чата: {chat_name}")
            
            df = self.load_and_process_json(json_path)
            df = self.preprocess_df(df)
            
            pdf_path = self.output_dir / f"report_{chat_name}.pdf"
            with PdfPages(pdf_path) as pdf_pages:
                # Титульная страница
                fig = plt.figure(figsize=(11, 8.5))
                plt.text(0.5, 0.5, f"Анализ чата: {chat_name}", 
                        ha='center', va='center', fontsize=20)
                plt.axis('off')
                pdf_pages.savefig(fig)
                plt.close()
                
                # Выполняем анализы
                self.analyze_active_users(df, chat_name, pdf_pages)
                self.analyze_time_patterns(df, chat_name, pdf_pages)
                self.analyze_text_content(df, chat_name, pdf_pages)
                self.analyze_network(df, chat_name, pdf_pages)
                self.additional_analysis(df, chat_name, pdf_pages)
            
            logger.info(f"PDF отчет сохранен в {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при анализе чата: {str(e)}")
            return False

def main():
    print("=== Анализатор чатов ===")
    json_path = input("Введите путь к JSON-файлу с данными чата: ").strip('"')
    
    if not os.path.exists(json_path):
        print(f"Ошибка: файл не найден - {json_path}")
        return
    
    analyzer = ChatAnalyzer()
    if analyzer.analyze_chat(json_path):
        print("Анализ успешно завершен! Проверьте папку 'output'") 
    else:
        print("Произошла ошибка при анализе. Проверьте логи для деталей.")

if __name__ == "__main__":
    main()