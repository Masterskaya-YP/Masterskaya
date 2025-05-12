# chat_analyzer.py
import json
import logging
import os
import warnings
import re
import ast
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
        # Пути к папкам относительно расположения скрипта
        script_dir = Path(__file__).parent
        self.input_dir = script_dir.parent / "input_data"  # Папка input в корне проекта
        self.output_dir = script_dir.parent / "output"  # Папка output в корне проекта
        # self.lib_dir = script_dir.parent / "EDA" / "lib"

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
                df['text'] = df['text'].apply(ast.literal_eval)
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
        
        # 2. Визуализация кластеров (только если есть достаточно данных)
        if len(G) >= 3:
            try:
                # Выявляем топ-3 кластера
                communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
                top_clusters = sorted(communities, key=len, reverse=True)[:3]
                
                # Цветовая палитра для кластеров
                colors = cm.get_cmap('tab20', len(top_clusters))
                
                fig = plt.figure(figsize=(18, 12))
                pos = nx.spring_layout(G, k=0.8, iterations=200, seed=42)
                
                # Рисуем все узлы серым (фон)
                nx.draw_networkx_nodes(G, pos, node_size=50, node_color='#dddddd', alpha=0.5)
                nx.draw_networkx_edges(G, pos, edge_color='#cccccc', alpha=0.3, width=0.5)
                
                # Выделяем кластеры
                legend_handles = []
                for i, cluster in enumerate(top_clusters):
                    cluster_color = colors(i)
                    nx.draw_networkx_nodes(
                        G, pos, 
                        nodelist=cluster,
                        node_size=[np.log(G.nodes[n]['size'])*100 for n in cluster],
                        node_color=[cluster_color for _ in cluster],
                        edgecolors='white',
                        linewidths=0.5,
                        label=f'Кластер {i+1}'
                    )
                    # Подписи для топ-3 узлов в кластере
                    pagerank = nx.pagerank(G)
                    top_in_cluster = sorted(cluster, key=lambda x: pagerank[x], reverse=True)[:3]
                    for node in top_in_cluster:
                        plt.text(
                            pos[node][0], pos[node][1]+0.03,
                            node, 
                            fontsize=9, ha='center', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                        )
                    legend_handles.append(mpatches.Patch(color=cluster_color, 
                                                        label=f'Кластер {i+1} ({len(cluster)} участников)'))
                
                plt.title(
                    f'Топ-3 кластера в чате "{chat_name}"\n'
                    'Размер узла = активность, Цвет = принадлежность к кластеру',
                    fontsize=16, pad=20
                )
                plt.legend(handles=legend_handles, loc='upper right')
                plt.axis('off')
                pdf_pages.savefig(fig)
                plt.close()
                
                # 3. Детальная визуализация каждого кластера
                for i, cluster in enumerate(top_clusters, 1):
                    cluster_graph = G.subgraph(cluster)
                    fig = plt.figure(figsize=(12, 12))
                    
                    # Круговая диаграмма для кластера
                    pos_circular = nx.circular_layout(cluster_graph)
                    
                    nx.draw_networkx_nodes(
                        cluster_graph, pos_circular,
                        node_size=[np.log(cluster_graph.nodes[n]['size'])*200 for n in cluster_graph.nodes()],
                        node_color=colors(i-1),
                        alpha=0.9
                    )
                    
                    nx.draw_networkx_edges(
                        cluster_graph, pos_circular,
                        width=[0.5 + cluster_graph[u][v]['weight']*0.5 for u,v in cluster_graph.edges()],
                        edge_color='#666666',
                        alpha=0.6
                    )
                    
                    # Подписи без наложений
                    for node in cluster_graph.nodes():
                        plt.text(
                            pos_circular[node][0], pos_circular[node][1],
                            node, 
                            fontsize=8, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                        )
                    
                    plt.title(
                        f'Кластер #{i}\n'
                        f'Участников: {len(cluster)}\n'
                        f'Связей: {cluster_graph.number_of_edges()}',
                        fontsize=14
                    )
                    plt.axis('off')
                    pdf_pages.savefig(fig)
                    plt.close()
                    
            except Exception as e:
                logger.error(f"Ошибка при визуализации кластеров: {e}")
        
        # Интерактивная визуализация (сохраняется как HTML)
        net = Network(notebook=False, height="800px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
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
        # Средняя длина сообщения
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
        
        # Динамика сообщений по времени
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
            
            # Скользящее среднее
            rolling_avg = daily_counts.rolling(window=7).mean()
            
            fig = plt.figure(figsize=(14, 6))
            rolling_avg.plot()
            plt.title(f'Скользящее среднее (7 дней) активности в {chat_name}')
            plt.xlabel('Дата')
            plt.ylabel('Количество сообщений')
            plt.grid(True)
            pdf_pages.savefig(fig)
            plt.close()

    def compare_chats(self, df_list, chat_names, pdf_pages):
        """Сравнительный анализ чатов"""
        comparison = []
        
        for df, name in zip(df_list, chat_names):
            user_activity = df['from'].value_counts()
            time_patterns = self.analyze_time_patterns(df, name, pdf_pages)
            
            row = {
                'chat': name,
                'total_messages': len(df),
                'active_users': len(user_activity),
                'top_user': user_activity.index[0] if not user_activity.empty else None,
                'top_user_msgs': user_activity.iloc[0] if not user_activity.empty else 0,
                'peak_hour': time_patterns['hourly_activity'].idxmax() if not time_patterns['hourly_activity'].empty else None,
                'peak_day': time_patterns['daily_activity'].idxmax() if not time_patterns['daily_activity'].empty else None
            }
            
            if 'text_clean' in df.columns:
                word_counts = self.analyze_text_content(df, name, pdf_pages)
                if word_counts:
                    row['top_word'] = word_counts[0][0]
                    row['top_word_count'] = word_counts[0][1]
                else:
                    row['top_word'] = None
                    row['top_word_count'] = 0
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        # Визуализация сравнения (только если есть данные)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Общее количество сообщений
            comparison_df.plot(x='chat', y='total_messages', kind='bar', ax=axes[0, 0], legend=False)
            axes[0, 0].set_title('Общее количество сообщений')
            axes[0, 0].set_ylabel('Количество')
            
            # Количество активных пользователей
            comparison_df.plot(x='chat', y='active_users', kind='bar', ax=axes[0, 1], legend=False, color='orange')
            axes[0, 1].set_title('Количество активных пользователей')
            axes[0, 1].set_ylabel('Количество')
            
            # Пиковый час активности
            comparison_df.plot(x='chat', y='peak_hour', kind='bar', ax=axes[1, 0], legend=False, color='green')
            axes[1, 0].set_title('Пиковый час активности')
            axes[1, 0].set_ylabel('Час дня')
            
            # Топовые слова (если есть данные)
            if 'top_word_count' in comparison_df.columns and comparison_df['top_word_count'].sum() > 0:
                comparison_df.plot(x='chat', y='top_word_count', kind='bar', ax=axes[1, 1], legend=False, color='red')
                axes[1, 1].set_title('Частота топового слова')
                axes[1, 1].set_ylabel('Количество')
            else:
                axes[1, 1].axis('off')
                axes[1, 1].text(0.5, 0.5, 'Нет данных о словах', ha='center', va='center')
            
            plt.suptitle('Сравнительный анализ чатов', fontsize=16)
            plt.tight_layout()
            pdf_pages.savefig(fig)
            plt.close()
        except Exception as e:
            logger.error(f"Ошибка при создании сравнительных графиков: {e}")
        
        # Сохранение сравнения в таблицу
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=comparison_df.values,
                        colLabels=comparison_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title('Сравнительная таблица чатов', y=1.08)
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        
        return comparison_df
    
    def analyze_chat(self, json_path):
        """Основной метод анализа чата"""
        try:
            chat_name = Path(json_path).stem
            logger.info(f"Анализ чата: {chat_name}")
            
            df = self.load_and_process_json(json_path)
            df = self.preprocess_df(df)
            
            pdf_path = self.output_dir / f"report_EDA_{chat_name}.pdf"
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
    
    analyzer = ChatAnalyzer()
    
    # Проверяем наличие файлов в папке input
    json_files = list(analyzer.input_dir.glob("*.json"))
    
    if not json_files:
        print(f"Ошибка: в папке {analyzer.input_dir} не найдены JSON-файлы")
        return
    
    for json_path in json_files:
        print(f"Обработка файла: {json_path.name}")
        if analyzer.analyze_chat(json_path):
            print(f"Анализ завершен для {json_path.name}")
        else:
            print(f"Произошла ошибка при анализе {json_path.name}")
    
    print("Обработка всех файлов завершена! Проверьте папку 'output'")

if __name__ == "__main__":
    main()