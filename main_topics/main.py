# pip install openpyxl
# pip install llama-cpp
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent/'src'))

import gc
import warnings
import nltk
import torch
import  numpy as np
import re
import pandas as pd
import os
import tiktoken
import threading
import time
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer, pipeline
from nltk.corpus import stopwords 

from  module import DictToDataFrameParser, InitBertopic

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# stop words для 'Vectorizers in BERTopic'
try:
    russian_stopwords = stopwords.words("russian") 
except:
    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")

FOLDER_INPUT = 'input_data'
# Создадим папку  'input_data' в  родительском каталоге проекта
path_dir_input = (Path(__file__).resolve().parent.parent)/FOLDER_INPUT
path_dir_input.mkdir(parents=True, exist_ok=True)


###################################################################################   

def time_work(time_last: float) -> None:
    time_diff = time.time() - time_last
    minutes = int(time_diff // 60)
    seconds = int(time_diff % 60)
    print(f'Время: {minutes} мин. и {seconds} сек.')    

####################################################################################
# функция активности
stop_event = threading.Event()
def spin():
    i = True
    while not stop_event.is_set():
     if i:
        print("\rI compute ...",end='')
        i=False 
        time.sleep(2)
     else:
        print("\rwaiting ...",end='') 
        i=not i 
        time.sleep(2) 
######################################################################################

pattern  = r'[^\w,.\!? ]'
def cleaned_string(string: str, pattern: str = pattern) -> str: 
    '''
    По умолчанию  удаляются все символы, не являющиеся буквами, цифрами, пробелами и знаками 
    препинания.Замена гиперссылок на  подстроку "гиперссылка"
    '''
    string = re.sub(r'https?://\S+', 'здесь_гиперсссылка', string)
    return re.sub(pattern, '', string).strip()    
#######################################################################################

#  Подготовка  текста без   учета  зависимости цепочек ответов (Гипотеза -тексты независмые)
def text_preparation_with_replay( df: pd.DataFrame, len_text: str = 3) -> tuple[list[int],list[str]]:
    ''' Подготавливает текстовые данные для передачи в BERTopic: применяет к каждому тексту функцию `cleaned_string`,
        и возвращает два списка:
        - список индексов для кластеризации,
        - очищенные строки (тексты), которые можно передать в модель.
        len_text -минимальная длина текста(количество символов)
    '''
    if not isinstance(df, pd.DataFrame):
       raise TypeError("Expected DataFrame, got another type.")
    if  'text'not in df.columns:
       raise KeyError("Column 'text' is missing in the DataFrame")
   
    text = list(df[df['text'].str.len() >= len_text]['text'].items())
    return  map(list,zip(*text)) 
##########################################################################################

## Парсим json
try:
    df_1 = DictToDataFrameParser(next(path_dir_input.glob('*.json')))
except:
    print(f'Проверь папку {FOLDER_INPUT} Возможно нет JSON-файла в папке.')
    sys.exit()
    
print(df_1.name)


LEN_TEXT = 10
index_df, text = text_preparation_with_replay(df_1.df, len_text= LEN_TEXT)
assert len(text) == len(index_df), (f'Lengths do not match: {len(text)} != {len(index_df)}')


# Параметры для Bertopic:
# модель для Embedding Models
# model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = "DeepPavlov/rubert-base-cased-conversational"
#параметры модели понижения размерности
umap_dict = dict(n_neighbors=3, n_components=5, min_dist=0.0, metric='cosine')
#параметры модели кластеризации
hdbscan_dict = dict(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
#параметры для Vectorizers
vectorizer_dict = dict(stop_words=russian_stopwords, ngram_range=(1, 3), min_df=1)
#параметры для TF_IDF
tf_idf_dict = dict(seed_words=["ссылка", "яндекс", "практикум", "стажировка", "курсы",'вакансия'],
                   seed_multiplier=2
                  )
                  
# nr_topics - количество  тем(ориентировочно)
# top_n_words - количество наиболее значимых слов для каждой темы.

NR_TOPIC = 15
TOP_N_WORDS = 10
bertopic_model = InitBertopic(
    nr_topics=NR_TOPIC,
    top_n_words=TOP_N_WORDS,
    model=model,
    umap_dict=umap_dict, 
    hdbscan_dict=hdbscan_dict, 
    vectorizer_dict=vectorizer_dict, 
    tf_idf_dict=tf_idf_dict
)

thread = threading.Thread(target=spin)
thread.start()
time_last = time.time()
topics, probs = bertopic_model.topic_model.fit_transform(text)

if 'topic' in df_1.df.columns:
    df_1.df = df_1.df.drop(columns=['topic', 'probs'], errors='ignore')
df_1.df = df_1.df.join(pd.DataFrame({'topic': topics,'probs':np.round(probs,3)}, index=index_df))


stop_event.set()
thread.join()
time_work(time_last)
print("Bertopic_OK")

############################ LLM ################################
model_path = hf_hub_download(
    repo_id="Ronny/YandexGPT-5-Lite-8B-instruct-Q8_0-GGUF",
    filename="yandexgpt-5-lite-8b-instruct-q8_0.gguf"
)
WINDOW_SIZE = 2304
max_tokens=2048
llama = Llama(model_path=model_path, n_ctx=WINDOW_SIZE,verbose=False)

enc = tiktoken.get_encoding("cl100k_base") # Для подсчета токенов, чтобы уложиться в размер окна
df_top =df_1.df.dropna()
df_top = df_top.sort_values(by='probs', ascending=False)

def count_cut_text( text, max_tokens ):
        tokens = enc.encode(text)[:max_tokens]
        return enc.decode(tokens)
topic_info = bertopic_model.topic_model.get_topic_info().set_index('Topic')# df с результомработы  Bertopic
##################################### LOOP LLM WORK
time_last = time.time()
stop_event.clear()
thread = threading.Thread(target=spin)
thread.start()

topic=[]
for i in tqdm(df_top['topic'].unique()):
         
    text = ' '.join(df_top[df_top['topic']== i ]['text'].to_list())
    text = count_cut_text(text, max_tokens)
    key_words = ','.join(topic_info.loc[i,'Representation'])
    text = cleaned_string(text)
    prompt_1 = f'Из списка сообщений выдели одну общую тему. Cообщения: {text}'
    prompt_2=f''' Проанализируй тексти keywords. Определи и сформулируй одну главную тему.
         {text},  keywords{key_words}. Ответ  начни так : Главная тема:
        '''
    output = llama(
    prompt_2,
    max_tokens=300,
    stop=["</s>"], 
    temperature=0.5,
    top_p=0.95
)
    topic.append((output['choices'][0]['text'].strip(), i))

stop_event.set()
thread.join()
del llama
gc.collect()    
time_work(time_last)
print("LLM_yandexGPT5.0_OK")

####################################################################################################

topic = [ (i[0].replace('Главная тема:', '').strip().capitalize() , int(i[1])) for i in topic ]

df_topic_list = pd.DataFrame(topic,columns=['Main topic','Topic'] )
df_topic_list = df_topic_list.merge(bertopic_model.topic_model.get_topic_freq(), how='left', on = 'Topic')
df_topic_list = df_topic_list.merge(topic_info['Representation'], how='left', on = 'Topic')
df_topic_list = df_topic_list.rename(columns={'Representation': 'Keywords'})
df_topic_list= df_topic_list[['Topic','Main topic','Count','Keywords']].sort_values(by='Count',ascending=False)

df_final = df_1.df.merge(df_topic_list, how='left', left_on ='topic', right_on='Topic')
df_final = df_final.drop('Topic', axis = 1)
df_final = df_final.dropna()


################ to write data to an Excel file ############################

FOLDER_OUTPUT = 'output'
# Создадим папку  'output' в  родительском каталоге проекта
path_dir_output = (Path(__file__).resolve().parent.parent)/FOLDER_OUTPUT
path_dir_output.mkdir(parents=True, exist_ok=True)


def save_excel( df: pd.DataFrame, name_file: str, suffix: str) -> None:
   name = Path(name_file).with_suffix(suffix)
   path_file = path_dir_output/name
   df.to_excel(path_file, index=False, engine='openpyxl')


TOPIC='topic' 
CHAT_TOTAL ='chat_total'
SUFFIX = '.xlsx'   

save_excel(df_final, CHAT_TOTAL,SUFFIX)
save_excel(df_topic_list, TOPIC,SUFFIX)