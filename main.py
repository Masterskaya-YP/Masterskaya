from module import DictToDataFrameParser
from pathlib import Path
import json
import pandas as pd

from IPython.display import display
folder_ignore = '.ipynb_checkpoints'
path_list =[  i for i  in Path.cwd().rglob('*.json') if folder_ignore not in list(i.parent.parts) ]
with pd.option_context('display.max_colwidth', None):
    display(pd.DataFrame({'path':list(map(str, path_list))}))
index =int(input('Введите номер строки для парсинга JSON: '))  

# словарь ключ - ключ из словаря  messages  значение соответствующее название столбца  df   
columns ={'date': 'date', 'id':'message_id','type':'type', 'from_id': 'from_id', 'actor_id':'actor_id' , 'reply_to_message_id': 'reply_to_message_id',
         'from': 'user_name', 'actor':'user_name_actor'}
df_1= DictToDataFrameParser(path_list[index],columns)
with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
    display(df_1.path)
    display(df_1.df.head())
df_1.save_to_csv('df_1_proba')    