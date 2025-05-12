from pathlib import Path
from typing import Optional
import json
import pandas as pd

class DictToDataFrameParser:
    '''
    Атрибуты:
    path (Path): Путь к JSON-файлу, из которого будет загружен словарь.
    columns (dict): Словарь, где ключи соответствуют ключам из JSON-структуры, 
            а значения — названиям столбцов в первичном DataFrame.
    result_dict (dict): Десериализованный словарь, полученный из JSON-файла.
    df (DataFrame): DataFrame, созданный на основе данных из result_dict и columns
    name: Название чата

    Методы:
    load_json(path): Загружает и десериализует JSON-файл.(статистический)
    build_df_chat(dict_to_df, columns): Строит DataFrame из десериализованного словаря, 
                                         используя заданные столбцы.(статистический)
    save_to_csv(file_path): Сохраняет DataFrame в CSV-файл по заданному пути.
    '''
    def __init__(self, path: Path, columns: Optional[dict[str, str]] = None):
        '''
        :param path: Путь к JSON-файлу для загрузки данных.
        :param columns: Словарь сопоставления ключей из 'messages' с названиями столбцов для первичного DataFrame.
        '''
        self.path = path
        if columns is None:
            self.columns = {'date': 'date', 'id':'message_id','type':'type', 'from_id': 'from_id', 
                'actor_id':'actor_id' , 'reply_to_message_id': 'reply_to_message_id','from': 'user_name', 'actor':'user_name_actor'}
        else: 
            self.columns = columns
        self.result_dict = DictToDataFrameParser.load_json(path)  # Здесь вызываем метод через имя класса
        self.df = DictToDataFrameParser.build_df_chat(self.result_dict, self.columns)

        self.name = self.result_dict.get('name')
        
    @staticmethod
    def load_json(path: Path) -> dict:
         with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)

    #  создает  дата фрейм из словаря
    @staticmethod
    def build_df_chat( dict_to_df : dict, columns: dict[str,str]) -> pd.DataFrame:
        df_dict = {
        **{columns.get(col): [dict_to_df['messages'][x].get(col) for x in range(len(dict_to_df['messages']))]
            for col in columns
        },
        'text': [ ' '.join( i.get('text') for  i in dict_to_df['messages'][x]['text_entities'])
            for x in range( len(dict_to_df['messages']))
        ]}
           
        df = pd.DataFrame(df_dict)
        df['date'] =  pd.to_datetime(df['date'])
        if 'user_name_actor' in df.columns:
            df['user_name'] = df['user_name'].fillna(df['user_name_actor'])
            del df['user_name_actor']
        df['user_name'] = df['user_name'].fillna('')
        df['chat_name'] =dict_to_df.get('name')
        df['chat_id'] =dict_to_df.get('id')
        column =list(df.columns)
        df = df[column[:1]+ column[-2:]+column[1:-2]]

        if 'user_name' in df.columns:
            df['first_name'] = df['user_name'].apply( lambda x:  x.split()[0] if len( x.split())>0 else '')
            df['last_name'] = df['user_name'].apply( lambda x: x.split()[1] if len( x.split())>1 else '' )
            column =list(df.columns)
            df = df[column[:-3]+column[-2:]+[column[-3]]]

        if 'actor_id' in df.columns:
            df['from_id'] = df['from_id'].fillna(df['actor_id'])
            del df['actor_id']
            # del df['user_name_actor']

        if 'from_id' in df.columns:
            df['sender_id'] = df['from_id'].str.extract(r'(\d+)')
            df['sender_type'] = df['from_id'].str.replace(r'\d+', '', regex=True)
            del df['from_id']
            column =list(df.columns)
            df=df[column[:3]+column[-2:]+column[3:-2]]
        
            # df = df.fillna(0)
            df[df.select_dtypes(include='number').columns]= df.select_dtypes(include='number').fillna(-1).astype('int64')
        return df

    def save_to_csv(self, name_file: str, path: Path = Path.cwd()) -> None:
        '''
        Сохраняет DataFrame в CSV файл по заданному пути.
        :param name_file: Имя файла (без расширения).
        :param path: Путь, по которому будет сохранён файл. По умолчанию используется текущая рабочая директория.
        
        '''
        path_file = (path/name_file).with_suffix('.csv')
        self.df.to_csv(path_file, index=False)
