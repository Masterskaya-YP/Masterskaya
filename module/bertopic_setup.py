from typing import Optional
from transformers import AutoTokenizer, pipeline
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
import numpy as np

from bertopic.backend import BaseEmbedder
from transformers import pipeline

class CustomEmbedder(BaseEmbedder):
    '''
    Назначение класса — обернуть инициализированный pipeline("feature-extraction", model=model)
      в кастомный класс Bertropic, обеспечивающий: 
      -корректную обрезку входных последовательностей по длине токенов,
      -устранение конфликтов при согласовании размерностей выходных эмбеддингов с модулем
        уменьшения размерности (Dimensionality Reduction) в BERTopic.
    '''

    def __init__(self, model):
        super().__init__()
        
        self.embedding_model = pipeline("feature-extraction", model=model)
        self.embedding_model.tokenizer.model_max_length = 512

    def embed(self, documents, verbose=False):
        # Извлечение эмбеддингов для списка документов
        embeddings = self.embedding_model(documents, truncation=True, padding=True, max_length=512)
        return np.array([embeddings[i][0][0] for i in range(len(embeddings))])








class InitBertopic:
    '''
    Инициализирует Bertopic c заданными парамтрами,
     атрибут topic_model-инициализированная модель Bertopic:
     <name_model>.topic_model.fit_transform(text: list[str]) - кластеризация
     <name_model>.topic_model.get_topic_info() - результат
     <name_model>.topic_model.get_topic([int]) - ключевые слова топика

     Параметры инициализации:

     # модель для Embedding Models
     model = "DeepPavlov/rubert-base-cased-conversational"
     
    #параметры модели понижения размерности
    umap_dict = dict(n_neighbors=3, n_components=5, min_dist=0.0, metric='cosine')
    
   #параметры модели кластеризации
   hdbscan_dict = dict(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
   
   #параметры для Vectorizers
   ectorizer_dict = dict(stop_words=russian_stopwords, ngram_range=(1, 3), min_df=1)
   
   #параметры для TF_IDF
   tf_idf_dict = dict(seed_words=["ссылка", "яндекс", "практикум", "стажировка", "курсы",'вакансия'],
                   seed_multiplier=2)

   # nr_topics - количество  тем(ориентировочно)
   # top_n_words - количество наиболее значимых слов для каждой темы.                
 '''
    
    def __init__(self,
             nr_topics: Optional[int] = None,
             top_n_words: int = 10,   
             model: Optional[str] = None, 
             umap_dict: Optional[dict] = None, 
             hdbscan_dict: Optional[dict] = None,
             vectorizer_dict: Optional[dict] = None, 
             tf_idf_dict: Optional[dict] = None) -> None:

        self.nr_topics = nr_topics
        self.top_n_words = top_n_words
        self.model = model
        self.umap_dict = umap_dict
        self.hdbscan_dict = hdbscan_dict
        self.vectorizer_dict = vectorizer_dict
        self.tf_idf_dict =  tf_idf_dict
        self.bertopic_dict, self.topic_model = self.init_bertopic()
        
    
    def init_bertopic(self):
        # embedding_model = pipeline("feature-extraction", model=self.model)
        embedding_model = CustomEmbedder(self.model)

        bertopic_dict = dict(embedding_model=embedding_model, nr_topics= self.nr_topics, top_n_words=self.top_n_words)
            
        if self.umap_dict:
            umap_model = UMAP(**self.umap_dict)
            bertopic_dict['umap_model'] = umap_model
            
        if self.hdbscan_dict:
             hdbscan_model = HDBSCAN(**self.hdbscan_dict)
             bertopic_dict['hdbscan_model'] =  hdbscan_model
            
        if  self.vectorizer_dict:
            vectorizer_model = CountVectorizer(**self.vectorizer_dict)
            bertopic_dict['vectorizer_model'] =   vectorizer_model
            
        if  self.tf_idf_dict:
            ctfidf_model = ClassTfidfTransformer(**self.tf_idf_dict)
            bertopic_dict['ctfidf_model'] =  ctfidf_model

        
        return bertopic_dict, BERTopic(**bertopic_dict)
        
