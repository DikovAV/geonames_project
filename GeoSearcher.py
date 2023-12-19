import sys

from typing import Optional, List

import pandas as pd
import numpy as np

from googletrans import Translator
from spellchecker import SpellChecker

from sentence_transformers import SentenceTransformer, util
import torch



class GeoSearcher():
    
    def __init__(self,
                 path_or_connection,
                 model: Optional[str] = 'LaBSE',
                 mode: Optional[str] = 'sql',
                 list_of_countrys: Optional[List[str]] = None,
                 translator: Optional[bool] = False):
        
        self.path_or_connection = path_or_connection
        self.translator = translator

        if mode=='sql':
            self.connection = path_or_connection
            self.sql_reader()
        elif mode=='csv':
            pass
        else:
            print('Тип подключения не определен')
            sys.exit()

        if list_of_countrys is not None:
            self.df = self.df.loc[self.df['country'].isin(list_of_countrys)]

        try:
            self.model = SentenceTransformer(model)
        except:
            self.model = SentenceTransformer('LaBSE')

        self.generate_embeddings()


    def sql_reader(self):

        query_geonames = 'SELECT * FROM city_15000'
        query_countyinfo = 'SELECT * FROM countryinfo'
        query_adminnames = 'SELECT * FROM admin_names'

        geonames = pd.read_sql(query_geonames,
                              con=self.connection,
                              index_col='geonameid'
                            )
        countrys = pd.read_sql(query_countyinfo,
                               con=self.connection,
                            )
        
        alter_names =  pd.read_sql(query_adminnames,
                                  con=self.connection,
                                )
        self.df = self.__merge_data(geonames=geonames,
                                    countrys=countrys,
                                    alter_names=alter_names) 
    
    def generate_embeddings(self):
        self.df['asciiname_embeddings'] = self.model.encode(
            self.df['asciiname'].values
            ).tolist()
        self.unique_embeddings_matrix = torch.tensor(
            np.stack(self.df['asciiname_embeddings'].values),
            dtype=torch.float32
            )
    
    def match_name(self,
                   query: List[str],
                   number_of_matching: Optional[int] = 5):
        result_list = []
        if type(query) == str:
            query = [query]
        for val in query:
            if self.translator:
                language, translated_text = self.detect_and_translate(val)
                try:
                    spell = SpellChecker(language=language)
                    translated_text = spell.correction(translated_text)
                    if translated_text is None:
                        translated_text = val
                except:
                    pass
            else:
                translated_text = val
            
            vector = torch.tensor(
                self.model.encode(translated_text),
                dtype=torch.float32
            )
            self.similarity = util.cos_sim(vector, self.unique_embeddings_matrix)
            self.values_indices = self.similarity.sort()
            self.max_positions = self.values_indices[1][0][-number_of_matching:]
            self.max_positions = torch.flip(self.max_positions, [0])
            self.top_n_recommendations = self.df.iloc[
                self.max_positions
            ]
            result_list.append(
                {'name':self.top_n_recommendations["asciiname"].values,
                 'region' : self.top_n_recommendations["region"].values,
                 'country' : self.top_n_recommendations["country"].values,
                 'similarity' : torch.flip(self.values_indices[0][0][-number_of_matching:],[0])
                })
        return result_list

    @staticmethod
    def __merge_data(geonames : pd.DataFrame,
                     countrys : pd.DataFrame,
                     alter_names : pd.DataFrame):
        countrys['iso'] = countrys['iso'].str.strip()
        countrys.rename(columns={'country':'country_names'}, inplace=True)
        df = geonames.merge(countrys[['iso','country_names']],
                            left_on='country',
                            right_on='iso',
                            how='left')
        df['admin_code'] = df['iso'] + '.' + df['admin1']
        df = df.merge(alter_names[['admin_code','name_1']],
                      left_on='admin_code',
                      right_on='admin_code')
        df.rename(columns={'name_1':'region',
                           'country':'country_short',
                           'country_names' : 'country'},
                  inplace=True)
        return df
    

    def detect_and_translate(self, text, target_language='en'):
        try:
            tranl = Translator()

            detected_language = tranl.detect(text).lang

            translated_text = tranl.translate(text, dest=target_language)

            return detected_language, translated_text.text
        except:
            return 'ru', text
        
        

        