# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:54:50 2023

@author: shushant
"""

import duckdb
from vectordb.ann.vectorizer import Vectorizer
from vectordb.ann.balltree_ann import BallTreeAnn
from vectordb.ann.model.query import Query
import numpy as np
import pandas as pd
from docarray import DocumentArray

def load_table():
    return duckdb.sql('SELECT * FROM "data/dataset.csv"').df()

def plot_search_result(df,search_result):
    print(f'search_result.ind {search_result.ind}')
    df_filtered = df.iloc[search_result.ind.flatten().tolist()]
    da = DocumentArray.from_dataframe(df_filtered)
    da.plot_image_sprites()

def main():
    if __name__ == "__main__":
        print('started vector db demo')
        df = load_table()
        print(f'df {df}')
        vectorizer = Vectorizer()
        data_type = {'color': 'text',
                     'country': 'text',
                     'width': 'number',
                     'height': 'number',
                     'brand': 'text'}
        vectors = vectorizer.vectorize(df, data_type)
        print(f'vectors {vectors}')
        balltree_ann = BallTreeAnn()
        balltree_ann.encode(vectors)
        arr = np.array([['Blue','CA',1650,1926,'Thirty Five Kent']])
        filter_df = pd.DataFrame(arr, columns=data_type.keys())
        filter_vector = vectorizer.transform(filter_df, data_type)
        filter_query = Query(v=filter_vector)
        search_result = balltree_ann.find(query=filter_query, n=3)
        print(f'search_result {search_result}')
        plot_search_result(df, search_result)
        
