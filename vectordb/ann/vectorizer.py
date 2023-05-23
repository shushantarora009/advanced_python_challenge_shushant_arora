# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:46:02 2023

@author: shushant
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


class Vectorizer:
    def __init__(self):
        self.vectorizers = {}

    def vectorize(self, data, data_type: dir):
        return self.transform(data, data_type, init_vectorizers=True)

    def transform(self, data, data_type: dir, init_vectorizers=False):
        text_columns = [key for key, value in data_type.items()
                        if value == 'text']
        numeric_columns = [key for key, value in data_type.items()
                           if value == 'number']
        numeric_data = data[numeric_columns].fillna(0).values
        vectors = None
        for text_column in text_columns:
            text_data = data[text_column].fillna(
                '').astype(str).values.flatten()
            if init_vectorizers:
                vectorizer = TfidfVectorizer()
                self.vectorizers[text_column] = vectorizer
                vector = vectorizer.fit_transform(text_data)
            else:
                vectorizer = self.vectorizers[text_column]
                vector = vectorizer.transform(text_data)
            
            if vectors is None:
                vectors = hstack([numeric_data.astype(float), vector])
            else:
                vectors = hstack([vectors, vector])
        return vectors
