# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:59:47 2023

@author: shushant
"""
from vectordb.ann.model.query import Query
from vectordb.ann.model.search_result import SearchResult


class BaseAnn:
    def encode(self,X):
        pass

    def find(self, query: Query, n: int) -> SearchResult:
        pass
