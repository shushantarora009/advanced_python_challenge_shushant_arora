# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:53:28 2023

@author: shushant
"""
from vectordb.ann.base_ann import BaseAnn
from vectordb.ann.model.query import Query
from vectordb.ann.model.search_result import SearchResult
from sklearn.neighbors import BallTree

class BallTreeAnn(BaseAnn):
    def __init__(self):
        pass

    def encode(self, X):
        self.tree = BallTree(X.toarray(), leaf_size=2)

    def find(self, query: Query, n: int) -> SearchResult:
        dist, ind = self.tree.query(query.v.toarray(), k=n)
        return SearchResult(dist, ind)
