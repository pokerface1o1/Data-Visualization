#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 00:18:30 2017

@author: Pranjal
"""

import glob
import codecs
import multiprocessing
import re
import nltk

book_names = sorted(glob.glob('religious-and-philosophical-texts/*.txt'))

corpus = ""
for name in book_names:
    file = codecs.open(name, 'r', 'utf-8')
    corpus += file.read()

corpus = corpus.split('\n')    
 
data = []
for line in corpus:    
    text = re.sub('[^a-zA-Z0-9]',' ',line)
    text = nltk.word_tokenize(text)
    data.append(text)

#delete empty rows     
data = list(filter(None, data))

import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE

num_workers = multiprocessing.cpu_count()

word2vec = w2v.Word2Vec(
              sg = 1,
              seed = 1,
              workers = num_workers,
              size = 300,
              min_count = 3,
              window = 7,
              sample = 1e-3
                    )

word2vec.build_vocab(data)

word2vec.train(data, total_examples=word2vec.corpus_count,epochs=word2vec.iter)

word_vector_matrix = word2vec.wv.syn0

tsne = TSNE(2, random_state = 0)

matrix_2d = tsne.fit_transform(word_vector_matrix)

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, matrix_2d[word2vec.wv.vocab[word].index])
            for word in word2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
    

sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))

