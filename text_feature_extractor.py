import pandas as pd
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, concatenate, Concatenate
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from keras.initializers import Constant

def text_extract(word_index):
    # hyperparameters
    words_to_keep = 4860
    sequence_length = 540
    embedding_dimension = 100

    # load in pretrained embeddings from GLOVE
    embeddings_index = {}
    with open('glove.6B.100d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    num_words = min(words_to_keep, len(word_index) + 1)
    print(num_words)
    embedding_matrix = np.zeros((num_words, embedding_dimension))
    for word, i in word_index.items():
        if i >= words_to_keep:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix