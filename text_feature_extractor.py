import pandas as pd
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding, concatenate
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from keras.initializers import Constant

def text_extract(x_train, x_test, y_train, y_test, word_index):
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
    embedding_matrix = np.zeros((num_words, embedding_dimension))
    for word, i in word_index.items():
        if i >= words_to_keep:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # NOTE: we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dimension,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=sequence_length,
                                trainable=False)

    # Text Feature Extractor Model ---
    # hyperparameters
    num_filters = 128
    num_window = 10

    # Model Architecture
    sequence_input = Input(shape=(sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # this for loop constructs an arbitrary number of parallel conv1d layers
    # with window size ranging from 1 to num_window
    kernels = []
    for i in range(1, num_window):
        window = Conv1D(filters=num_filters, kernel_size=i, activation='relu')(embedded_sequences)
        window = GlobalMaxPooling1D()(window)
        kernels.append(window)
    concat = concatenate(kernels)

    return sequence_input, concat
