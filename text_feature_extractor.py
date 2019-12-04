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

def text_feature_extractor():
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
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dimension,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=sequence_length,
                                trainable=False)

    num_filters = 128
    num_window = 10

    sequence_input = Input(shape=(sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    kernels = []
    for i in range(1, num_window):
        window = Conv1D(filters=num_filters, kernel_size=i, activation='relu')(embedded_sequences)
        window = GlobalMaxPooling1D()(window)
        kernels.append(window)
    concat = concatenate(kernels)
    preds = Dense(2, activation='softmax')(concat)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    model.fit(x_train, y_train,
              batch_size=50,
              epochs=5,
              validation_data=(x_test, y_test))



