# preprocessing
# Will Patterson
import pandas as pd
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
from gensim.models import Word2Vec
import api


# DATA LAYOUT ----
# Column 1: the ID of the statement ([ID].json).
# Column 2: the label.
# Column 3: the statement.
# Column 4: the subject(s).
# Column 5: the speaker.
# Column 6: the speaker's job title.
# Column 7: the state info.
# Column 8: the party affiliation.
# Column 9-13: the total credit history count, including the current statement.
# 9: barely true counts.
# 10: false counts.
# 11: half true counts.
# 12: mostly true counts.
# 13: pants on fire counts.
# Column 14: the context (venue / location of the speech or statement).
# -----

# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

#parameters
words_to_keep = 9000
sequence_length = 540


header_names = ['ID', 'Label', 'Statement', 'Subjects', 'Speaker',
                'Speaker Job', 'State info', 'Party Affiliation',
                'Barely true count', 'False count', 'Half true count',
                'Mostly true count', 'Pants on fire count', 'Context']

train_data = pd.read_csv(train_path, sep='\t', header=None, names=header_names)
test_data = pd.read_csv(test_path, sep='\t', header=None, names=header_names)
valid_data = pd.read_csv(valid_path, sep='\t', header=None, names=header_names)

data = pd.concat([train_data, test_data, valid_data], axis=0)

data["Binary Label"] = np.zeros(len(data["Label"]))
data["Binary Label"] = train_data["Label"].replace(to_replace={'mostly-true':0, 'false':1, 'barely-true':1, 'true':0, 'half-true':1, 'pants-fire':1})

data = data.drop(labels=['Barely true count', 'False count', 'Half true count',
                'Mostly true count', 'Pants on fire count', 'Label'], axis = 1)

text = data["Statement"]
text = [re.split(r'\W+', i) for i in text]
print(text[0:5])

labels = data["Binary Label"]

tokenizer = Tokenizer(num_words=words_to_keep)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index

# length=0
# for i in sequences:
#     if len(i) > length:
#         length = len(i)
# print(length)

data_pad = pad_sequences(sequences, maxlen=sequence_length)
labels = keras.utils.to_categorical(np.asarray(labels))

#data pad only has the statement column, and nothing else
#figure out later how to include rest of features
x_train, x_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2)

#thought maybe we could remove the topic of abortion and test on it
# print(list(data["Subjects"]).count("abortion"))

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


##Attempt at training our own embedding
# model = Sequential()
# model.add(Embedding(input_dim=len(word_index), output_dim=100, input_length=540))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# print(model.summary())
# model.fit(data_pad, labels, epochs=5)
# loss, accuracy = model.evaluate(data_pad, labels)

# print(accuracy)

# output_array = model.predict(data_pad)
# print(np.shape(output_array))

# word2vec = Word2Vec(text, size=30, window=5, min_count=2, iter=30)
# print(word2vec.wv["Boston"])

