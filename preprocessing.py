# preprocessing
# Will Patterson
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import numpy as np
from sklearn.model_selection import train_test_split

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
data["Tokenized_Statement"] = text
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
print('Shape of data tensor:', data_pad.shape)
print('Shape of label tensor:', labels.shape)

#data pad only has the statement column, and nothing else
#figure out later how to include rest of features
x_train, x_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2)

print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))
print(np.shape(y_test))

