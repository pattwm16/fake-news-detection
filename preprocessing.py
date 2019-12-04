import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import re

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

def get_data(train_path, test_path, valid_path):
    # paramters
    words_to_keep = 4860
    sequence_length = 540
    embedding_dimension = 100

    header_names = ['ID', 'Label', 'Statement', 'Subjects', 'Speaker',
                    'Speaker Job', 'State info', 'Party Affiliation',
                    'Barely true count', 'False count', 'Half true count',
                    'Mostly true count', 'Pants on fire count', 'Context']

    train_data = pd.read_csv(train_path, sep='\t', header=None, names=header_names)
    test_data = pd.read_csv(test_path, sep='\t', header=None, names=header_names)
    valid_data = pd.read_csv(valid_path, sep='\t', header=None, names=header_names)

    data = pd.concat([train_data, test_data, valid_data], axis=0)

    data["Binary Label"] = np.zeros(len(data["Label"]))
    data["Binary Label"] = train_data["Label"].replace(
        to_replace={'mostly-true': 0, 'false': 1, 'barely-true': 1, 'true': 0, 'half-true': 1, 'pants-fire': 1})

    data = data.drop(labels=['Barely true count', 'False count', 'Half true count',
                             'Mostly true count', 'Pants on fire count', 'Label'], axis=1)

    text = data["Statement"]
    text = [re.split(r'\W+', i) for i in text]

    labels = data[["Binary Label","Subjects"]]

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

    # TODO: This must be changed to only look at first subject (before first comma)
    #       You should use Regex or x.split(). We couldn't figure out what was wrong.
    #       We want this to return only the first subject for each row, since some have
    #       multiple subjects.
    # unique_events = [ x.split(',')[0] for x in labels.Subjects.unique()]
    # print("Unique Events", unique_events)
    # print("Len unique events", len(unique_events))
    # unique_events = len(unique_events)
    unique_events = len(labels.Subjects.unique())

    # TODO: One-hot encode subjects before returning (e.g have columns of length # of unique events)
    #subject_labels = keras.utils.to_categorical(np.asarray(labels[1]))


    # TODO: Make sure to update the labels inputs parameter in train_test_split to
    #       have one hot encoded that you implemented above.

    x_train, x_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2)

    return x_train, x_test, y_train[["Binary Label"]], y_test[["Binary Label"]], y_train[["Subjects"]], y_test[["Subjects"]], word_index, unique_events
