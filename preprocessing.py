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
    # hyperparamters for tokenizer
    words_to_keep = 4860
    sequence_length = 540
    embedding_dimension = 100

    # read in data as pandas dataframe
    train_data = pd.read_csv(train_path, sep='\t', header=None, names=header_names)
    test_data = pd.read_csv(test_path, sep='\t', header=None, names=header_names)
    valid_data = pd.read_csv(valid_path, sep='\t', header=None, names=header_names)

    # create header names for pandas dataframe
    header_names = ['ID', 'Label', 'Statement', 'Subjects', 'Speaker',
                    'Speaker Job', 'State info', 'Party Affiliation',
                    'Barely true count', 'False count', 'Half true count',
                    'Mostly true count', 'Pants on fire count', 'Context']

    # concatenate all data to be split after processing
    data = pd.concat([train_data, test_data, valid_data], axis=0)

    # create binary label (see below)
    #       1: false, barely-true, pants-fire
    #       0: true, mostly-true
    data["Binary Label"] = np.zeros(len(data["Label"]))
    data["Binary Label"] = train_data["Label"].replace(
        to_replace={'mostly-true': 0, 'false': 1, 'barely-true': 1, 'true': 0, 'half-true': 1, 'pants-fire': 1})

    data = data.drop(labels=['Barely true count', 'False count', 'Half true count',
                             'Mostly true count', 'Pants on fire count', 'Label'], axis=1)

    # pull out text from statement column and clearn
    text = data["Statement"]
    text = [re.split(r'\W+', i) for i in text]

    # create 2D labels vector
    # "Binary Label" is label for fake news detector
    # "Subjects" is label for Event Discriminator
    labels = data[["Binary Label","Subjects"]]

    # instantiate tokenizer, fit on texts from dataset, and extract word index
    tokenizer = Tokenizer(num_words=words_to_keep)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index

    # Below is code used to determine sequence_length given our corpus
    # length=0
    # for i in sequences:
    #     if len(i) > length:
    #         length = len(i)
    # print(length)

    # pad all sequences to be same length
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

    # 80/20 train/test split
    x_train, x_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2)

    # need to return: data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events
    return x_train, x_test, y_train[["Binary Label"]], y_test[["Binary Label"]], y_train[["Subjects"]], y_test[["Subjects"]], word_index, unique_events
