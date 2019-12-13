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

# Helper functions
def stringify_list(lst):
    """
    takes a list of a data and returns a list of strins
    :param data: data to be unified as a list
    :returns: data as a list with each element as string
    """
    cleaned_lst = []
    for elem in lst:
        if type(elem) != str:
            cleaned_lst.append(str(elem))
        else:
            cleaned_lst.append(elem)
    return cleaned_lst

def get_data(train_path, test_path, valid_path, verbose=False):
    # hyperparamters for tokenizer
    words_to_keep = 4860
    sequence_length = 540
    embedding_dimension = 100

    # create header names for pandas dataframe
    header_names = ['ID', 'Label', 'Statement', 'Subjects', 'Speaker',
                    'Speaker Job', 'State info', 'Party Affiliation',
                    'Barely true count', 'False count', 'Half true count',
                    'Mostly true count', 'Pants on fire count', 'Context']

    # read in data as pandas dataframe
    train_data = pd.read_csv(train_path, sep='\t', header=None, names=header_names)
    test_data = pd.read_csv(test_path, sep='\t', header=None, names=header_names)
    valid_data = pd.read_csv(valid_path, sep='\t', header=None, names=header_names)


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

    # NOTE: stringify_list is defined above, it is needed as some values of
    #       the "Subjects" column are of type float and cannot be split
    # TODO: Fix SettingWithCopyWarning returned by this line
    list_of_all_subj = [x.split(',')[0].lower() for x in stringify_list(labels.Subjects)]
    labels["Subjects"] = list_of_all_subj
    # this determines K from paper. Event discriminator is a K classification problem.
    # NOTE: should be 142 unique events in total dataset
    unique_events = list(set(labels["Subjects"]))

    # TODO: One-hot encode subjects before returning (e.g have columns of length # of unique events)
    # create dictionary of subjects in entire dataset (142 keys)
    dict_of_subjects = {i:unique_events[i] for i in range(0, len(unique_events))}
    rev_dict_of_subjects = {v:k for k,v in dict_of_subjects.items()}
    subbed_subjects = [rev_dict_of_subjects.get(item,item) for item in labels["Subjects"]]
    labels["Subjects"] = subbed_subjects


    # TODO: Make sure to update the labels inputs parameter in train_test_split to
    #       have one hot encoded that you implemented above.

    # 80/20 train/test split where y_train/test is label data
    x_train, x_test, y_train, y_test = train_test_split(data_pad, labels, test_size=0.2)


    one_hot_y_train = keras.utils.to_categorical(np.asarray(y_train["Subjects"]))
    one_hot_y_test = keras.utils.to_categorical(np.asarray(y_test["Subjects"]))

    # print out diagnostic data for preprocessing.py
    if verbose:
        print("Preprocessing run successfully")
        print("|---------------------------------------------------|")
        print("| Preprocessing run successfully")
        print("| {} data points ({}/{} train/test split)".format(len(data_pad), len(x_train), len(x_test)))
        print("| {} unique events in dataset".format(len(unique_events)))
        print("|---------------------------------------------------|")

    # need to return: data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events
    return x_train, x_test, y_train[["Binary Label"]].values, y_test[["Binary Label"]].values, one_hot_y_train, one_hot_y_test, word_index, len(unique_events)
