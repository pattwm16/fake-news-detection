from preprocessing import get_data
from text_feature_extractor import text_extract
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam


# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events = get_data(train_path, test_path, valid_path)

sequence_input, concat = text_extract(data_train, data_test, labels_train, labels_test, word_index)

# Fake news detector in functional form
preds = Dense(1, activation='softmax')(concat)

fake_news_detector = Model(sequence_input, preds, name="Fake News Detector")

# Event Discriminator Layers
hl_size = 120
ed_fc1_out = Dense(hl_size, activation='relu')(concat)
ed_fc2_out = Dense(unique_events, activation='softmax')(ed_fc1_out)

event_discrim = Model(sequence_input, ed_fc2_out, name="Event Discriminator")

# Compile and fit fake news detector
# fake_news_detector.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=0.002),
#               metrics=['acc'])
# print(fake_news_detector.summary())
# fake_news_detector.fit(data_train, labels_train,
#           batch_size=250,
#           epochs=5,
#           validation_data=(data_test, labels_test))


# Compile and fit the event discriminator
event_discrim.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.002),
              metrics=['acc'])
print(event_discrim.summary())
print(len(data_test))
print(len(subjects_test))
event_discrim.fit(data_train, subjects_train,
          batch_size=250,
          epochs=5,
          validation_data=(data_test, subjects_test))

# TODO: Implement the model integration and minimax
