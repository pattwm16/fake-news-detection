from preprocessing import get_data
from text_feature_extractor import text_extract
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K
from gradient_flip import GradientReversal
from math import floor
from keras.callbacks import LearningRateScheduler


# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

# appy get_data from preprocessing.py and call
data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events = get_data(train_path, test_path, valid_path, verbose=True)

# apply text feature extractor (text_extract) from text_feature_extractor
sequence_input, concat = text_extract(data_train, data_test, labels_train, labels_test, word_index)

# NOTE: Since the size of the fake news detector was relatively small, we did
#       not create a new file for it.
# Fake News Detector Layers ---
preds = Dense(1, activation='softmax', name="fake_news_detector")(concat) # concat is output from text feature extractor (in R^T space)
# fake_news_detector = Model(sequence_input, preds, name="Fake News Detector")

# Compile and fit fake news detector
# fake_news_detector.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=0.002),
#               metrics=['acc'])
# print(fake_news_detector.summary())
# fake_news_detector.fit(data_train, labels_train,
#           batch_size=250,
#           epochs=5,
#           validation_data=(data_test, labels_test))

# Event Discriminator Layers ---
hl_size = 120 # TODO: Tune this hyperparameter
neg_lambda = 1 # the trade-off
# TODO: Implement GRL here (identity on forward, multiply by neg lambda on backprop)
# HINT: See https://github.com/michetonu/gradient_reversal_keras_tf for implmentation
ed_reversal = GradientReversal(neg_lambda)(concat)
ed_fc1_out = Dense(hl_size, activation='relu')(ed_reversal)
ed_fc2_out = Dense(unique_events, activation='softmax', name="event_discriminator")(ed_fc1_out)
print("unique_events", unique_events)
# event_discrim = Model(sequence_input, ed_fc2_out, name="Event Discriminator")

# Compile and fit the event discriminator
# event_discrim.compile(loss='categorical_crossentropy',
#               optimizer=Adam(learning_rate=0.002),
#               metrics=['acc'])
# print(event_discrim.summary())
# print(len(data_test))
# print(len(subjects_test))
# print(subjects_test.shape)
# event_discrim.fit(data_train, subjects_train,
#           batch_size=250,
#           epochs=5,
#           validation_data=(data_test, subjects_test))

########################################################################################
# TODO:
#	- change final loss calculation (default (current) is addition, but we want subtraction)
#	- tune hyperparameters
#	- check functionality of gradient reversal layer
########################################################################################

# parameter for training
EPOCHS = 5
BATCH_SZ = 250
iterations = EPOCHS * floor(data_train.shape[0] / BATCH_SZ)
decay_rate = 10 / iterations

# Define learning rate decay schedule
def decay_lr(epoch):
	initial_lrate = 2e-3
	# alpha and beta are defined in paper
	alpha = 10
	beta = 0.75
	# p is measure of training progress
	# TODO: Is this based on epochs or # of examples seen?
	p = float(epoch / EPOCHS)
	lrate = initial_lrate / ((1 + (alpha*p))**beta)
	return lrate

# Compile and fit the integrated model
f_model = Model(inputs=[sequence_input], outputs=[preds,ed_fc2_out])
f_model.compile(loss={"fake_news_detector" : 'binary_crossentropy', "event_discriminator" : 'categorical_crossentropy'},
              optimizer=Adam(),
              metrics=['acc'])

# Create learning rate scheduler
lrate = LearningRateScheduler(decay_lr)
callbacks_list = [lrate]

# Print model summary
print(f_model.summary())

# Fit model
f_model.fit(data_train, {"fake_news_detector" : labels_train, "event_discriminator" : subjects_train},
      batch_size=BATCH_SZ,
          epochs=EPOCHS,
          validation_data=(data_test, {"fake_news_detector" : labels_test, "event_discriminator" : subjects_test}),
          callbacks=callbacks_list
          )
# fnd_grad = K.gradients(fake_news_detector.output, fake_news_detector.trainable_weights)
