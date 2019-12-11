import tensorflow as tf
from preprocessing import get_data
from text_feature_extractor import text_extract
from keras.models import Sequential, Model
import keras.layers as layers
import keras.initializers as initializers
from keras.optimizers import Adam
import keras.backend as K
from gradient_flip import GradientReversal
from math import floor
from keras.callbacks import LearningRateScheduler
from text_feature_extractor_2 import text_extract
import keras
import numpy as np

class Extractor_Model(tf.keras.Model):
    def __init__(self, embedding_matrix):
        super(Extractor_Model, self).__init__()

        #hyperparameters
        num_filters = 100
        sequence_length = 540
        embedding_dimension = 100
        num_words = 4860

        #model
        self.model = Sequential()
        self.model.add(layers.Embedding(input_dim=num_words, output_dim=embedding_dimension,
                                        embeddings_initializer=initializers.Constant(embedding_matrix),
                                        input_length=sequence_length, trainable=False))

        self.conv_2 = layers.Conv1D(filters=num_filters, kernel_size=2, padding='same', activation='relu')
        self.conv_3 = layers.Conv1D(filters=num_filters, kernel_size=3, padding='same', activation='relu')
        self.conv_4 = layers.Conv1D(filters=num_filters, kernel_size=4, padding='same', activation='relu')
        self.conv_5 = layers.Conv1D(filters=num_filters, kernel_size=5, padding='same', activation='relu')
        self.conv_6 = layers.Conv1D(filters=num_filters, kernel_size=6, padding='same', activation='relu')

        self.global_max_pool = layers.GlobalMaxPooling1D()

        #optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        #loss ?????

    @tf.function
    def call(self, inputs):
        embedded_sequences = self.model.call(inputs)
        conv_2 = self.conv_2(embedded_sequences)
        conv_3 = self.conv_3(embedded_sequences)
        conv_4 = self.conv_4(embedded_sequences)
        conv_5 = self.conv_5(embedded_sequences)
        conv_6 = self.conv_6(embedded_sequences)

        concat = layers.concatenate([conv_2, conv_3, conv_4, conv_5, conv_6])

        return concat

    @tf.function
    def loss_function(self, probs, labels):
        pass

class Detector_Model(tf.keras.Model):
    def __init__(self):
        super(Detector_Model, self).__init__()

        #model
        self.model = Sequential()
        self.model.add(layers.Dense(1, activation='softmax'))

        #loss
        self.loss = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def call(self, concat):
        return self.model(concat)
    @tf.function
    def loss_function(self, probs, labels):
        return tf.reduce_mean(self.loss(probs, labels))

class Discriminator_Model(tf.keras.Model):
    def __init__(self, unique_events, hp_lambda):
        super(Discriminator_Model, self).__init__()

        #model
        self.model = Sequential()
        self.model.add(GradientReversal(hp_lambda))
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(unique_events, activation='softmax'))

        #loss
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def call(self, concat):
        return self.model(concat)
    @tf.function
    def loss_function(self, probs, labels):
        return tf.reduce_mean(self.loss(probs, labels))

# def optimize(tape, model, loss):
#     gradients = tape.gradient(loss, model.trainable_variables)
#     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #I don't think this will work

def accuracy(true, pred):
    return tf.reduce_mean(tf.argmax(pred, axis=1) == true)

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

# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

# hyperparameters
EPOCHS = 5
BATCH_SZ = 100
neg_lambda = 1

data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events = get_data(train_path, test_path, valid_path, verbose=True)
embedding_matrix = text_extract(word_index)

extractor = Extractor_Model(embedding_matrix)
detector = Detector_Model()
discriminator = Discriminator_Model(unique_events, neg_lambda)
# concat = extractor.call(data_train)

labels_train = labels_train.values
# subjects_train = subjects_train.values
labels_test = labels_test.values
# subjects_test = subjects_test.values

DATA_TRAIN_SIZE = data_train.shape[0]
ITERATIONS = floor(DATA_TRAIN_SIZE / BATCH_SZ)
for ep in range(EPOCHS):
    print("Epoch " + str(1+ep) + "/" + str(EPOCHS))
    for i in range(ITERATIONS):
        print(str((i+1) * BATCH_SZ) + "/" + str(DATA_TRAIN_SIZE))

        batch_data = data_train[i * BATCH_SZ: (i+1) * BATCH_SZ]
        batch_label = labels_train[i * BATCH_SZ: (i+1) * BATCH_SZ]
        batch_subject = subjects_train[i * BATCH_SZ: (i+1) * BATCH_SZ]

        lr = decay_lr(ep)

        with tf.GradientTape(persistent=True) as tape:
            concat = extractor.call(data_train)
            output_d = detector.call(concat)
            output_e = discriminator.call(concat)
            print(output_d.shape)
            loss_d = detector.loss(output_d, batch_label)
            loss_e = discriminator.loss(output_e, batch_subject)
            loss_f = loss_d - (neg_lambda * loss_e)

        grad_f = tape.gradient(loss_f, extractor.trainable_variables)
        grad_e = tape.gradient(loss_e, detector.trainable_variables)
        grad_d = tape.gradient(loss_d, discriminator.trainable_variables)
        Adam(lr).apply_gradients(zip(grad_f), extractor.trainable_variables)
        Adam(lr).apply_gradients(zip(grad_e), detector.trainable_variables)
        Adam(lr).apply_gradients(zip(grad_d), discriminator.trainable_variables)

        # test
        concat = extractor.call(data_test)
        output_d = detector.call(concat)
        output_e = discriminator.call(concat)
        acc_d = accuracy(output_d, labels_test)
        acc_e = accuracy(output_e, subjects_test)

        # print loss & accuracy
        tf.print(loss_d, loss_e, output_stream=sys.stdout, sep=',')
        tf.print(acc_d, acc_e, output_stream=sys.stdout, sep=',')
        # tf.print("fake_news_detector loss:", loss_d, "event_discriminator loss:", loss_e, "fake_news_detector accuracy:", acc_d, "event_discriminator accuracy:", acc_e, output_stream=sys.stdout, sep=',')


# def accuracy(true, pred):
#     return tf.reduce_mean(tf.argmax(pred, axis=1) == true)

# # Define learning rate decay schedule
# def decay_lr(epoch):
#     initial_lrate = 2e-3
#     # alpha and beta are defined in paper
#     alpha = 10
#     beta = 0.75
#     # p is measure of training progress
#     # TODO: Is this based on epochs or # of examples seen?
#     p = float(epoch / EPOCHS)
#     lrate = initial_lrate / ((1 + (alpha*p))**beta)
#     return lrate

#
# def train(l):
#     detector_model = Detector_Model(sequence_input, concat)
#     discriminator_model = Discriminator_Model(sequence_input, concat, neg_lambda=1, hl_sz=120, unique_events=unique_events)
#
#     detector_model.fit()
#
#     for i in range(ITERATIONS):
#         with tf.GradientTape(persistent=True) as tape:
#             detector_output = detector_model
#             discriminator_output = discriminator_model(concat)
#
#             detector_loss = detector_model.loss_function(detector_output, labels_train)
#             discriminator_loss = discriminator_model.loss_function(discriminator_output, subjects_train)
#
#         # call optimize on the generator and the discriminator
#         optimize(tape, detector_model, detector_loss)
#         optimize(tape, discriminator_model, discriminator_loss)
#
# train(concat, labels_train, subjects_train)

# # load pre-trained word embeddings into an Embedding layer
# # NOTE: we set trainable = False so as to keep the embeddings fixed
#
#
# # Text Feature Extractor Model ---
# # hyperparameters
# num_filters = 128
# num_window = 10
#
# # Model Architecture
# sequence_input = Input(shape=(sequence_length,), dtype='int32')
# # this for loop constructs an arbitrary number of parallel conv1d layers
# # with window size ranging from 1 to num_window
#
#
# return sequence_input, concat