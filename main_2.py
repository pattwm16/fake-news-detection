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
import sys

class Extractor_Model(tf.keras.Model):
   def __init__(self, embedding_matrix):
       super(Extractor_Model, self).__init__()

       #hyperparameters
       num_filters = 10 # TODO: change back to 100
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
       learning_rate = 0.001
       decay_rate = learning_rate / ((1 + 10 * np.random.randint(0, 2)) ** 0.75)
       self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

   @tf.function
   def call(self, inputs):
       embedded_sequences = self.model.call(inputs)

       conv_2 = self.conv_2(embedded_sequences)
       conv_3 = self.conv_3(embedded_sequences)
       conv_4 = self.conv_4(embedded_sequences)
       conv_5 = self.conv_5(embedded_sequences)
       conv_6 = self.conv_6(embedded_sequences)
       conv_2 = self.global_max_pool(conv_2)
       conv_3 = self.global_max_pool(conv_3)
       conv_4 = self.global_max_pool(conv_4)
       conv_5 = self.global_max_pool(conv_5)
       conv_6 = self.global_max_pool(conv_6)

       concat = layers.concatenate([conv_2, conv_3, conv_4, conv_5, conv_6])

       return concat

   @tf.function
   def loss_function(self, detector, discriminator, lambd):
       return detector + discriminator

class Detector_Model(tf.keras.Model):
   def __init__(self):
       super(Detector_Model, self).__init__()

       #model
       self.model = Sequential()
       self.model.add(layers.Dense(2, activation='softmax'))

       #loss
       self.loss = tf.keras.losses.BinaryCrossentropy()

       #optimizer
       learning_rate = 0.001
       decay_rate = learning_rate / ((1 + 10 * np.random.randint(0, 2)) ** 0.75)
       self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

   @tf.function
   def call(self, concat):
       return self.model(concat)

   @tf.function
   def loss_function(self, probs, labels):
       return tf.reduce_mean(self.loss(probs, labels))

class Discriminator_Model(tf.keras.Model):
   def __init__(self, unique_events):
       super(Discriminator_Model, self).__init__()

       #hyperparameters
       hp_lambda=1

       #model
       self.model = Sequential()
       self.model.add(GradientReversal(hp_lambda, name="reversal"))
       self.model.add(layers.Dense(100, activation='relu', name="dense1"))
       self.model.add(layers.Dense(unique_events, activation='softmax', name="dense2"))

       #loss
       self.loss = tf.keras.losses.CategoricalCrossentropy()

       #optimizer
       learning_rate = 0.001
       decay_rate = learning_rate/((1+10*np.random.randint(0,2))**0.75)
       self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

   @tf.function
   def call(self, concat):
       return self.model(concat)
   @tf.function
   def loss_function(self, probs, labels):
       return self.loss(probs, labels)

# Define learning rate decay schedule
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

def train(data_train, labels_train, subjects_train, extractor, detector, discriminator):
   EPOCHS = 150
   BATCH_SZ = 128
   DATA_TRAIN_SIZE = data_train.shape[0]
   ITERATIONS = floor(DATA_TRAIN_SIZE / BATCH_SZ)
   neg_lambda= 1

   for ep in range(EPOCHS):
       print("Epoch " + str(1+ep) + "/" + str(EPOCHS))
       epoch_accuracy = 0
       count = 0
       final_loss = 0
       for i in range(ITERATIONS):

           batch_data = data_train[i * BATCH_SZ : (i+1) * BATCH_SZ]
           batch_label = labels_train[i * BATCH_SZ : (i+1) * BATCH_SZ]
           batch_label = tf.cast(batch_label, tf.float32)
           batch_subject = subjects_train[i * BATCH_SZ : (i+1) * BATCH_SZ]

           with tf.GradientTape(persistent=True) as tape:
               concat = extractor.call(batch_data)

               output_d = detector.call(concat)

               output_e = discriminator.call(concat)

               loss_d = detector.loss(output_d, batch_label)

               loss_e = discriminator.loss(output_e, batch_subject)

               loss_f = extractor.loss_function(loss_d, loss_e, neg_lambda)

           grad_f = tape.gradient(loss_f, extractor.trainable_variables)
           grad_e = tape.gradient(loss_e, discriminator.trainable_variables)
           grad_d = tape.gradient(loss_d, detector.trainable_variables)

           extractor.optimizer.apply_gradients(zip(grad_f, extractor.trainable_variables))
           discriminator.optimizer.apply_gradients(zip(grad_e, discriminator.trainable_variables))
           detector.optimizer.apply_gradients(zip(grad_d, detector.trainable_variables))

           epoch_accuracy+= accuracy(output_d, batch_label)
           count+= 1
           final_loss += loss_f

           print(i, "iteration final loss", final_loss / count)
           print(i, "iteration accuracy", epoch_accuracy / count)

       print(ep, "epoch final loss", final_loss/count)
       print(ep, "epoch accuracy", epoch_accuracy/count)

def test():
   pass

def accuracy(preds, labels):
   return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.cast(tf.reshape(labels, [128]), tf.int64)), tf.float32))


def main():
   # path to training dataset
   train_path = "liar_dataset/train.tsv"
   test_path = "liar_dataset/test.tsv"
   valid_path = "liar_dataset/valid.tsv"

   data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events = get_data(
       train_path, test_path, valid_path, verbose=True)
   embedding_matrix = text_extract(word_index)

   extractor = Extractor_Model(embedding_matrix)
   detector = Detector_Model()
   discriminator = Discriminator_Model(unique_events)

   train(data_train, labels_train, subjects_train, extractor, detector, discriminator)


if __name__ == "__main__":
   main()
