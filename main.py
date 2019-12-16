import tensorflow as tf
from preprocessing import get_data
from text_feature_extractor import text_extract
from keras.models import Sequential
import keras.layers as layers
import keras.initializers as initializers
from gradient_flip import GradientReversal
from math import floor
from text_feature_extractor_2 import text_extract
import numpy as np
from sklearn.utils import shuffle

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
        learning_rate = 0.001
        decay_rate = learning_rate/((1 + 10 * np.random.randint(0, 2)) ** 0.75)
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
    def loss_function(self, detector, discriminator):
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
        decay_rate = learning_rate/((1 + 10 * np.random.randint(0, 2)) ** 0.75)
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
        self.model.add(GradientReversal(hp_lambda))
        self.model.add(layers.Dense(100, activation="relu"))
        self.model.add(layers.Dense(unique_events, activation='softmax'))

        #loss
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        #optimizer
        learning_rate = 0.001
        decay_rate = learning_rate/((1 + 10 * np.random.randint(0, 2)) ** 0.75)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

    @tf.function
    def call(self, concat):
        return self.model(concat)
    @tf.function
    def loss_function(self, probs, labels):
        return self.loss(probs, labels)

def train(extractor, detector, discriminator, data_train, labels_train, subjects_train):
    EPOCHS = 5
    BATCH_SZ = 128
    DATA_TRAIN_SIZE = data_train.shape[0]
    ITERATIONS = floor(DATA_TRAIN_SIZE / BATCH_SZ)

    acc_array = []
    loss_array = []

    for ep in range(EPOCHS):
        print("Epoch " + str(1+ep) + "/" + str(EPOCHS))
        iteration_count = 0
        epoch_accuracy = 0
        final_loss = 0

        for i in range(ITERATIONS):

            data_train, labels_train, subjects_train = shuffle(data_train, labels_train, subjects_train)

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
                loss_f = extractor.loss_function(loss_d, loss_e)

            grad_f = tape.gradient(loss_f, extractor.trainable_variables)
            grad_e = tape.gradient(loss_e, discriminator.trainable_variables)
            grad_d = tape.gradient(loss_d, detector.trainable_variables)

            extractor.optimizer.apply_gradients(zip(grad_f, extractor.trainable_variables))
            discriminator.optimizer.apply_gradients(zip(grad_e, discriminator.trainable_variables))
            detector.optimizer.apply_gradients(zip(grad_d, detector.trainable_variables))

            iteration_count+=1
            epoch_accuracy+= accuracy(output_d, batch_label)
            final_loss += loss_f

        acc_array.append(epoch_accuracy/iteration_count)
        loss_array.append(final_loss/iteration_count)

        print(ep, "epoch final loss", final_loss/iteration_count)
        print(ep, "epoch accuracy", epoch_accuracy/iteration_count)

    print("train loss", tf.reduce_mean(loss_array), loss_array)
    print("train accuracy", tf.reduce_mean(acc_array), acc_array)

def test(extractor, detector, discriminator, data_test, label_test, subject_test):
    BATCH_SZ = 128
    DATA_TEST_SIZE = data_test.shape[0]
    ITERATIONS = floor(DATA_TEST_SIZE / BATCH_SZ)

    total_accuracy = 0
    final_loss = 0
    count = 0

    for i in range(ITERATIONS):
        batch_data = data_test[i * BATCH_SZ: (i + 1) * BATCH_SZ]
        batch_label = label_test[i * BATCH_SZ: (i + 1) * BATCH_SZ]
        batch_label = tf.cast(batch_label, tf.float32)
        batch_subject = subject_test[i * BATCH_SZ: (i + 1) * BATCH_SZ]

        concat = extractor.call(batch_data)

        output_d = detector.call(concat)
        output_e = discriminator.call(concat)

        loss_d = detector.loss(output_d, batch_label)
        loss_e = discriminator.loss(output_e, batch_subject)
        loss_f = extractor.loss_function(loss_d, loss_e)

        total_accuracy += accuracy(output_d, batch_label)
        final_loss += loss_f
        count += 1

    print("test loss", final_loss/count)
    print("test accuracy", total_accuracy/count)

def accuracy(preds, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.cast(tf.reshape(labels, [128]), tf.int64)), tf.float32))


def main():
    # path to training dataset
    train_path = "liar_dataset/train.tsv"
    test_path = "liar_dataset/test.tsv"
    valid_path = "liar_dataset/valid.tsv"

    data_train, data_test, labels_train, labels_test, subjects_train, subjects_test, word_index, unique_events = \
        get_data(train_path, test_path, valid_path, verbose=True)
    embedding_matrix = text_extract(word_index)

    extractor = Extractor_Model(embedding_matrix)
    detector = Detector_Model()
    discriminator = Discriminator_Model(unique_events)

    train(extractor, detector, discriminator, data_train, labels_train, subjects_train)

    test(extractor, detector, discriminator, data_test, labels_test, subjects_test)

if __name__ == "__main__":
    main()
