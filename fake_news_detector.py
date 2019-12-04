# fake news detector
# calculate loss for fake news detector
# Jaja Sothanaphan
import tensorflow as tf

class fake_news_detector(tf.keras.layers.Layer):
    def __init__(self):
        super(fake_news_detector, self).__init__()

        # TODO: probably should be in main
        # self.lr = self.lr / ((1 + 10 * self.p) ** 0.75)
        self.optimizer = tf.keras.optimizers.SGD(2e-3)

        self.preds = tf.keras.layers.Dense(1, activation='softmax')

    # takes in text+image (multi modal features) as input and return probabilities of being fake news
    @tf.function
    def call(self, concat):

        return self.preds(concat)

    # loss for fake news detector
    # multi_modal_feat_rep is of shape (-1,hidden_size)
    # labels is a list/numpy of integers of shape = (label_size,)
    def loss(self, concat, labels):

        prbs = self.call(multi_modal_feat_rep)
        loss = tf.keras.losses.binary_crossentropy(labels,prbs)
        return tf.reduce_sum(loss)
