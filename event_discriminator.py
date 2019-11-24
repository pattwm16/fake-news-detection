# event discriminator
# calculate loss for event discriminator
# Jaja Sothanaphan
import tensorflow as tf

class event_discriminator(tf.keras.layers.Layer):
    def __init__(self, hidden_size, event_num):
        super(event_discriminator, self).__init__()
        
        # TODO: probably should be in main
        self.trade_off = 1

        self.hidden_size = hidden_size
        self.event_num = event_num
        # TODO: probably should use the same opt as fake news detector (for optimal parameter update, minimax)
        self.optimizer = tf.keras.optimizers.SGD(2e-3)

        self.reverse = lambda x: -self.trade_off * x
        self.fc_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(self.event_num, activation='softmax')
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # takes in text+image (multi modal features) as input and return probabilities of being classified as each event
    @tf.function
    def call(self, multi_modal_feat_rep):
        op = self.reverse(multi_modal_feat_rep)
        op = self.fc_1(op)
        op = self.fc_2(op)
        return op

    # loss for event discriminator
    # multi_modal_feat_rep is of shape (-1,hidden_size)
    # labels is a list/numpy of integers (shape = [label_size,])
    def loss(self, multi_modal_feat_rep, labels):

        prbs = self.call(multi_modal_feat_rep)
        loss = self.loss_func(labels,prbs)
        return tf.reduce_sum(loss)

