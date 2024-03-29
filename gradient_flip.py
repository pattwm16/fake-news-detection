import tensorflow as tf
from keras.engine import Layer
import keras.backend as K
# import tensorflow.keras.backend as K

#################################
# CREDIT: https://github.com/michetonu/gradient_reversal_keras_tf
# TODO: potentially alter the code to be our own
#################################

# @tf.custom_gradient
# def grad_reverse(x):
#     y = tf.identity(x)
#     def custom_grad(dy):
#         return -dy
#     return y, custom_grad

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    # g = K.get_session().graph
    # g = K.get_default_graph()
    # g = tf.compat.v1.keras.backend.get_session().graph()
    with tf.Graph().as_default() as g:
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
