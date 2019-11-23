# visual feature extractor
# Will Patterson
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Lambda
from keras.models import Model

# this model should take images as input
# it should return softmax-ed probabilities

# hyperparameters ----
# p is the dimension of the final visual feature representation
# TODO: Tune this
p = 32
# hidden size, defined in paper
hidden_size = 32
# number of classes in vgg19
num_classes = 1000
# input shape, must be tuple with 3 channels
inp_shape = (224,224,3)

# model ----
# include_top is false to remove final classification layer
# this allows us to add additional dense layer
vgg19_model = keras.applications.vgg19.VGG19(include_top=False,
                                                weights='imagenet',
                                                input_shape=inp_shape,
                                                pooling=None)
print("VGG19 model instantiated")
x = Flatten()(vgg19_model.output)

# paper prevented back prop through vgg19 model
x_1_stop_grad = Lambda(lambda x: tf.stop_gradient(x))(x)
x_1 = Dense(hidden_size)(x_1_stop_grad)
x_1 = LeakyReLU(alpha=0.3)(x)

# softmax is applied on last layer
predictions = Dense(p, activation = 'softmax')(x_1)

#create graph of new model
image_extractor = Model(input = vgg19_model.input, output = predictions)

#compile the model
image_extractor.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# provide model summary
image_extractor.summary()
