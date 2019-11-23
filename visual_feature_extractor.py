# visual feature extractor
# Will Patterson
import keras
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.models import Model

# this model should take images as input
# it should return softmax-ed probabilities

# hyperparameters ----
# p is the dimension of the final visual feature representation
# TODO: Tune this
p = 32
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
x = Dense(p)(x)
x = LeakyReLU(alpha=0.3)(x)
#x = Dropout(0.5)(x)
#x = BatchNormalization()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

#create graph of new model
head_model = Model(input = vgg19_model.input, output = predictions)

#compile the model
head_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# provide model summary
head_model.summary()
