__author__ = 'Guillaume'

from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, \
    merge, RepeatVector, TimeDistributed, LSTM
from keras.optimizers import SGD, Adam


def CNN_3x3(grid_shape, lr=0.1, dueling=True, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3)(grid)
    x = Activation("relu")(x)
    # Output layer
    if dueling:
        a = Convolution2D(4, 1, 1)(x)
        a = Flatten()(a)

        v = Convolution2D(1, 1, 1)(x)
        v = Flatten()(v)
        v = RepeatVector(4)(v)
        v = Flatten()(v)

        q = merge([a,v], mode='sum')
    else:
        q = Convolution2D(4, 1, 1)(x)
        q = Flatten()(q)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def CNN_7x7(grid_shape, lr=0.1, dueling=True, adam=False):
    # Input node
    grid = Input(grid_shape)
    # Hidden layer
    x = Convolution2D(16, 3, 3, subsample=(2,2))(grid)
    x = Activation("relu")(x)
    x = Convolution2D(16, 3, 3)(x)
    x = Activation("relu")(x)
    # Output layer
    if dueling:
        a = Convolution2D(4, 1, 1)(x)
        a = Flatten()(a)

        v = Convolution2D(1, 1, 1)(x)
        v = Flatten()(v)
        v = RepeatVector(4)(v)
        v = Flatten()(v)

        q = merge([a,v], mode='sum')
    else:
        q = Convolution2D(4, 1, 1)(x)
        q = Flatten()(q)
    # Keras model
    neuralnet = Model(grid, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    if adam:
        neuralnet.compile(Adam(lr=lr), loss="mse")
    else:
        neuralnet.compile(SGD(lr=lr), loss="mse")
    return neuralnet


def RCNN_3x3(input_shape, lr=0.1):
    # Input node
    grids = Input(batch_shape=input_shape)
    #x = Masking(mask_value=0.)(grids)
    # Hidden layer
    x = LSTM(16, return_sequences=True, stateful=True)(grids)
    # Output layer
    q = TimeDistributed(Dense(4))(x)
    # Keras model
    neuralnet = Model(grids, q)
    # Compile
    #neuralnet.compile(Adam(lr=lr), loss="mse")
    neuralnet.compile(Adam(lr=lr), loss="mse")
    return neuralnet

