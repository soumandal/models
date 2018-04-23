from keras.layers import Dense, Input, Lambda, LSTM, Embedding
from keras.models import Model
from keras import backend as K
from encoder import encode
import numpy as np


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def euclidean(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_network(input_shape):
    inp = Input(shape=input_shape)
    encoder = Embedding(output_dim=15, input_dim=27)(inp)
    encoder = LSTM(128, activation='relu', return_sequences=True)(encoder)
    encoder = LSTM(128, activation='relu', return_sequences=True)(encoder)
    encoder = Dense(1, activation='sigmoid')(encoder)
    return Model(inp, encoder)

input_shape = (15, )
base_network = create_base_network(input_shape)

l_input = Input(shape=input_shape)
r_input = Input(shape=input_shape)

l_twin = base_network(l_input)
r_twin = base_network(r_input)

distance = Lambda(euclidean, output_shape=output_shape)([l_twin, r_twin])

siamese = Model([l_input, r_input], distance)
siamese.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
siamese.load_weights("siamese_weights.h5")

w1 = "bad"
w2 = "had"

l_encoding = [encode(w1)]
r_encoding = [encode(w2)]

l_encoding = np.array(l_encoding, dtype=int)
r_encoding = np.array(r_encoding, dtype=int)

x = siamese.predict([l_encoding, r_encoding])

print(x)
