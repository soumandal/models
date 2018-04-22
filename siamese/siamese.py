from keras.layers import Dense, Input, Lambda, LSTM, Embedding, Merge
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import pickle, random
import numpy as np

with open("l_encodings.pkl", "rb") as a:
    l_encodings = pickle.load(a)
with open("r_encodings.pkl", "rb") as b:
    r_encodings = pickle.load(b)
with open("target.pkl", "rb") as c:
    target = pickle.load(c)

combined = list(zip(l_encodings, r_encodings, target))
random.Random(10).shuffle(combined)

l_encodings[:], r_encodings[:], target[:] = zip(*combined)
size = len(l_encodings)


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
siamese.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
history = siamese.fit([l_encodings, r_encodings], target, epochs=1, batch_size=500)
siamese.save("siamese.h5")
siamese.save_weights("siamese_weights.h5")
siamese.summary()

# ACCURACY
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# LOSS
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

