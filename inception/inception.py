import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.layers import Dropout, Flatten
from keras.models import Model

with open("bn_encodings.pkl", "rb") as f:
    bn_data = pickle.load(f)
with open("en_encodings.pkl", "rb") as g:
    en_data = pickle.load(g)
data = bn_data[:6632] + en_data[:6632]
data = np.array(data, dtype=int)

target = []
for x in range(6632):
    target.append([0])
for y in range(6632):
    target.append([1])
target = np.array(target, dtype=int)

combined = list(zip(data, target))
random.Random(10).shuffle(combined)

data[:], target[:] = zip(*combined)


def inception():

    inp_layer = Input(shape=(15, ))
    embedding = Embedding(27, 15, input_length=15)(inp_layer)

    # SUBNET 1
    conv_1 = Conv1D(filters=32, kernel_size=2, strides=1, activation='relu')(embedding)
    drop_1 = Dropout(0.2)(conv_1)
    pool_1 = MaxPooling1D(pool_size=2)(drop_1)
    flat_1 = Flatten()(pool_1)

    # SUBNET 2
    conv_2 = Conv1D(filters=32, kernel_size=3, strides=1, activation='relu')(embedding)
    drop_2 = Dropout(0.2)(conv_2)
    pool_2 = MaxPooling1D(pool_size=2)(drop_2)
    flat_2 = Flatten()(pool_2)

    # SUBNET 3
    conv_3 = Conv1D(filters=32, kernel_size=4, strides=1, activation='relu')(embedding)
    drop_3 = Dropout(0.2)(conv_3)
    pool_3 = MaxPooling1D(pool_size=2)(drop_3)
    flat_3 = Flatten()(pool_3)

    # SUBNET 4
    lstm_stack_1 = LSTM(15, return_sequences=True)(embedding)
    lstm_stack_2 = LSTM(35, return_sequences=True)(lstm_stack_1)
    lstm_stack_3 = LSTM(25, return_sequences=True)(lstm_stack_2)
    flat_4 = Flatten()(lstm_stack_3)

    # MERGE SUBNET OUTPUTS
    merged = concatenate([flat_1, flat_2, flat_3, flat_4])

    # OUTPUT LAYER
    outputs = Dense(15, activation='relu')(merged)
    out_layer = Dense(1, activation='sigmoid')(outputs)
    model = Model(inputs=inp_layer, outputs=out_layer)

    # COMPILE
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

inception_net = inception()
history = inception_net.fit(data, target, epochs=50, batch_size=128)
inception_net.save("inception_lid.h5")

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
