from nltk.tokenize import word_tokenize
from keras.models import load_model
import numpy as np


def encode(word):

    encoding_dict = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4,
        'e': 5, 'f': 6, 'g': 7, 'h': 8,
        'i': 9, 'j': 10, 'k': 11, 'l': 12,
        'm': 13, 'n': 14, 'o': 15, 'p': 16,
        'q': 17, 'r': 18, 's': 19, 't': 20,
        'u': 21, 'v': 22, 'w': 23, 'x': 24,
        'y': 25, 'z': 26
    }
    chars = [char for char in word]
    encoded = []
    for char in chars:
        encoded.append(encoding_dict[char])
    return encoded + [0]*(15 - len(encoded))


def roundup(value):
    if value <= 0.92:
        return 0
    else:
        return 1


def tag(sentence):

    model = load_model('model.h5')
    tokens = word_tokenize(sentence)

    tagged = []
    for token in tokens:
        enc = np.array([encode(token)])
        if roundup(model.predict(enc)[0]) == 0:
            tagged.append(token + "\\bn")
        else:
            tagged.append(token + "\\en")

    return " ".join(tagged)


def collect(sentence):

    model = load_model('model.h5')
    tokens = word_tokenize(sentence)

    bn_words, en_words = [], []
    for token in tokens:
        enc = np.array([encode(token)])
        if roundup(model.predict(enc)[0]) == 0:
            bn_words.append(token)
        else:
            en_words.append(token)

    return bn_words, en_words