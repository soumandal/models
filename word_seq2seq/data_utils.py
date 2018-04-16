import pickle
from collections import Counter

PATH = "PATH"
en_vocab_size, bn_vocab_size = 1000, 1000



def read_sentences(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as reader:
        for s in reader:
            sentences.append(s.strip())
    return sentences



def create_dataset(en_sentences, bn_sentences):

    en_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in en_sentences for word in sentence.split())
    bn_vocab_dict = Counter(word.strip('৷') for sentence in bn_sentences for word in sentence.split())

    en_vocab = map(lambda x: x[0], sorted(en_vocab_dict.items(), key=lambda x: -x[1]))
    bn_vocab = map(lambda x: x[0], sorted(bn_vocab_dict.items(), key=lambda x: -x[1]))

    en_vocab = list(en_vocab)[:en_vocab_size]
    bn_vocab = list(bn_vocab)[:bn_vocab_size]

    start_idx = 2
    en_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(en_vocab)])
    en_word2idx["<unk>"] = 0
    en_word2idx["<pad>"] = 1

    en_idx2word = dict([(idx, word) for word, idx in en_word2idx.items()])

    start_idx = 4
    bn_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(bn_vocab)])
    bn_word2idx["<unk>"] = 0
    bn_word2idx["<go>"] = 1
    bn_word2idx["<eos>"] = 2
    bn_word2idx["<pad>"] = 3

    bn_idx2word = dict([(idx, word) for word, idx in bn_word2idx.items()])

    x = [[en_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in en_sentences]
    y = [[bn_word2idx.get(word.strip('৷'), 0) for word in sentence.split()] for sentence in bn_sentences]

    X, Y = [], []
    for i in range(len(x)):
        n1, n2 = len(x[i]), len(y[i])
        n = n1 if n1 < n2 else n2
        if abs(n1 - n2) <= 0.3 * n:
            if n1 <= 15 and n2 <= 15:
                X.append(x[i])
                Y.append(y[i])

    return X, Y, en_word2idx, en_idx2word, en_vocab, bn_word2idx, bn_idx2word, bn_vocab



def save_dataset(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, -1)



def read_dataset(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)



def main():

    en_sentences = read_sentences(PATH + "\\" + "en.txt")
    bn_sentences = read_sentences(PATH + "\\" + "bn.txt")

    save_dataset(PATH + "\\" + "data.pkl", create_dataset(en_sentences, bn_sentences))


if __name__ == '__main__':
    main()
