import tensorflow as tf
import numpy as np
import pickle

SAVE = "PATH"

LATENT_DIMENSIONS = 256
WORD_EMBEDDING_SIZE = 100

with open(SAVE + "\\" + "en_word2idx.pkl", "rb") as a:
    en_word2idx = pickle.load(a)
with open(SAVE + "\\" + "bn_idx2word.pkl", "rb") as b:
    bn_idx2word = pickle.load(b)
with open(SAVE + "\\" + "bn_word2idx.pkl", "rb") as c:
    bn_word2idx = pickle.load(c)
with open(SAVE + "\\" + "input_seq_len.pkl", "rb") as d:
    input_seq_len = pickle.load(d)
with open(SAVE + "\\" + "output_seq_len.pkl", "rb") as e:
    output_seq_len = pickle.load(e)
with open(SAVE + "\\" + "en_vocab_size.pkl", "rb") as f:
    en_vocab_size = pickle.load(f)
with open(SAVE + "\\" + "bn_vocab_size.pkl", "rb") as g:
    bn_vocab_size = pickle.load(g)


with tf.Graph().as_default():

    encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='encoder{}'.format(i)) for i in range(input_seq_len)]
    decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='decoder{}'.format(i)) for i in range(output_seq_len)]

    w_t = tf.get_variable('proj_w', [bn_vocab_size, LATENT_DIMENSIONS], tf.float32)
    b = tf.get_variable('proj_b', [bn_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)

    outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                tf.contrib.rnn.BasicLSTMCell(LATENT_DIMENSIONS),
                                                num_encoder_symbols=en_vocab_size,
                                                num_decoder_symbols=bn_vocab_size,
                                                embedding_size=WORD_EMBEDDING_SIZE,
                                                feed_previous=True,
                                                output_projection=output_projection,
                                                dtype=tf.float32)

    outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

    en_sentences = []
    for line in open(SAVE + "\\" + "TEST_DATA.txt", encoding="utf-8").read().split("\n")[:-1]:
        en_sentences.append(line.strip())

    en_sentences_encoded = [[en_word2idx.get(word, 0) for word in en_sentence.split()] for en_sentence in en_sentences]

    for i in range(len(en_sentences_encoded)):
        en_sentences_encoded[i] += (input_seq_len - len(en_sentences_encoded[i])) * [en_word2idx['<pad>']]

    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint(SAVE + "\\" + 'checkpoints')

    outputs = open(SAVE + "\\" + "OUTPUT_DATA.txt", "w", encoding="utf-8")

    with tf.Session() as sess:
        saver.restore(sess, path)

        feed = {}
        for i in range(input_seq_len):
            feed[encoder_inputs[i].name] = np.array([en_sentences_encoded[j][i] for j in range(len(en_sentences_encoded))], dtype=np.int32)

        feed[decoder_inputs[0].name] = np.array([bn_word2idx['<go>']] * len(en_sentences_encoded), dtype=np.int32)

        output_sequences = sess.run(outputs_proj, feed_dict=feed)

        def softmax(x):
            n = np.max(x)
            e_x = np.exp(x - n)
            return e_x / e_x.sum()

        def decode_output(output_seq):
            words = []
            for i in range(output_seq_len):
                smax = softmax(output_seq[i])
                idx = np.argmax(smax)
                words.append(bn_idx2word[idx])
            return words

        for i in range(len(en_sentences_encoded)):
            ouput_seq = [output_sequences[j][i] for j in range(output_seq_len)]
            words = decode_output(ouput_seq)
            cleaned_words = [word for word in words if word not in ['<eos>', '<pad>', '<go>']]
            outputs.write(" ".join(cleaned_words) + "\n")
