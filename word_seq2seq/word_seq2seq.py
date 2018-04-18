import tensorflow as tf
import numpy as np
import pickle
import time
import data_utils
import matplotlib.pyplot as plt


SAVE = "PATH"
TRAIN_SIZE = 200
LATENT_DIMENSIONS = 256
WORD_EMBEDDING_SIZE = 100
BATCH_SIZE = 64
LR = 5e-3
STEPS = 500

X, Y, en_word2idx, en_idx2word, en_vocab, bn_word2idx, bn_idx2word, bn_vocab = data_utils.read_dataset(SAVE + "\\" + 'data.pkl')

with open(SAVE + "\\" + "en_word2idx.pkl", "wb") as a:
    pickle.dump(en_word2idx, a)
with open(SAVE + "\\" + "bn_word2idx.pkl", "wb") as b:
    pickle.dump(bn_word2idx, b)
with open(SAVE + "\\" + "bn_idx2word.pkl", "wb") as c:
    pickle.dump(bn_idx2word, c)


def data_padding(x, y, length=15):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [en_word2idx['<pad>']]
        y[i] = [bn_word2idx['<go>']] + y[i] + [bn_word2idx['<eos>']] + (length-len(y[i])) * [bn_word2idx['<pad>']]

data_padding(X, Y)

X_train, Y_train = X[:TRAIN_SIZE], Y[:TRAIN_SIZE]

input_seq_len = max([len(vector) for vector in X])
with open(SAVE + "\\" + "input_seq_len.pkl", "wb") as d:
    pickle.dump(input_seq_len, d)
output_seq_len = max([len(vector) for vector in Y])
with open(SAVE + "\\" + "output_seq_len.pkl", "wb") as e:
    pickle.dump(output_seq_len, e)
en_vocab_size = len(en_vocab) + 2
with open(SAVE + "\\" + "en_vocab_size.pkl", "wb") as f:
    pickle.dump(en_vocab_size, f)
bn_vocab_size = len(bn_vocab) + 4
with open(SAVE + "\\" + "bn_vocab_size.pkl", "wb") as g:
    pickle.dump(bn_vocab_size, g)

encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='encoder{}'.format(i)) for i in range(input_seq_len)]
decoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name='decoder{}'.format(i)) for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]

targets.append(tf.placeholder(dtype=tf.int32, shape=[None], name='last_target'))
target_weights = [tf.placeholder(dtype=tf.float32, shape=[None], name='target_w{}'.format(i)) for i in range(output_seq_len)]

LATENT_DIMENSIONS
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
                                            feed_previous=False,
                                            output_projection=output_projection,
                                            dtype=tf.float32)


def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
                        weights=w_t,
                        biases=b,
                        labels=tf.reshape(labels, [-1, 1]),
                        inputs=logits,
                        num_sampled=LATENT_DIMENSIONS,
                        num_classes=bn_vocab_size)

loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function=sampled_loss)


def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()


def feed_dict(x, y, BATCH_SIZE):

    feed = {}

    idxes = np.random.choice(len(x), size=BATCH_SIZE, replace=False)

    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes], dtype = np.int32)

    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes], dtype=np.int32)

    feed[targets[len(targets)-1].name] = np.full(shape=[BATCH_SIZE], fill_value=bn_word2idx['<pad>'], dtype=np.int32)

    for i in range(output_seq_len-1):
        batch_weights = np.ones(BATCH_SIZE, dtype=np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(BATCH_SIZE):
            if target[j] == bn_word2idx['<pad>']:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights

    feed[target_weights[output_seq_len-1].name] = np.zeros(BATCH_SIZE, dtype=np.float32)

    return feed

outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

optimizer = tf.train.RMSPropOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()


def forward_step(sess, feed):
    output_sequences = sess.run(outputs_proj, feed_dict=feed)
    return output_sequences


def backward_step(sess, feed):
    sess.run(optimizer, feed_dict=feed)

losses = []

saver = tf.train.Saver()

print('TRAINING STARTED')

with tf.Session() as sess:
    sess.run(init)

    t = time.time()
    for step in range(STEPS):
        feed = feed_dict(X_train, Y_train, BATCH_SIZE)

        backward_step(sess, feed)

        if step % 5 == 4 or step == 0:
            loss_value = sess.run(loss, feed_dict=feed)
            print('STEP: {}, LOSS: {}'.format(step, loss_value))
            losses.append(loss_value)

        if step % 20 == 19:
            saver.save(sess, SAVE + "\\" + "checkpoints\\", global_step=step)
            print('CHECKPOINT IS SAVED')

    print('TRAINING TIME FOR {} STEPS: {}s'.format(STEPS, time.time() - t))

    # SAVING LOSS FOR LATER STUDY
    with open("losses.pkl", "wb") as g:
        pickle.dump(losses, g)

plt.plot(losses)
plt.xlabel('steps')
plt.ylabel('loss')
plt.show()
