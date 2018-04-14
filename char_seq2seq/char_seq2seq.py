from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pickle


def char_seq2seq():

    SAVE_NAME = "PATH"
    FILENAME = "SAMPLE.txt"
    OPTIMIZER = "rmsprop"
    LOSS = "categorical_crossentropy"
    ACTIVATION = "softmax"
    BATCH_SIZE = 32
    EPOCHS = 100
    LATENT_DIMENSIONS = 128
    TRAIN_SIZE = 1000

    source_texts, target_texts = [], []
    source_characters, target_characters = set(), set()

    text_lines = open(FILENAME, encoding="utf-8").read().split('\n')

    for line in text_lines[: min(TRAIN_SIZE, len(text_lines) - 1)]:
        source_text, target_text = line.split("\t")
        target_text = '\t' + target_text + '\n'
        source_texts.append(source_text)
        target_texts.append(target_text)
        for character in source_text:
            if character not in source_characters:
                source_characters.add(character)
        for character in target_text:
            if character not in target_characters:
                target_characters.add(character)

    source_characters = sorted(list(source_characters))
    target_characters = sorted(list(target_characters))

    num_encoder_tokens = len(source_characters)
    with open(SAVE_NAME + "\\" + "num_encoder_tokens.pkl", "wb") as b:
        pickle.dump(num_encoder_tokens, b)
    num_decoder_tokens = len(target_characters)
    with open(SAVE_NAME + "\\" + "num_decoder_tokens.pkl", "wb") as a:
        pickle.dump(num_decoder_tokens, a)

    max_encoder_seq_length = max([len(txt) for txt in source_texts])
    with open(SAVE_NAME + "\\" + "max_encoder_seq_length.pkl", "wb") as b:
        pickle.dump(max_encoder_seq_length, b)
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    with open(SAVE_NAME + "\\" + "max_decoder_seq_length.pkl", "wb") as b:
        pickle.dump(max_decoder_seq_length, b)

    MODEL_SUMMARY = open(SAVE_NAME + "\\" + "MODEL_SUMMARY.txt", "w", encoding="utf-8")

    MODEL_SUMMARY.write('NUMBER OF SAMPLES : ' + str(len(source_texts)) + "\n")
    MODEL_SUMMARY.write('NO OF UNIQUE INPUT TOKENS : ' + str(num_encoder_tokens) + "\n")
    MODEL_SUMMARY.write('NO OF UNIQUE OUTPUT TOKENS : ' + str(num_decoder_tokens) + "\n")
    MODEL_SUMMARY.write('MAX SEQUENCE LENGTH FOR INPUTS : ' + str(max_encoder_seq_length) + "\n")
    MODEL_SUMMARY.write('MAX SEQUENCE LENGTH FOR OUTPUTS : ' + str(max_decoder_seq_length) + "\n")
    MODEL_SUMMARY.write('LATENT DIMENSIONS : ' + str(LATENT_DIMENSIONS) + "\n")
    MODEL_SUMMARY.write('BATCH SIZE : ' + str(BATCH_SIZE) + "\n")
    MODEL_SUMMARY.write('ACTIVATION : ' + str(ACTIVATION) + "\n")
    MODEL_SUMMARY.write('LOSS : ' + str(LOSS) + "\n")

    source_token_index = dict([(char, i) for i, char in enumerate(source_characters)])
    with open(SAVE_NAME + "\\" + "source_token_index.pkl", "wb") as c:
        pickle.dump(source_token_index, c)
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    with open(SAVE_NAME + "\\" + "target_token_index.pkl", "wb") as d:
        pickle.dump(target_token_index, d)

    encoder_input_data = np.zeros(
        (len(source_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(source_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(source_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for index, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        for t, char in enumerate(source_text):
            encoder_input_data[index, t, source_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[index, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[index, t - 1, target_token_index[char]] = 1.

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(LATENT_DIMENSIONS, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    rmsprop = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(LATENT_DIMENSIONS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation=ACTIVATION)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=rmsprop, loss=LOSS, metrics=["accuracy"])

    model.load_weights("model_transfer_weights.h5")

    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)

    model.save(SAVE_NAME + "\\" + "model.h5")
    model.summary()

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(LATENT_DIMENSIONS,))
    decoder_state_input_c = Input(shape=(LATENT_DIMENSIONS,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

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

    acc = history.history['acc']
    loss = history.history['loss']

    with open(SAVE_NAME + "\\" + "acc.pkl", "wb") as e:
        pickle.dump(acc, e)
    with open(SAVE_NAME + "\\" + "loss.pkl", "wb") as f:
        pickle.dump(loss, f)

    print("TRAINING COMPLETED")

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(LATENT_DIMENSIONS,))
    decoder_state_input_c = Input(shape=(LATENT_DIMENSIONS,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    reverse_input_char_index = dict(
        (i, char) for char, i in source_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    def decode_sequence(source_sequence):

        # ENCODE INPUT AS STATE VECTORS
        states_value = encoder_model.predict(source_sequence)

        # GENERATE EMPTY TARGET SEQUENCE OF LENGTH 1
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # SET FIRST CHARACTER OF TARGET SEQUENCE AS START
        target_seq[0, 0, target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # EXIT CONDITION : EITHER HIT MAX LENGTH OR FIND STOPPING SYMBOL
            if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # UPDATE TARGET SEQUENCE
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # UPDATE STATES
            states_value = [h, c]

        return decoded_sentence

    for seq_index in range(10):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('input sequence :', source_texts[seq_index])
        print('decoded sequence :', decoded_sentence)