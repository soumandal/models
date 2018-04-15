from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import pickle
import time


def evaluate(word):

    LATENT_DIMENSIONS = 128
    SAVE_NAME = 'PATH'

    with open(SAVE_NAME + "\\" + "num_encoder_tokens.pkl", "rb") as a:
        num_encoder_tokens = pickle.load(a)
    with open(SAVE_NAME + "\\" + "num_decoder_tokens.pkl", "rb") as b:
        num_decoder_tokens = pickle.load(b)
    with open(SAVE_NAME + "\\" + "max_encoder_seq_length.pkl", "rb") as c:
        max_encoder_seq_length = pickle.load(c)
    with open(SAVE_NAME + "\\" + "max_decoder_seq_length.pkl", "rb") as d:
        max_decoder_seq_length = pickle.load(d)
    with open(SAVE_NAME + "\\" + "source_token_index.pkl", "rb") as e:
        source_token_index = pickle.load(e)
    with open(SAVE_NAME + "\\" + "target_token_index.pkl", "rb") as f:
        target_token_index = pickle.load(f)
    model = load_model(SAVE_NAME + "\\" + 'model.h5')

    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(LATENT_DIMENSIONS,), name='input_3')
    decoder_state_input_c = Input(shape=(LATENT_DIMENSIONS,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    source_texts = [word]

    encoder_input_data = np.zeros((len(source_texts), max_encoder_seq_length,
                                   num_encoder_tokens), dtype='float32')

    for index, source_text in enumerate(source_texts):
        for t, char in enumerate(source_text):
            try:
                encoder_input_data[index, t, source_token_index[char]] = 1.
            except IndexError:
                pass

    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if (sampled_char == '\n' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]
        return decoded_sentence

    output = []
    for sequence_index in range(len(source_texts)):
        source_sequence = encoder_input_data[sequence_index:sequence_index + 1]
        decoded_sequence = decode_sequence(source_sequence)
        output.append(str(decoded_sequence))

    return output[0]