from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import os


np.random.seed(42)
whitelist = 'abcdefghijklmnopqrstuvwxyz1234567890'

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 30
MAX_TARGET_SEQ_LENGTH = 30
MAX_VOCAB_SIZE = 600
DATA_DIR_PATH = 'data'
WEIGHT_FILE_PATH = 'models/word-weights.h5'
input_texts = []
target_texts = []

def in_white_list(_word):
    for char in _word:
        if char in whitelist:
            return True

    return False


def generate_batch(input_data, output_text_data, target_word2idx, num_decoder_tokens, BATCH_SIZE, encoder_max_seq_length, decoder_max_seq_length):
    num_batches = len(input_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = 0  # default [UNK]
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


def get_text_arrays(DATA_DIR_PATH, MAX_TARGET_SEQ_LENGTH):
    input_texts = []
    target_texts = []
    input_counter = Counter()
    target_counter = Counter()
    for file in os.listdir(DATA_DIR_PATH):
        filepath = os.path.join(DATA_DIR_PATH, file)
        if os.path.isfile(filepath):
            print('processing file: ', file)
            lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
            prev_words = []
            for line in lines:

                if line.startswith('- - '):
                    prev_words = []

                if line.startswith('- - ') or line.startswith('  - '):
                    line = line.replace('- - ', '')
                    line = line.replace('  - ', '')
                    next_words = [w.lower() for w in nltk.word_tokenize(line)]
                    next_words = [w for w in next_words if in_white_list(w)]
                    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

                    if len(prev_words) > 0:
                        input_texts.append(prev_words)
                        for w in prev_words:
                            input_counter[w] += 1

                        target_words = next_words[:]
                        target_words.insert(0, 'START')
                        target_words.append('END')
                        for w in target_words:
                            target_counter[w] += 1
                        target_texts.append(target_words)

                    prev_words = next_words
    #print(target_counter)
    return input_texts, target_texts, input_counter, target_counter


def get_token_arrays(input_counter, target_counter, MAX_VOCAB_SIZE):
    input_word2idx = dict()
    target_word2idx = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
        input_word2idx[word[0]] = idx + 2
    for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
        target_word2idx[word[0]] = idx + 1

    input_word2idx['PAD'] = 0
    input_word2idx['UNK'] = 1
    target_word2idx['UNK'] = 0

    input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
    target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

    np.save('models/word-input-word2idx.npy', input_word2idx)
    np.save('models/word-input-idx2word.npy', input_idx2word)
    np.save('models/word-target-word2idx.npy', target_word2idx)
    np.save('models/word-target-idx2word.npy', target_idx2word)

    return input_word2idx, input_idx2word, target_word2idx, target_idx2word


def get_max_lengths(input_idx2word, input_word2idx, target_idx2word, target_word2idx, input_texts, target_texts):
    num_encoder_tokens = len(input_idx2word)
    num_decoder_tokens = len(target_idx2word)
    encoder_input_data = []

    encoder_max_seq_length = 0
    decoder_max_seq_length = 0

    for input_words, target_words in zip(input_texts, target_texts):
        encoder_input_wids = []
        for w in input_words:
            w2idx = 1  # default [UNK]
            if w in input_word2idx:
                w2idx = input_word2idx[w]
            encoder_input_wids.append(w2idx)

        encoder_input_data.append(encoder_input_wids)
        encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
        decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

    context = dict()
    context['num_encoder_tokens'] = num_encoder_tokens
    context['num_decoder_tokens'] = num_decoder_tokens
    context['encoder_max_seq_length'] = encoder_max_seq_length
    context['decoder_max_seq_length'] = decoder_max_seq_length

    print(context)
    np.save('models/word-context.npy', context)
    return encoder_input_data, num_encoder_tokens, num_decoder_tokens, encoder_max_seq_length, decoder_max_seq_length


def get_model( num_encoder_tokens, num_decoder_tokens, encoder_max_seq_length, HIDDEN_UNITS):
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                                  input_length=encoder_max_seq_length, name='encoder_embedding')
    encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_states = [encoder_state_h, encoder_state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
    decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
    decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    json = model.to_json()
    open('models/word-architecture.json', 'w').write(json)
    return model

def main():

    #input, target and token frequency arrays
    input_texts, target_texts, input_counter, target_counter = get_text_arrays(DATA_DIR_PATH, MAX_TARGET_SEQ_LENGTH)

    # word to index and index to word
    input_word2idx, input_idx2word, target_word2idx, target_idx2word = get_token_arrays(input_counter, target_counter, MAX_VOCAB_SIZE)

    #Get encoder input and max limits
    encoder_input_data, num_encoder_tokens, num_decoder_tokens, encoder_max_seq_length, decoder_max_seq_length = get_max_lengths(input_idx2word, input_word2idx, target_idx2word, target_word2idx, input_texts, target_texts)

    #Get Model
    model = get_model(num_encoder_tokens, num_decoder_tokens, encoder_max_seq_length, HIDDEN_UNITS)

    #Trian model
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_input_data, target_texts, test_size=0.2, random_state=42)
    train_gen = generate_batch(Xtrain, Ytrain, target_word2idx, num_decoder_tokens, BATCH_SIZE, encoder_max_seq_length, decoder_max_seq_length)
    test_gen = generate_batch(Xtest, Ytest, target_word2idx, num_decoder_tokens, BATCH_SIZE, encoder_max_seq_length, decoder_max_seq_length)
    train_num_batches = len(Xtrain) // BATCH_SIZE
    test_num_batches = len(Xtest) // BATCH_SIZE

    checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
    model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                            epochs=NUM_EPOCHS,
                            verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])
    model.save_weights(WEIGHT_FILE_PATH)

if __name__== "__main__":
    main()
