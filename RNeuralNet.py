import helper
import numpy as np
from collections import Counter
from tensorflow.contrib import seq2seq
import tensorflow as tf

def create_lookup_tables(text):
    text_count = Counter(text)
    sorted_text = sorted(text_count, key=text_count.get, reverse=True)
    int_to_vocab = {ii:word for ii, word in enumerate(sorted_text)}
    vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

def token_lookup():
    token = {'.': '|period|',
             ',': '|comma|',
             '"': '|quotation_mark|',
             ';': '|semicolon|',
             '!': '|exclamation_mark|',
             '?': '|question_mark|',
             '(': '|left_parentheses|',
             ')': '|right_parentheses|',
             '--': '|dash|',
             '\n': '|return|'}
    return token

def get_inputs():
    input = tf.placeholder(tf.int32, shape=(None, None), name='input')
    targets = tf.placeholder(tf.int32, shape=(None, None))
    learning_rate = tf.placeholder(tf.float32)
    return input, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, 'initial_state')
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def build_rnn(cell, inputs):
    rnn, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return rnn, final_state

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed = get_embed(input_data, vocab_size, embed_dim)
    rnn, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(rnn, vocab_size, activation_fn=None)
    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    batch_size_max = batch_size * seq_length
    batch_count = len(int_text) // batch_size_max
    column_size = batch_count * seq_length
    batches = []
    max_ind = batch_count * batch_size_max
    for fbi in range(batch_count):
        batch = []
        input_data = []
        output_data = []
        full_batch_off = fbi * seq_length
        for bi in range(batch_size):
            batch_off = bi * column_size + full_batch_off
            input_data.append(np.array([int_text[batch_off + i] for i in range(seq_length)]))
            output_data.append(np.array([int_text[(batch_off + i + 1) % max_ind] for i in range(seq_length)]))

        batch.append(np.array(input_data))
        batch.append(np.array(output_data))
        batches.append(np.array(batch))

    return np.array(batches)




def Train(embed_dim= 512, num_epochs=20, learning_rate= 0.01, seq_length= 10, rnn_size= 700, batch_size = 100):
    data_dir = './data/simpsons/moes_tavern_lines.txt'
    text = helper.load_data(data_dir)
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]
    helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
    int_text, _, int_to_vocab, _ = helper.load_preprocess()
    show_every_n_batches = 50
    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

        # Probabilities for generating words
        tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, "./save")
        print('Model Trained and Saved')

    # Save parameters for checkpoint
    helper.save_params((seq_length, "./save"))

if __name__ == "__main__":
    Train()