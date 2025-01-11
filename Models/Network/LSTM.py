#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

class xLSTMCell(tf.keras.layers.AbstractRNNCell):
    """Custom xLSTM cell implementation"""
    def __init__(self, units, **kwargs):
        super(xLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [tf.TensorShape([units]), tf.TensorShape([units])]
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Input gates
        self.Wi = self.add_weight('Wi', shape=[input_dim, self.units])
        self.Ui = self.add_weight('Ui', shape=[self.units, self.units])
        self.bi = self.add_weight('bi', shape=[self.units])
        
        # Forget gates
        self.Wf = self.add_weight('Wf', shape=[input_dim, self.units])
        self.Uf = self.add_weight('Uf', shape=[self.units, self.units])
        self.bf = self.add_weight('bf', shape=[self.units])
        
        # Output gates
        self.Wo = self.add_weight('Wo', shape=[input_dim, self.units])
        self.Uo = self.add_weight('Uo', shape=[self.units, self.units])
        self.bo = self.add_weight('bo', shape=[self.units])
        
        # Cell state
        self.Wc = self.add_weight('Wc', shape=[input_dim, self.units])
        self.Uc = self.add_weight('Uc', shape=[self.units, self.units])
        self.bc = self.add_weight('bc', shape=[self.units])
        
        # Extra xLSTM gates
        self.Wx = self.add_weight('Wx', shape=[input_dim, self.units])
        self.Ux = self.add_weight('Ux', shape=[self.units, self.units])
        self.bx = self.add_weight('bx', shape=[self.units])
        
        self.built = True
        
    def call(self, inputs, states):
        h_prev, c_prev = states
        
        # Input gate
        i = tf.sigmoid(
            tf.matmul(inputs, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        
        # Forget gate
        f = tf.sigmoid(
            tf.matmul(inputs, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        
        # Output gate
        o = tf.sigmoid(
            tf.matmul(inputs, self.Wo) + tf.matmul(h_prev, self.Uo) + self.bo)
        
        # Extra xLSTM gate
        x = tf.sigmoid(
            tf.matmul(inputs, self.Wx) + tf.matmul(h_prev, self.Ux) + self.bx)
        
        # Cell state candidate
        c_candidate = tf.tanh(
            tf.matmul(inputs, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)
        
        # Updated cell state with xLSTM modification
        c = f * c_prev + i * c_candidate + x * tf.tanh(c_prev)
        
        # Updated hidden state
        h = o * tf.tanh(c)
        
        return h, [h, c]

def xLSTM(Input, max_time, n_input, lstm_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''
    Modified LSTM implementation using xLSTM architecture for EEG signal processing
    
    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of xLSTM Model
        n_input: The input signal size at one time
        lstm_size: The number of xLSTM units
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer
    Returns:
        FC_2: Final prediction of xLSTM Model
        FC_1: Extracted features from the first fully connected layer
    '''
    # Reshape input
    Input = tf.reshape(Input, [-1, max_time, n_input])
    
    # Create xLSTM cell
    cell_encoder = xLSTMCell(lstm_size)
    encoder_drop = tf.keras.layers.Dropout(rate=1-keep_prob)(cell_encoder)
    
    # Create RNN layer
    rnn_layer = tf.keras.layers.RNN(encoder_drop, 
                                   return_state=True,
                                   return_sequences=False)
    
    # Get outputs and states
    outputs, hidden_state, cell_state = rnn_layer(Input)
    
    # First fully-connected layer with batch normalization
    FC_1 = tf.matmul(hidden_state, weights_1) + biases_1
    FC_1 = tf.keras.layers.BatchNormalization()(FC_1)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)
    
    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)
    
    return FC_2, FC_1
