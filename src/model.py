import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

class LSTMHedger(tf.keras.Model):
    def __init__(self, units=32):
        super(LSTMHedger, self).__init__()
        self.units = units
        self.lstm_cell = LSTMCell(units)
        self.output_layer = Dense(1, activation='tanh', name='delta_t_output')

    def call(self, inputs, states):
        # inputs shape: (batch, 3) -> [S_t_norm, BS_delta_t, delta_prev]
        lstm_out, new_states = self.lstm_cell(inputs, states)
        predicted_delta = self.output_layer(lstm_out)
        return predicted_delta, new_states

    def get_initial_states(self, batch_size):
        return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]