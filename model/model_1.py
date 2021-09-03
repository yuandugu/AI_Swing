import tensorflow as tf

from tensorflow.keras import Model, activations
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, Activation, LSTM, Reshape, Concatenate
from tensorflow.keras.layers import TimeDistributed, Add, AveragePooling1D
from tensorflow.keras.regularizers import l2
from tcn import TCN

from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Accuracy, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from spektral.layers import GraphConv, GraphAttention, GlobalAvgPool
from tensorflow.python.util.nest import flatten_dict_items

l2_reg_gcn        = 5e-4
n_out             = 1
learning_rate_gcn = 1e-3
N = 39

class GCN_Net(Model):
    """
    Graph Level Prediction

    graphconv(32,l2)+graphconv(32,l2)+flatten+fc(1000)+sigmoid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1   = GraphConv(32, kernel_regularizer=l2(l=l2_reg_gcn))
        self.conv2   = GraphConv(32, kernel_regularizer=l2(l=l2_reg_gcn))
        self.flatten = Flatten()
        self.fc1     = Dense(1000, activation='relu')
        self.fc2     = Dense(n_out, activation='sigmoid')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

if __name__ == '__main__':
    # test
    net = 'GAT'
    if net == 'GCN':
        learning_rate = 1e-3                    # Learning rate for Adam
    elif net == 'GAT':
        learning_rate = 5e-3
    model = GCN_Net()
    model.build(input_shape=[(14, 20), (14, 14)])
    optimizer = Adam(lr=learning_rate)
    loss_fn = BinaryCrossentropy()
    acc_fn = BinaryAccuracy()
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='model.png')
