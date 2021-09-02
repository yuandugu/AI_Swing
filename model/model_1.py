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

class GCN_2_Net(Model):
    """
    Graph Level Prediction

    graphconv(16,l2,relu)+graphconv(16,l2,relu)+graphconv(8,l2,relu)+flatten+fc(64)+fc(32)+sigmoid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1   = GraphConv(16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')
        self.conv2   = GraphConv(16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')
        self.conv3   = GraphConv(8, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')
        self.flatten = Flatten()
        self.fc1     = Dense(64, activation='relu')
        self.fc2     = Dense(32, activation='relu')
        self.fc3     = Dense(n_out, activation='sigmoid')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

class GCN_LSTM_Net(Model):
    """
    Graph Level Prediction

    graphconv(16,l2,relu)+graphconv(16,l2,relu)+graphconv(8,l2,relu)+flatten+fc(64)+LSTM(64)+fc(32)+sigmoid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1   = GraphConv(16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')
        self.conv2   = GraphConv(16, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')
        self.conv3   = GraphConv(8, kernel_regularizer=l2(l=l2_reg_gcn), activation='relu')

        # self.GAP     = GlobalAvgPool
        self.flatten = Flatten()
        self.fc1     = Dense(64, activation='relu')
        self.reshape = Reshape(target_shape=(64,1))
        self.lstm    = LSTM(units=64)
        self.fc2     = Dense(32, activation='relu')
        self.fc3     = Dense(n_out, activation='sigmoid')

    def call(self, inputs):
        x, fltr = inputs
        x = self.conv1([x, fltr])
        x = self.conv2([x, fltr])
        x = self.conv3([x, fltr])
        # output = self.GAP(x)
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.reshape(output)
        output = self.lstm(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

dropout       = 0.6                     # Dropout rate for the features and adjacency matrix
channels      = 8                       # Number of channel in each head of the first GAT layer
n_attn_heads  = 8                       # Number of attention heads in first GAT layer
n_classes     = 3
l2_reg_gat    = 5e-4/2

class GAT_Net(Model):
    """
    Graph Level Prediction
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drop_1  = Dropout(dropout)
        self.att_1   = GraphAttention(channels,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.drop_2  = Dropout(dropout)
        self.att_2   = GraphAttention(channels,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.flatten = Flatten()
        self.fc1     = Dense(1000, activation='relu')
        self.fc2     = Dense(n_out, activation='sigmoid')
    
    def call(self, inputs):
        x, fltr = inputs
        x = self.drop_1(x)
        x = self.att_1([x, fltr])
        x = self.drop_2(x)
        x = self.att_2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

class GAT_2_Net(Model):
    """
    Node Level Prediction
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drop_1  = Dropout(dropout)
        self.att_1   = GraphAttention(channels,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.drop_2  = Dropout(dropout)
        self.att_2   = GraphAttention(n_classes,
                                      attn_heads=1,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='softmax',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
    
    def call(self, inputs):
        x, fltr = inputs
        x = self.drop_1(x)
        x = self.att_1([x, fltr])
        x = self.drop_2(x)
        output = self.att_2([x, fltr])

        return output

class GAT_3_Net(Model):
    """
    Graph Level Prediction
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drop_1  = Dropout(dropout)
        self.att_1   = GraphAttention(16,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.drop_2  = Dropout(dropout)
        self.att_2   = GraphAttention(16,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.att_3   = GraphAttention(8,
                                      attn_heads=n_attn_heads,
                                      concat_heads=True,
                                      dropout_rate=dropout,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg_gat),
                                      attn_kernel_regularizer=l2(l2_reg_gat))
        self.flatten = Flatten()
        self.fc1     = Dense(64, activation='relu')
        self.fc2     = Dense(32, activation='relu')
        self.fc3     = Dense(n_out, activation='sigmoid')
    
    def call(self, inputs):
        x, fltr = inputs
        # x = self.drop_1(x)
        x = self.att_1([x, fltr])
        # x = self.drop_2(x)
        x = self.att_2([x, fltr])
        output = self.flatten(x)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

if __name__ == '__main__':
    # test
    net = 'GAT'
    if net == 'GCN':
        learning_rate = 1e-3                    # Learning rate for Adam
    elif net == 'GAT':
        learning_rate = 5e-3
    model = GAT_Net()
    model.build(input_shape=[(14, 20), (14, 14)])
    optimizer = Adam(lr=learning_rate)
    loss_fn = BinaryCrossentropy()
    acc_fn = BinaryAccuracy()
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='model.png')
