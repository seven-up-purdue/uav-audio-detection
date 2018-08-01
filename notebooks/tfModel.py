import numpy as np
import librosa
import tensorflow as tf
import glob

from myAudio import Audio


CHUNK_SIZE = 8192
SR = 22050

batch_size = 2048
num_classes = 20            #분류할 사전의 크기 

learning_rate = 0.01
sequence_length = 16 #9         

output_dim = 3
layers = 3 

def mfcc(raw, chunk_size=8192, sr=22050, n_mfcc=num_classes):
    mfcc = np.empty((num_classes, 0))
    for i in range(0, len(raw), chunk_size):
        mfcc_slice = librosa.feature.mfcc(raw[i:i+chunk_size], sr=sr, n_mfcc=n_mfcc)
        mfcc = np.hstack((mfcc, mfcc_slice))
    return mfcc
def makeHot(dataX, sequence_length):
    X_hot_list= []
    #Y_hot_tmp = dataY[sequence_length-1:]

    for i in range(0, dataX.shape[0] - sequence_length+1):
        _x = dataX[i:i + sequence_length]
        #if i<10:
            #print(_x, "->", Y_hot_tmp[i])
        X_hot_list.append(_x)

    X_hot = np.array(X_hot_list[:])
    #Y_hot= Y_hot_tmp.reshape((len(Y_hot_tmp),n_unique_labels))
    return X_hot[:]#, Y_hot[:]
def extractFeature(raw):
    dataX = mfcc(raw).T
    X_hot = makeHot(dataX,sequence_length = sequence_length)
    return X_hot[:]

class Data:
    def __init__(self,X,Y,BatchSize):
        self.X = X
        self.Y = Y
        self.len = len(Y)
        self.bs = BatchSize
        self.bs_i = 0
    def getBatchData(self):
        s = self.bs_i
        e = self.bs_i + self.bs
        if e> self.len:
            e -= self.len
            result =  np.vstack((self.X[s:],self.X[:e])), np.vstack((self.Y[s:],self.Y[:e]))
        else:
            result =  self.X[s:e], self.Y[s:e]
            
        self.bs_i = e
        return result
###########################################   Model   #########################################
X = tf.placeholder(tf.float32, [None, sequence_length,num_classes], name="X")
Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")

cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_classes, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*layers, state_is_tuple= True)

BatchSize = tf.placeholder(tf.int32, [], name='BatchSize')
initial_state = cell.zero_state(BatchSize, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X,initial_state=initial_state,dtype=tf.float32)

dense1 = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
dense2 = tf.layers.dense(inputs=dense1, units=num_classes, activation=tf.nn.relu)

Y_pred= tf.layers.dense(inputs=dense2, units=output_dim)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
lr = tf.placeholder(tf.float32,shape=(), name='learning_rate')
train = tf.train.AdamOptimizer(lr).minimize(cost)
##################################################################################################
sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, '../models/RNN/my_RNN_model')
