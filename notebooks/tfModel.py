import numpy as np
import librosa
import tensorflow as tf
import glob

from myAudio import Audio


CHUNK_SIZE = 8192
SR = 22050
N_MFCC = 16

#mfcc = np.empty((0, N_MFCC, 16))
def mfcc4(raw, chunk_size=8192, window_size=4096, sr=22050, n_mfcc=16, n_frame=16):
    mfcc = np.empty((0, n_mfcc, n_frame))
    print(raw.shape)
    for i in range(0, len(raw), chunk_size//2):
        mfcc_slice = librosa.feature.mfcc(raw[i:i+chunk_size], sr=sr, n_mfcc=n_mfcc) #n_mfcc,17
        if mfcc_slice.shape[1] < 17:
            print("small end:", mfcc_slice.shape)
            continue
        mfcc_slice = mfcc_slice[:,:-1]
        mfcc_slice = mfcc_slice.reshape((1, mfcc_slice.shape[0], mfcc_slice.shape[1]))
        mfcc = np.vstack((mfcc, mfcc_slice))
    return mfcc
###########################################   Model   #########################################
n_mfcc = 16
n_frame = 16
n_classes = 3
n_channels = 1

kernel_size = 3
stride = 1
pad = "SAME"

learning_rate = 0.005
training_epochs = 20

X = tf.placeholder(tf.float32, shape=[None,n_mfcc*n_frame*n_channels])
X = tf.reshape(X, [-1, n_mfcc, n_frame, n_channels])
Y = tf.placeholder(tf.float32, shape=[None,n_classes])

conv1 = tf.layers.conv2d(inputs=X, filters=1, kernel_size=[3, 3],
                         padding="SAME", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                padding="SAME", strides=1)
#dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=True)

flat = tf.reshape(pool1, [-1, 16*16*1])

dense2 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
#dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=True)
logits = tf.layers.dense(inputs=dense2, units=3)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
######################################################################################################
sess = tf.Session()
saver = tf.train.Saver()
# 재성이 형 여기 모델 불러오는 path
saver.restore(sess, '../models/CNN/my_test_model_cnn')
