# Import library
import IPython.display
import librosa.display
import numpy as np
import librosa
import tensorflow as tf
import glob
from myAudio import Audio

# Declare Variable
tf.reset_default_graph()
n_mfcc = 16
n_frame = 16
n_classes = 3
n_channels = 1
learning_rate = 0.0002

# Function Part
# Feature extraction Method: MFCC
def mfcc4(raw, label, chunk_size=8192, window_size=4096, sr=22050, n_mfcc=16, n_frame=16):
    mfcc = np.empty((0, n_mfcc, n_frame))
    y = []
    print(raw.shape)
    for i in range(0, len(raw), chunk_size//2):
        mfcc_slice = librosa.feature.mfcc(raw[i:i+chunk_size], sr=sr, n_mfcc=n_mfcc) #n_mfcc,17
        if mfcc_slice.shape[1] < 17:
            print("small end:", mfcc_slice.shape)
            continue
        mfcc_slice = mfcc_slice[:,:-1]
        mfcc_slice = mfcc_slice.reshape((1, mfcc_slice.shape[0], mfcc_slice.shape[1]))
        mfcc = np.vstack((mfcc, mfcc_slice))
        y.append(label)
    y = np.array(y)
    return mfcc, y


# Extraction function
def extraction(raw):
    soundData, _ = mfcc4(raw, 0)  # Get MFCC from raw data

    # Reshape dataset (?, 16,16) => (?, 256)
    dataX = np.reshape(soundData, (soundData.shape[0], -1))

    print("X: ", dataX.shape)
    print("Extract feature is finished")

    return dataX

''' Model part '''
X = tf.placeholder(tf.float32, shape=[None,n_mfcc*n_frame*n_channels])
Y = tf.placeholder(tf.float32, shape=[None,n_classes])

keep_prob = tf.placeholder(tf.float32)

dense1 = tf.layers.dense(inputs=X, units=256, activation=tf.nn.relu)
dropout1 = tf.nn.dropout(dense1, keep_prob=keep_prob)


dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
dropout2 = tf.nn.dropout(dense2, keep_prob=keep_prob)

dense3 = tf.layers.dense(inputs=dropout2, units=512, activation=tf.nn.relu)
dropout3 = tf.nn.dropout(dense3, keep_prob=keep_prob)

#이거 지워보고 돌려보고
dense4 = tf.layers.dense(inputs=dropout3, units=512, activation=tf.nn.relu)
dropout4 = tf.nn.dropout(dense4, keep_prob=keep_prob)


dense5 = tf.layers.dense(inputs=dropout4, units=256, activation=tf.nn.relu)
dropout5 = tf.nn.dropout(dense5, keep_prob=keep_prob)


logits= tf.layers.dense(inputs=dropout5, units=3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, './my_test_model_cnn')

def getDetectionResult():
    print("Start Process")
    raw = Audio.getStream(sample_rate = 22050, chunk_size = 8192, chunk_num = 1, isWrite=True)
    print("\nRaw data is created")
    dataX = extraction(raw)
    print("Feature is extracted")
    y_pred = sess.run(tf.argmax(logits,1),feed_dict={X: dataX, keep_prob: 1})
    print("Process is finished")
    return y_pred