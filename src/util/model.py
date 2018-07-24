import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, sess, name, model_path):
        self.sess = sess
        self.name = name
        self.model_path = model_path
        self.build_net()

    def build_net(self):
        # with tf.variable_scope(self.name):
        # input place holders
        n_mfcc = 16#self.config.N_MFCC
        n_frame = 43#self.config.N_FRAME
        n_channels = 1#self.config.N_CHANNELS
        n_classes = 2#self.config.N_CLASSES
        X = tf.placeholder(tf.float32, shape=[None, n_mfcc * n_frame * n_channels])
        self.X = tf.reshape(X, [-1, n_mfcc, n_frame, n_channels])
        self.Y = tf.placeholder(tf.float32, shape=[None, n_classes])

        self.conv1 = tf.layers.conv2d(inputs=self.X, filters=1, kernel_size=[13, 4],
                                 activation=tf.nn.relu)

        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[3, 3],
                                        padding='SAME', strides=1)

        self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=1, kernel_size=[2, 2],
                                 padding="SAME", activation=tf.nn.relu)

        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2],
                                        padding="SAME", strides=2)

        self.flat = tf.reshape(self.pool2, [-1, 2*20*1])

        self.dense3 = tf.layers.dense(inputs=self.flat, units=200, activation=tf.nn.relu)
        self.logits = tf.layers.dense(inputs=self.dense3, units=2)
        self.pred = tf.nn.softmax(self.logits)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_path)

    def predict(self, X_test):
        y_pred = self.sess.run(self.pred, feed_dict={self.X: X_test})
        # y_pred = self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: X_test})
        return y_pred

    def print_variables(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)
