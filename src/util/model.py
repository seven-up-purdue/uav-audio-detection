import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, name, path, n_classes=2):
        self.name = name
        self.model_path = path
        self.n_classes = n_classes
        self.n_dim = 13
        self.n_hidden_units_one = 280
        self.n_hidden_units_two = 300
        self.sd = 1 / np.sqrt(self.n_dim)
        #self.learning_rate = 0.01

        self.X = tf.placeholder(tf.float32, [None, self.n_dim])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])

        # declare the weights connecting the input to the hidden layer
        self.W_1 = tf.Variable(tf.random_normal([self.n_dim, self.n_hidden_units_one], mean=0, stddev=self.sd))
        self.b_1 = tf.Variable(tf.random_normal([self.n_hidden_units_one], mean=0, stddev=self.sd))
        # calculate the output of the hidden layer
        self.h_1 = tf.nn.tanh(tf.matmul(self.X, self.W_1) + self.b_1)

        # and the weights connecting the hidden layer to the output layer
        self.W_2 = tf.Variable(tf.random_normal([self.n_hidden_units_one, self.n_hidden_units_two], mean=0, stddev=self.sd))
        self.b_2 = tf.Variable(tf.random_normal([self.n_hidden_units_two], mean=0, stddev=self.sd))
        # calculate the output of the hidden layer
        self.h_2 = tf.nn.sigmoid(tf.matmul(self.h_1, self.W_2) + self.b_2)

        self.W = tf.Variable(tf.random_normal([self.n_hidden_units_two, self.n_classes], mean=0, stddev=self.sd))
        self.b = tf.Variable(tf.random_normal([self.n_classes], mean=0, stddev=self.sd))
        # output layer
        self.y_ = tf.nn.softmax(tf.matmul(self.h_2, self.W) + self.b)

    def initialize(self):
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

