import tensorflow as tf

def model():

    numFeature = 10
    numHidden1 = 10
    numHidden2 = 10
    numHidden3 = 10
    numLabel = 2
    RATE = 0.01


    # Set place
    feature = tf.placeholder(tf.float32, [None, numFeature])    # Get Sound data
    label = tf.placeholder(tf.float32, [None, numLabel])        # True answer


    # Model structure
    layer1 = tf.layers.dense(feature, units=numHidden1, activation=tf.nn.sigmoid)
    layer2 = tf.layers.dense(layer1, units=numHidden2, activation=tf.nn.sigmoid)
    layer3 = tf.layers.dense(layer2, units=numHidden3, activation=tf.nn.sigmoid)
    classes = tf.layers.dense(layer3, units=numLabel)

    loss = tf.losses.sigmoid_cross_entropy(label, classes)
    trainOptimize = tf.train.GradientDescentOptimizer(RATE).minimize(loss)


def initiation():


    return

def checkModel():
    return

"""
Things have to do
model definition
variable initialization
Train
Cjheck Accuracy
"""