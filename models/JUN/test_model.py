import tensorflow as tf
import numpy as np
from dataProcess import Sound


# Data line
load = Sound("../../soundData/0720A/load/", "wav")
load.load()
load.dataCutting()
cleanData = load.preProcess()

# Divide train set & test set
tmp = round(len(cleanData)*0.8)
xValue = cleanData[:tmp]
print("Training set number: ", tmp)
xTest = cleanData[(tmp - len(cleanData)):]
print("Testing set number: ", len(cleanData)-tmp)
yValue = [1] * tmp
yTest = [1] * (len(cleanData)-tmp)



"""
Model-SVM
"""
# Variable
training = 4410
numLabel = 1
RATE = 0.1
trainRate = 0.2
epoch = 4000

# Model
sess = tf.Session()

# Data
batchSize = 100
xTrain = tf.placeholder(shape=[None, training], dtype=tf.float32)
yTarget = tf.placeholder(shape=[None], dtype=tf.float32)

# Slope rate: A, intercept: b
A = tf.Variable(tf.random_normal([training, numLabel]))
b = tf.Variable(tf.random_normal([numLabel, numLabel]))

# Output
outPut = tf.subtract(tf.matmul(xTrain, A), b)

# Maximize margin loss
l2Norm = tf.reduce_sum(tf.square(A))
alpha = tf.constant([RATE])
classificationTerm = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(outPut, yTarget))))
loss = tf.add(tf.multiply(alpha, l2Norm), classificationTerm)

# Prediction & Accuracy
prediction = tf.sign(outPut)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, yTarget), tf.float32))
residuals = prediction - yTarget

# Creating scalrs and historams in the tensorboard
with tf.name_scope("Loss"):
    tf.summary.histogram("Histogram Error: ", accuracy)
    tf.summary.histogram("Histogram Residuals: ", residuals)
    lossSummaryOpt = tf.summary.scalar("Loss: ", loss[0])

summaryOpt = tf.summary.merge_all()

# Declare optimizer and initialize model variable
optimize = tf.train.GradientDescentOptimizer(trainRate)
trainStep = optimize.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

# Running Session
lossVec = []
trainAccuracy = []
testAccuracy = []


for i in range(epoch):
    randIndex = np.random.choice(len(xValue), size=batchSize)
    randX = []
    randY = []
    for j in randIndex:
        randX.append(xValue[j])
        randY.append(yValue[j])
    randY = np.transpose(randY)


    _, trainLoss, summary = sess.run([trainStep, loss, summaryOpt], feed_dict = {xTrain: randX, yTarget: randY})

    testLoss, testResids = sess.run([loss, residuals], feed_dict={xTrain: xTest, yTarget: np.transpose(yTest)})

    lossVec.append(trainLoss)
    trainAccuTmp = sess.run(accuracy, feed_dict={xTrain: xValue, yTarget: np.transpose(yValue)})
    trainAccuracy.append(trainAccuTmp)
    testAccuTmp = sess.run(accuracy, feed_dict={xTrain: xTest, yTarget: np.transpose(yTest)})
    testAccuracy.append(testAccuTmp)

    if (i+1) % 250 == 0:
        print("Step #", i+1, "A = ", sess.run(A), "b = ", sess.run(b))
        print("Loss = ", trainLoss)
        print("Train Accuracy = ", np.mean(trainAccuracy))
        print("Test Accuracy = ", np.mean(testAccuracy))
        print()



"""
To do
model definition
variable initialization
Train
Check Accuracy
"""