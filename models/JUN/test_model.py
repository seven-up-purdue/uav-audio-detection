import tensorflow as tf
import numpy as np
from dataProcess import Sound
# import matplotlib.pyplot as plt

# Variable
training = 2205
numLabel = 1
RATE = 0.1
trainRate = 0.2
epoch = 4000
batchSize = 150

# Data line
load = Sound("../../soundData/0720A/load/", "wav")
load.load(1)
load.dataCutting()
loadData, loadLabel = load.Process(training)
unload = Sound("../../soundData/0720A/unload/", "wav")
unload.load(0)
unload.dataCutting()
unloadData, unloadLabel = unload.Process(training)

# Divide train set & test set - load UAV data
trainDataLen = round(len(loadData)*0.8)
testDataLen = len(loadData)-trainDataLen
loadTrain = loadData[:trainDataLen]
print("Training set number: ", trainDataLen)
loadTest = loadData[-testDataLen:]
print("Testing set number: ", testDataLen)
loadTrainLabel = loadLabel[:trainDataLen]
loadTestLabel = loadLabel[-testDataLen:]

# Divide train set & test set - unload UAV data
trainDataLen = round(len(unloadData)*0.8)
testDataLen = len(unloadData)-trainDataLen
unloadTrain = unloadData[:trainDataLen]
print("Training set number: ", trainDataLen)
unloadTest = unloadData[-testDataLen:]
print("Testing set number: ", testDataLen)
unloadTrainLabel = unloadLabel[:trainDataLen]
unloadTestLabel = unloadLabel[-testDataLen:]


# Combine two data
xValue = loadTrain + unloadTrain
xTest = loadTest + unloadTest
yValue = loadTrainLabel + unloadTrainLabel
yTest = loadTestLabel + unloadTestLabel
print("Train data length: ", len(xValue))
print("Test data length: ", len(xTest))
print("Train Label length: ", len(yValue))
print("Test Label length: ", len(yTest))

"""
Model-SVM
"""
# Model
sess = tf.Session()

# Data
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

    _, trainLoss, summary = sess.run([trainStep, loss, summaryOpt], feed_dict={xTrain: randX, yTarget: np.transpose(randY)})

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
# Show
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = - a2 / a1
yIntercept = b / a1
value = None
bestFit = []
for i in value:
    bestFit.append(slope*i + yIntercept)
    loadUavX = None
    loadUavY = None
    unloadUavX = None
    unloadUavY = None

# Graph draw
# Input data and linear separator
plt.plot(loadUavX, loadUavY, "O", label="Load UAV")
plt.plot(unloadUavX, unloadUavY, "X", label="unLoad UAV")
plt.plot(value, bestFit, "r-", label="Linear Separator", linewidth=3)
plt.title("Load UAV vs unLoad UAV")
# plt.ylim([0,10])
# plt.xlabel()
# plt.ylabel()
plt.legend(loc="lower right")
plt.show()

# Accuracy Graph
plt.plot(trainAccuracy, "k-", label="Train Accuracy")
plt.plot(testAccuracy, "r--", label="Test Accuracy")
plt.title("Train and Test sets Accuracies")
# plt.xlabel()
# plt.ylabel()
plt.legend(loc="lower right")
plt.show()



"""
sess.close()
"""
To do
model definition
variable initialization
Train
Check Accuracy
"""