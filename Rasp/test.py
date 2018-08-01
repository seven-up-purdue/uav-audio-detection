from tfModel import *
while(True):
    raw = Audio.getStream(sample_rate= 22050, chunk_size = 8192, chunk_num = 1, isWrite= True)
    dataX = extractFeature(raw)
    y_pred = sess.run(tf.argmax(Y_pred, 1), feed_dict={X:dataX, BatchSize: len(dataX)})
    
    print('\t', y_pred)