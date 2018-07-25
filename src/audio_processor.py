import pyaudio
import wave
import numpy as np
import time
import multiprocessing as mp
from websocket import create_connection
import ctypes
import json
import tensorflow as tf
import librosa
from util.config import Config
from util.model import CNN

##### Fixed configuration #####
CHUNK_SIZE = 11025#8192
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
N_MFCC = 16
N_FRAME = 43

##### User configuration #####
config = Config('../config.yaml')
print(config.__dict__)

BUFFER_HOURS = config.BUFFER_HOURS
WAVE_FILENAME = config.WAV_PATH
print(WAVE_FILENAME)
DEVICE_NAME = config.DEVICE_NAME


tf.reset_default_graph()

##### Server connection #####
SERVER_ADDRESS = config.SERVER_ADDRESS
ws = create_connection("ws://%s/ws?device=rpi"%SERVER_ADDRESS)



def process_audio(shared_mfcc, shared_time, shared_pos, lock):
    """
    Grab some audio from the mic, save it to the file and calculate mfcc 
    :param shared_mfcc:
    :param shared_time:
    :param shared_pos:
    :param lock:
    :return:
    """

    # open default audio input stream
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=CHUNK_SIZE)
    wf = wave.open(WAVE_FILENAME, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
    wf.setframerate(SAMPLE_RATE)

    devinfo = p.get_device_info_by_index(1)
    print(devinfo)

    prev_audio = np.fromstring(stream.read(CHUNK_SIZE), np.int16)

    # j = 1
    try:
        while True:
            t1 = time.time()
            # grab audio chunk
            audio = np.fromstring(stream.read(CHUNK_SIZE), np.int16) # window: 44100//4
            # append current audio chunk to the previous audio chunk
            window = np.hstack((prev_audio, audio))
            # point prev_audio to current audio
            prev_audio = audio
            # calculate mfcc from the chunk
            mfcc = librosa.feature.mfcc(window.astype(float), sr=SAMPLE_RATE, n_mfcc=N_MFCC)[:,:-1] # 16,43
            # print(mfcc.shape)

            # acquire lock
            lock.acquire()

            # increment index counter and make sure not longer than bufferlen
            shared_pos.value = (shared_pos.value+1) % len(shared_time)
            print("shared_pos.value: ", shared_pos.value)
            # record current time
            shared_time[shared_pos.value] = t1
            # save mfcc values flat
            shared_mfcc[shared_pos.value,:] = mfcc.flatten()

            # release lock
            lock.release()

            # write to the audio file
            wf.writeframes(b''.join(audio))
            # j+=1

            t2 = time.time()
            print("******time for mfcc: ",t2-t1)

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()

    # after exiting the loop
    stream.stop_stream()
    stream.close()
    p.terminate()


def process_requests(shared_mfcc, shared_time, shared_pos, lock):
    """
    Run the pretrained classification model and send the results to the server
    :param shared_mfcc:
    :param shared_time:
    :param shared_pos:
    :param lock:
    :return:
    """
    # i = 0
    try:
        with tf.Session() as sess:
            # tensorflow model import and restore
            print("model importing...")
            cnn = CNN(sess, config.NAME, config.MODEL_PATH)
            cnn.initialize()
            print("model imported!")

            prev = -1
            while True:
                if prev == shared_pos.value:
                    print(prev, shared_pos.value)
                    time.sleep(0.05)
                    continue
                t3 = time.time()

                # acquire lock
                lock.acquire()

                # get current_pos to get a slice of mfcc from shared_mfcc
                current_pos = shared_pos.value
                current_time = shared_time[current_pos]
                X = shared_mfcc[current_pos,:]
                X = X.reshape((1,N_MFCC,N_FRAME,1))
                print(X.shape)
                prev = shared_pos.value

                # release lock
                lock.release()

                y_pred = cnn.predict(X)
                percent = '%.2f'% y_pred[0,1] # probability of detecting drone
                print(y_pred)

                results = {'device': DEVICE_NAME,
                           'current_tme': str(int(current_time)),
                           'percent_detected': str(percent)}
                ws.send(json.dumps(results))
                # i+=1
                t4 = time.time()
                print("########time for detection:" , t4-t3)
                time.sleep(0.1)

    except KeyboardInterrupt:
        print('ended')

    # after exiting the loop
    print('ended')


def init_server():
    print("audio server start")

    # sample rate/chunk size == 4.0 (0.25 window)
    buffer_len = int(BUFFER_HOURS * 60 * 60 * (SAMPLE_RATE/float(CHUNK_SIZE)))
    print(buffer_len)

    # create shared memory
    lock = mp.Lock()
    shared_mfcc_base = mp.Array(ctypes.c_double, buffer_len*N_FRAME*N_MFCC, lock=False)
    shared_mfcc = np.frombuffer(shared_mfcc_base)
    shared_mfcc = shared_mfcc.reshape(buffer_len, N_FRAME*N_MFCC)
    shared_time = mp.Array(ctypes.c_double, buffer_len, lock=False)
    shared_pos = mp.Value('i', -1, lock=False)

    # start 2 processes:
    # 1. a process to save audio data and mfcc values
    # 2. a process to run the trained model and send results to the server
    p1 = mp.Process(target=process_audio, args=(shared_mfcc, shared_time, shared_pos, lock))
    p2 = mp.Process(target=process_requests, args=(shared_mfcc, shared_time, shared_pos, lock))
    p1.start()
    p2.start()


if __name__ == '__main__':
    init_server()
