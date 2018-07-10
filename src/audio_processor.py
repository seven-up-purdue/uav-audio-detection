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

##### Fixed configuration #####
CHUNK_SIZE = 8192
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
N_MFCC = 13

##### User configuration #####
config = Config('../config.yaml')
BUFFER_HOURS = config.BUFFER_HOURS
WAVE_FILENAME = config.WAV_PATH
model = Model(config.MODEL_TYPE, config.MODEL_PATH)
# print(config.__dict__)

SERVER_ADDRESS = config.SERVER_ADDRESS



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
    j = 1
    try:
        # while data != '':
        while True:
            t1 = time.time()
            # grab audio and timestamp
            audio = np.fromstring(stream.read(CHUNK_SIZE), np.int16)
            # calculate mfcc from the chunk
            mfcc = librosa.feature.mfcc(audio.astype(float), sr=44100, n_mfcc=13).T

            # acquire lock
            lock.acquire()

            # increment index counter by 17
            shared_pos.value = (shared_pos.value + 17) % len(shared_time)
            # record current time
            shared_time[shared_pos.value // 17] = t1
            # save mfcc values of the current chunk
            shared_mfcc[shared_pos.value:shared_pos.value+17,:] = mfcc
            # print("shared_pos.value: ", shared_pos.value)

            # write to the audio file
            wf.writeframes(b''.join(audio))
            j+=1

            # release lock
            lock.release()
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
    i = 0
    ws = create_connection("ws://%s/ws?device=rpi"%SERVER_ADDRESS)
    # ws = create_connection("ws://192.168.1.4:8090/ws?device=rpi")
    # time.sleep(1)
    prev = -1
    with tf.Session() as sess:
        # restore pretrained model
        model.saver.restore(sess, model.model_path)
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
            current_time = shared_time[current_pos // 17]
            X = shared_mfcc[current_pos:current_pos+17,:]
            prev = shared_pos.value
            # release lock
            lock.release()

            y_pred = sess.run(tf.argmax(model.y_, 1), feed_dict={model.X: X})
            print(y_pred)
            percent = (y_pred == 1).sum() / len(y_pred) * 100
            print(percent)
            results = {'device': 'Macbook Air',
                       'current_tme': str(int(current_time)),
                       'percent_detected': str(percent)}
            ws.send(json.dumps(results))
            i+=1
            t4 = time.time()
            print("########time for detection:" , t4-t3)
            #time.sleep(0.1)

def init_server():
    print("audio server start")
    print("model importing...")
    model.initialize()
    print("model imported!")
    # figure out how big the buffer needs to be to contain BUFFER_HOURS of audio
    # buffer_len = int(BUFFER_HOURS * 60 * 60 * (SAMPLE_RATE / float(CHUNK_SIZE)))
    # get the size of mfcc
    buffer_len = int(BUFFER_HOURS * 60 * 60 * (SAMPLE_RATE / float(CHUNK_SIZE)) * 17)
    print(buffer_len)
    # create shared memory
    lock = mp.Lock()
    shared_mfcc_base = mp.Array(ctypes.c_double, buffer_len*N_MFCC, lock=False)
    shared_mfcc = np.frombuffer(shared_mfcc_base)
    shared_mfcc = shared_mfcc.reshape(buffer_len, N_MFCC)
    # shared_audio = mp.Array(ctypes.c_short, buffer_len, lock=False)
    shared_time = mp.Array(ctypes.c_double, buffer_len, lock=False)
    shared_pos = mp.Value('i', 0, lock=False)

    np.set_printoptions(precision=8)

    # start 2 processes:
    # 1. a process to save audio data and mfcc values
    # 2. a process to run the trained model and send results to the server
    p1 = mp.Process(target=process_audio, args=(shared_mfcc, shared_time, shared_pos, lock))
    p2 = mp.Process(target=process_requests, args=(shared_mfcc, shared_time, shared_pos, lock))
    p1.start()
    p2.start()


if __name__ == '__main__':
    #config = Config(config_path)
    init_server()
