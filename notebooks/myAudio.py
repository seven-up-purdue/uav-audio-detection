'''
@@@사용 예시@@@
from myAudio import *

Audio.getStream(sample_rate = 22050, chunk_size = 8192,chunk_num = 1, isWrite=True)
'''

import pyaudio
import time
import wave
import numpy as np

import librosa

test_var = 123
class Audio:
    def __init__(self ):
        print('hi')
        
    @staticmethod
    def getStream(sample_rate = 22050, chunk_size = 8192,chunk_num = 1, isWrite=False):  
        AUDIO_FORMAT = pyaudio.paInt16
        SAMPLE_RATE = sample_rate
        CHUNK_SIZE = chunk_size
        CHUNK_NUM = chunk_num
        
        WAVE_FILENAME = '../data/NewData/EDM30_test1.wav'
        
        p = pyaudio.PyAudio()
        stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=CHUNK_SIZE)
        
        frame = []    
        t1 = time.time()
        for i in range(CHUNK_NUM):
            frame.append(stream.read(CHUNK_SIZE,exception_on_overflow = False))
            
        frame = b''.join(frame)
        audio = np.fromstring(frame, np.int16)
        t2 = time.time()
        
        
        stream.stop_stream()
        stream.close()
        
        # write to the audio file
        if isWrite == True: 
            wf = wave.open(WAVE_FILENAME, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(audio))
            
        print("time: %.4f \t"%(t2-t1),end='')
    
        return audio[:]/32768