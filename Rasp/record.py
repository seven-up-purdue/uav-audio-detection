'''
compile like this:
python3 record.py filename
ex> python3 record.py test1
'''
import pyaudio
import time
import datetime
import wave
import numpy as np 
import sys
#chunk_num = 180 # 1Minute
def getStream(sample_rate = 22050, chunk_size = 8192,chunk_num = 180, isWrite=False):  
   AUDIO_FORMAT = pyaudio.paInt16
   SAMPLE_RATE = sample_rate
   CHUNK_SIZE = chunk_size
   CHUNK_NUM = chunk_num
   p = pyaudio.PyAudio()
       
   
 
   while(True):
       WAVE_FILENAME = './Record/'+datetime.datetime.now().strftime('%m-%d %H_%M_%S')+'.wav'
       stream = p.open(format=AUDIO_FORMAT, channels=1, rate=SAMPLE_RATE,
   input=True, frames_per_buffer=CHUNK_SIZE)
       frame = []  
       t1 = time.time()
       cn = 0
       for i in range(CHUNK_NUM):
           frame.append(stream.read(CHUNK_SIZE,exception_on_overflow = False))
           cn+=1
           
       frame = b''.join(frame)
       audio = np.fromstring(frame, np.int16)
       
       t2 = time.time()
       
       # write to the audio file
       wf = wave.open(WAVE_FILENAME, 'wb')
       wf.setnchannels(1)
       wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
       wf.setframerate(SAMPLE_RATE)
       wf.writeframes(b''.join(audio))
       print("time: %.4f \t"%(t2-t1),end='')
       
       stream.stop_stream()
       stream.close()

getStream()