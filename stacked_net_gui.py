from tkinter import *
import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import argparse
import pyaudio
import numpy as np
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model, load_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy import signal
import time
import sys
import scipy, pylab

RATE     = int(44100)
CHUNK    = int(1024)
CHANNELS = int(1)
NUM_CHUNKS = 10
ind = 0
proc_ind = 2
len_window = 4096 #Specified length of analysis window
hop_length_ = 1024 #Specified percentage hop length between windows
xfade_in = np.linspace(0,1,num=CHUNK)
xfade_out = np.flip(xfade_in)

class Application(Frame):
    global make_sine2
    def make_sine2(seg_length,ii):
        global bass1
        global bass2
        global drum1
        global drum2 
        global CHUNK
        global encoder
        global full_net
        global full_net_graph
        global scales 
        global drum_scales
        global drum_gain
        global bass_scales
        global bass_gain 
        global app 
        global num_latents
        global new_data
        global make_audio
        global drumalpha
        global bassalpha
        global len_window
        global tick 
        num_samps = seg_length*CHUNK
        make_audio = False
        
        # num_samps = seg_length*CHUNK

        the_range = range((num_samps*ii),(num_samps*(ii+1)+4*CHUNK))
        drumsnip1 = drum1.take(the_range,mode='wrap') #Snip Track 1
        drumsnip2 = drum2.take(the_range,mode='wrap') #Snip Track 2         
        basssnip1 = bass1.take(the_range,mode='wrap')#Snip Track 1
        basssnip2 = bass2.take(the_range,mode='wrap') #Snip Track 2         
        # basssnip2 = bass2[(num_samps*ii):(num_samps*(ii+1)+4*CHUNK)] #Snip Track 2         


        _,_,D1 = signal.stft(drumsnip1, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = D1 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberD1 = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magD1 = mag / rememberD1 #Normalizing
        phaseD1 = np.angle(D1) #Phase response of STFT
        magD1 = magD1.T

        _,_,D2 = signal.stft(drumsnip2, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = D2 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberD2 = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magD2 = mag / rememberD2 #Normalizing
        phaseD2 = np.angle(D2) #Phase response of STFT
        magD2 = magD2.T

        _,_,B1 = signal.stft(basssnip1, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = B1 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberB1 = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magB1 = mag / rememberB1 #Normalizing
        phaseB1 = np.angle(B1) #Phase response of STFT
        magB1 = magB1.T

        _,_,B2 = signal.stft(basssnip2, nperseg=len_window, nfft=len_window, fs=44100, noverlap=3*1024) #STFT
        mag = B2 
        mag = np.abs(mag) #Magnitude response of the STFT
        rememberB2 = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        magB2 = mag / rememberB2 #Normalizing
        phaseB2 = np.angle(B2) #Phase response of STFT
        magB2 = magB2.T

        temp_drumalpha = np.tile(drumalpha*drum_scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 1
        temp_drumnegalpha = np.tile((1-drumalpha)*drum_scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 2
        temp_drumphase =drumalpha*phaseD1+(1-drumalpha)*phaseD2 #Unstack and Interpolate Phase
        temp_drumremember = drumalpha*rememberD1+(1-drumalpha)*rememberD2 #Unstack and Interpolate Normalizing gains

        temp_bassalpha = np.tile(bassalpha*bass_scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 1
        temp_bassnegalpha = np.tile((1-bassalpha)*bass_scales,(NUM_CHUNKS+5,1)) #Match dims for XFade 2
        temp_bassphase =bassalpha*phaseB1+(1-bassalpha)*phaseB2 #Unstack and Interpolate Phase
        temp_bassremember = bassalpha*rememberB1+(1-bassalpha)*rememberB2 #Unstack and Interpolate Normalizing gains

        a = full_net.predict([magD1,magD2,magB1,magB2,temp_drumalpha,temp_drumnegalpha,temp_bassalpha,temp_bassnegalpha])

        temp_drum_mag = a[0]
        temp_bass_mag = a[1]


        drum_out_mag = temp_drum_mag.T * temp_drumremember
        D = drum_out_mag*np.exp(1j*temp_drumphase)
        _, now_out = np.float32(signal.istft(0.24*D, fs=44100, noverlap=3*1024))
        drum_out = now_out[CHUNK:-2*CHUNK]

        bass_out_mag = temp_bass_mag.T * temp_bassremember
        B = bass_out_mag*np.exp(1j*temp_bassphase)
        _, now_out = np.float32(signal.istft(0.24*B, fs=44100, noverlap=3*1024))
        bass_out = now_out[CHUNK:-2*CHUNK]

        out = 4*(drum_gain*drum_out+bass_gain*bass_out)

        myshape = int(len(out)/CHUNK)

        new_data = out.reshape((myshape,CHUNK))

    global callback 
    def callback(in_data, frame_count, time_info, status):
        global ind
        global proc_ind 
        global NUM_CHUNKS
        global all_data
        global new_data
        global make_audio
        global tick 
        global all_data

        if ind>=NUM_CHUNKS:
            ind = 0
            proc_ind+=1
        if ind==0:
            xfade = xfade_in*new_data[0,:] + xfade_out*all_data[-1,:]
            all_data = new_data
            all_data[0,:] = xfade
        if ind==1:
            make_audio = True
            tick = time.time()

        data = all_data[ind,:] #Send a chunk to the audio buffer when it asks for one
        ind +=1 
        return (data, pyaudio.paContinue)  


    def model_to_mem(self):
        global encoder
        global enc_graph
        global decoder 
        global dec_graph
        global full_net
        global full_net_graph

        data_path_net = os.path.join(os.getcwd(),'models/'+self.model_name.get()+'_trained_dnb.h5')
        full_net = load_model(data_path_net, compile=False)
        full_net._make_predict_function()
        full_net_graph = tf.get_default_graph()


    def process_drums(self):
        global mag1
        global phase1
        global remember1
        global mag2
        global phase2
        global remember2
        global drum1
        global drum2 
        global smallest_drums

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = 'audio/'+self.track1_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        drum1, _ = librosa.load(data_path, sr=44100, mono=True)

        filename_in = 'audio/'+self.track2_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        drum2, _ = librosa.load(data_path, sr=44100, mono=True)

        smallest_drums = np.min((len(drum1),len(drum2)))
        smallest_drums = int(smallest_drums/CHUNK)
        print(smallest_drums)

        print('drums loaded')

    def process_bass(self):
        global mag1
        global phase1
        global remember1
        global mag2
        global phase2
        global remember2
        global bass1
        global bass2 
        global smallest_bass

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = 'audio/'+self.track3_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        bass1, _ = librosa.load(data_path, sr=44100, mono=True)

        filename_in = 'audio/'+self.track4_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        bass2, _ = librosa.load(data_path, sr=44100, mono=True)

        smallest_bass = np.min((len(bass1),len(bass2)))
        smallest_bass = int(smallest_bass/CHUNK)
        print(smallest_bass)

        print('bass loaded')


    def start_net(self):
        global p 
        global stream
        global make_audio
        global NUM_CHUNKS
        global proc_ind
        global tick 

        tick = time.time()

        make_audio = True
        make_sine2(NUM_CHUNKS,1)

        p = pyaudio.PyAudio()
        print("opening stream")
        stream = p.open(format=pyaudio.paFloat32,
                        channels=CHANNELS,
                        frames_per_buffer=CHUNK,
                        rate=RATE,
                        output=True,
                        stream_callback=callback)


        stream.start_stream()
        time.sleep(0.1)

    def pause_sounds(self):
        global p 
        global stream
        global ind
        global proc_ind 
        
        stream.stop_stream()
        print('sounds paused')
        stream.close()
        p.terminate()
        ind = NUM_CHUNKS+1
        proc_ind = 0

    def quit(self):
        root.destroy()
        

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack()
        self.QUIT.place(relx=0.45,rely=0.95)

        self.model_name = Entry(self)
        self.model_name.pack()
        self.model_name.place(relx=0.4,rely=0.37)
        self.label = Label(self,text='Model Name')
        self.label.pack()
        self.label.place(relx=0.25,rely=0.37)

        self.LOADMODEL = Button(self)
        self.LOADMODEL["text"] = "LOAD MODEL"
        self.LOADMODEL["fg"]   = "black"
        self.LOADMODEL["command"] =  lambda: self.model_to_mem()
        self.LOADMODEL.pack()
        self.LOADMODEL.place(relx=0.62,rely=0.37)

        self.track1_name = Entry(self)
        self.track1_name.pack()
        self.track1_name.place(relx=0.20,rely=0.32)
        self.label_1 = Label(self,text='Drum 1')
        self.label_1.pack()
        self.label_1.place(relx=0.14,rely=0.32)

        self.track2_name = Entry(self)
        self.track2_name.pack()
        self.track2_name.place(relx=0.66,rely=0.32)
        self.label_2 = Label(self,text='Drum 2')
        self.label_2.pack()
        self.label_2.place(relx=0.6,rely=0.32)

        self.track3_name = Entry(self)
        self.track3_name.pack()
        self.track3_name.place(relx=0.20,rely=0.42)
        self.label_3 = Label(self,text='Bass 1')
        self.label_3.pack()
        self.label_3.place(relx=0.14,rely=0.42)

        self.track4_name = Entry(self)
        self.track4_name.pack()
        self.track4_name.place(relx=0.66,rely=0.42)
        self.label_4 = Label(self,text='Bass 2')
        self.label_4.pack()
        self.label_4.place(relx=0.6,rely=0.42)

        self.START = Button(self)
        self.START["text"] = "START"
        self.START["fg"]   = "green"
        self.START["command"] =  lambda: self.start_net()
        self.START.pack()
        self.START.place(relx=0.45,rely=0.9)

        self.PAUSE = Button(self)
        self.PAUSE["text"] = "PAUSE"
        self.PAUSE["fg"]   = "black"
        self.PAUSE["command"] =  lambda: self.pause_sounds()
        self.PAUSE.pack()
        self.PAUSE.place(relx=0.45,rely=0.85)

        self.LOAD = Button(self)
        self.LOAD["text"] = "LOAD DRUMS"
        self.LOAD["fg"]   = "black"
        self.LOAD["command"] =  lambda: self.process_drums()
        self.LOAD.pack()
        self.LOAD.place(relx=0.45,rely=0.32)

        self.LOAD = Button(self)
        self.LOAD["text"] = "LOAD BASS"
        self.LOAD["fg"]   = "black"
        self.LOAD["command"] =  lambda: self.process_bass()
        self.LOAD.pack()
        self.LOAD.place(relx=0.45,rely=0.42)

        self.FADEDRUM = Scale(self,from_=100, to=0,length=300, orient='horizontal')
        self.FADEDRUM.set(0)
        self.FADEDRUM.pack()
        self.FADEDRUM.place(relx=0.30,rely=0.25)

        self.DRUMVOL = Scale(self,from_=150, to=0,length=150, orient='vertical')
        self.DRUMVOL.set(50)
        self.DRUMVOL.pack()
        self.DRUMVOL.place(relx=0.01,rely=0.35)

        self.FADEBASS = Scale(self,from_=100, to=0,length=300, orient='horizontal')
        self.FADEBASS.set(0)
        self.FADEBASS.pack()
        self.FADEBASS.place(relx=0.30,rely=0.47)

        self.BASSVOL = Scale(self,from_=150, to=0,length=150, orient='vertical')
        self.BASSVOL.set(50)
        self.BASSVOL.pack()
        self.BASSVOL.place(relx=0.93,rely=0.35)



    def createDrumSliders(self):
        global drum_scales 
        global num_latents
        drum_scales = np.ones(num_latents)
        self.drum_scale_list = []
        for w in range(num_latents):
            drum_scale = Scale(self,from_=200, to=0,length=200)
            drum_scale.pack()
            drum_scale.place(relx=w/float(num_latents),rely=0.01)
            drum_scale.set(100)
            drum_scales[w]=drum_scale.get()
            self.drum_scale_list.append(drum_scale)

    def createBassSliders(self):
        global bass_scales 
        global num_latents
        bass_scales = np.ones(num_latents)
        self.bass_scale_list = []
        for w in range(num_latents):
            bass_scale = Scale(self,from_=200, to=0,length=200)
            bass_scale.pack()
            bass_scale.place(relx=w/float(num_latents),rely=0.55)
            bass_scale.set(100)
            bass_scales[w]=bass_scale.get()
            self.bass_scale_list.append(bass_scale)

    def update_scales(self):
        global scales 
        global recorded_scales
        global make_audio
        global NUM_CHUNKS
        global proc_ind
        global drumalpha
        global bassalpha
        global temp_drum_scales
        global temp_bass_scales
        global drum_scales
        global drum_gain
        global bass_scales
        global bass_gain 
        global smallest_bass
        global smallest_drums

        smallest = np.min((smallest_bass,smallest_drums))

        drumalpha = self.FADEDRUM.get()/100.
        bassalpha = self.FADEBASS.get()/100.
        drum_gain = self.DRUMVOL.get()/50.
        bass_gain = self.BASSVOL.get()/50.
        if make_audio:
            proc_ind = np.mod(proc_ind,smallest)
            make_sine2(NUM_CHUNKS,proc_ind)


        for w in range(num_latents):
            temp_drum_scales[w]=self.drum_scale_list[w].get()/100.
            temp_bass_scales[w]=self.bass_scale_list[w].get()/100.
        drum_scales = temp_drum_scales
        bass_scales = temp_bass_scales

        self.after(POLL_TIME, self.update_scales)


    def __init__(self, master=None):
        global recorded_scales
        global POLL_TIME
        global make_audio
        global alpha
        global temp_scales
        global temp_drum_scales
        global temp_bass_scales
        global all_data
        global smallest_bass
        global smallest_drums

        alpha = 1
        temp_scales = np.ones(num_latents)
        temp_drum_scales = np.ones(num_latents)
        temp_bass_scales = np.ones(num_latents)
        all_data = np.zeros((21,1024))
        smallest_drums = 100
        smallest_bass = 100

        make_audio = False
        POLL_TIME = 1

        Frame.__init__(self, master,width=800, height=800)
        self.pack()
        self.createWidgets()
        self.createDrumSliders()
        self.createBassSliders()
        recorded_scales = []
        self.update_scales()

global app 
global num_latents
num_latents = 10
root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()