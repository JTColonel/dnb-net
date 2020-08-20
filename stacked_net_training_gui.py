from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import tensorflow.keras 
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Dropout, LeakyReLU, Multiply, Add
from tensorflow.keras.models import Model, Sequential, load_model, clone_model
from tensorflow.keras.activations import linear
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd
import argparse
import os
import librosa

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tkinter import *
import soundfile as sf
import argparse
import pyaudio
from scipy import signal
import time
import sys
import scipy, pylab
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils


global alpha
global beta
beta = K.variable(3e-7)
alpha = K.variable(0.3)
np.random.seed(90210)

class Args:
    def __init__(self):
        self.n_epochs = 100
        self.net_type = 'ae'
        self.skip = True
        self.filename_in_1 = []
        self.filename_in_2 = []
        self.drum_frames = []
        self.bass_frames = []
        self.filename_out = []
        self.trained_model_name = ''

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
    def Manne_init(self, args):
        self.frames = []
        self.X_train = []
        self.X_val = []
        self.X_test = []
        self.encoder = []
        self.decoder = []
        self.network = []
        self.dnb_net = []
        self.encoder_widths = []
        self.decoder_widths = []

        self.z_mean = K.placeholder(shape=(8,))
        self.z_log_var = K.placeholder(shape=(8,))
        self.beta_changer = []

        self.n_epochs = args.n_epochs
        self.net_type = args.net_type
        self.skip = args.skip
        self.filename_in_1 = args.filename_in_1
        self.filename_in_2 = args.filename_in_2
        self.filename_out = args.filename_out
        self.trained_model_name = args.trained_model_name

    def do_everything(self):
        self.load_dataset()
        self.define_net()
        self.train_net()

    def my_mse2(self, inputs, outputs):
        inspec1, outspec1 = inputs[0], outputs[0]
        inspec2, outspec2 = inputs[1], outputs[1]
        outloss = K.sum(mse(inspec1,outspec1)+mse(inspec2,outspec2))
        return outloss

    def load_dataset(self):
        global drum_frames
        global bass_frames 

        orig_frames_1 = drum_frames 
        np.random.shuffle(orig_frames_1)
        len_frames = orig_frames_1.shape[0]

        orig_frames_2 = bass_frames
        np.random.shuffle(orig_frames_2)
        len_frames = orig_frames_2.shape[0]

        self.frames = np.hstack((orig_frames_1[:int(self.datasize.get()),:],orig_frames_2[:int(self.datasize.get()),:])) 

        orig_frames_1 = None
        orig_frames_2 = None

        orig_frames = None


        in_size = int(self.frames.shape[0])
        self.X_train =  self.frames[:int(0.95*in_size),:] 
        self.X_val =  self.frames[int(0.95*in_size):int(0.975*in_size),:]
        self.X_test =  self.frames[int(0.975*in_size):,:]



    def define_net(self):
        global decoder_outdim
        if self.net_type=='vae':
            a=1
        else:
            l2_penalty = 1e-7

        self.encoder_widths = [1024,512,256,128,64,32,16,10]
        self.decoder_widths = [16,32,64,128,256,512,1024]


        decoder_outdim = 2049
        drop = 0.0
        alpha_val=0.1
        input_spec1 = Input(shape=(decoder_outdim,))
        input_spec2 = Input(shape=(decoder_outdim,))

        encoded1 = Dense(units=self.encoder_widths[0],
                activation=None,
                kernel_regularizer=l2(l2_penalty))(input_spec1)
        encoded2 = Dense(units=self.encoder_widths[0],
                activation=None,
                kernel_regularizer=l2(l2_penalty))(input_spec2)
        encoded1 = LeakyReLU(alpha=alpha_val)(encoded1)
        encoded2 = LeakyReLU(alpha=alpha_val)(encoded2)
        for width in self.encoder_widths[1:-1]:
            encoded1 = Dense(units=width,
                activation=None,
                kernel_regularizer=l2(l2_penalty))(encoded1)
            encoded2 = Dense(units=width,
                activation=None,
                kernel_regularizer=l2(l2_penalty))(encoded2)
            encoded1 = LeakyReLU(alpha=alpha_val)(encoded1)
            encoded2 = LeakyReLU(alpha=alpha_val)(encoded2)

        encoded1 = Dense(units=self.encoder_widths[-1], activation='sigmoid', kernel_regularizer=l2(l2_penalty))(encoded1)
        encoded2 = Dense(units=self.encoder_widths[-1], activation='sigmoid', kernel_regularizer=l2(l2_penalty))(encoded2)

        self.encoder = Model(inputs=[input_spec1,input_spec2], outputs=[encoded1,encoded2])

        if self.net_type == 'vae':
            a=1
        else:
            a = 1

        if self.skip == True:
            input_latent1 = Input(shape=(self.encoder_widths[-1],))
            input_latent2 = Input(shape=(self.encoder_widths[-1],))
        else:
            a=1

        decoded1 = Dense(units=self.decoder_widths[0],
            activation=None,
            kernel_regularizer=l2(l2_penalty))(input_latent1)
        decoded2 = Dense(units=self.decoder_widths[0],
            activation=None,
            kernel_regularizer=l2(l2_penalty))(input_latent2)
        decoded1 = LeakyReLU(alpha=alpha_val)(decoded1)
        decoded2 = LeakyReLU(alpha=alpha_val)(decoded2)
        for width in self.decoder_widths[1:]:
            decoded1 = Dense(units=width,
                activation=None,
                kernel_regularizer=l2(l2_penalty))(decoded1)
            decoded2 = Dense(units=width,
                activation=None,
                kernel_regularizer=l2(l2_penalty))(decoded2)
            decoded1 = LeakyReLU(alpha=alpha_val)(decoded1)
            decoded2 = LeakyReLU(alpha=alpha_val)(decoded2)

        decoded1 = Dense(units=int(decoder_outdim),
            activation='relu',
            kernel_regularizer=l2(l2_penalty))(decoded1)
        decoded2 = Dense(units=int(decoder_outdim),
            activation='relu',
            kernel_regularizer=l2(l2_penalty))(decoded2)

        self.decoder = Model(inputs=[input_latent1,input_latent2], outputs=[decoded1,decoded2])

        my_batch = [input_spec1,input_spec2]
        my_encoded = self.encoder(my_batch)
        my_decoded = self.decoder(my_encoded)
        self.network = Model(inputs=my_batch, outputs=my_decoded)

        print('\n net summary \n')
        self.network.summary()
        print('\n enc summary \n')
        self.encoder.summary()
        print('\n dec summary \n')
        self.decoder.summary()



    def train_net(self):
        global decoder_outdim
        adam_rate = 1e-4
        if self.skip == True: #Handling case where Keras expects two inputs
            train_data = [self.X_train[:,:2049],self.X_train[:,2049:4098]]#,self.X_train[:,4098:4113],self.X_train[:,4113:]]
            train_target = [self.X_train[:,:2049],self.X_train[:,2049:4098]]
            val_data = [self.X_val[:,:2049],self.X_val[:,2049:4098]]#,self.X_val[:,4098:4113],self.X_val[:,4113:]]
            val_target = [self.X_val[:,:2049],self.X_val[:,2049:4098]]
        else:
            a=1

        if self.net_type == 'vae':
            a=1

        else:
            self.network.compile(optimizer=Adam(lr=adam_rate), loss=self.my_mse2)
            self.network.fit(x=train_data, y=train_target,
                    epochs=int(self.epochs.get()),
                    batch_size=200,
                    shuffle=True,
                    validation_data=(val_data, val_target)
                    )


        modalpha1 = Input(shape=(self.encoder_widths[-1],))
        modnegalpha1 = Input(shape=(self.encoder_widths[-1],))
        modalpha2 = Input(shape=(self.encoder_widths[-1],))
        modnegalpha2 = Input(shape=(self.encoder_widths[-1],))
        final_spec_1 = Input(shape=(decoder_outdim,))#drum track 1
        final_spec_2 = Input(shape=(decoder_outdim,))#drum track 2
        final_spec_3 = Input(shape=(decoder_outdim,))#bass track 1
        final_spec_4 = Input(shape=(decoder_outdim,))#bass track 2

        my_batch0 = [final_spec_1,final_spec_3]#drum track 1, bass track 1
        my_encoded0 = self.encoder(my_batch0)
        my_batch1 = [final_spec_2,final_spec_4]#drum track 2, bass track 2
        my_encoded1 = self.encoder(my_batch1)
        blarg0 = Multiply()([my_encoded0[0],modalpha1])
        blarg1 = Multiply()([my_encoded1[0],modnegalpha1])
        mod_latent1 = Add()([blarg0,blarg1])
        belch0 = Multiply()([my_encoded0[1],modalpha2])
        belch1 = Multiply()([my_encoded1[1],modnegalpha2])
        mod_latent2 = Add()([belch0,belch1])
        final_decoded = self.decoder([mod_latent1,mod_latent2])
        self.dnb_net = Model(inputs=[final_spec_1,final_spec_2,final_spec_3,final_spec_4,modalpha1,modnegalpha1,modalpha2,modnegalpha2],
            outputs=final_decoded)

        self.dnb_net.save('models/'+self.model_name.get()+'_trained_dnb.h5')
        print('Done training!')

    def save_latents(self):

        indat = self.frames
        enc_mag = self.encoder.predict(indat,verbose=1)

        if self.net_type == 'vae':
            a = enc_mag[0]
            b = enc_mag[1]
            print(a.shape)
            print(b.shape)
            enc_mag = np.hstack((enc_mag[0],enc_mag[1]))

        df = pd.DataFrame(enc_mag)
        df.to_csv('encoded_mags.csv')

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
        global drum_frames 

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = 'audio/'+self.track1_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        drum1, _ = librosa.load(data_path, sr=44100, mono=True)

        D = librosa.stft(drum1,n_fft=4096, window='hann')
        temp = D[:,:]
        temp = np.abs(temp)
        temp = temp / (temp.max(axis=0)+0.000000001)
        temp = np.transpose(temp)
        drum_frames = temp[~np.all(temp == 0, axis=1)]

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
        global bass_frames 

        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        filename_in = 'audio/'+self.track3_name.get()
        data_path = os.path.join(os.getcwd(),filename_in)
        bass1, _ = librosa.load(data_path, sr=44100, mono=True)

        D = librosa.stft(bass1,n_fft=4096, window='hann')
        temp = D[:,:]
        temp = np.abs(temp)
        temp = temp / (temp.max(axis=0)+0.000000001)
        temp = np.transpose(temp)
        bass_frames = temp[~np.all(temp == 0, axis=1)]

        print('bass loaded')


    def quit(self):
        root.destroy()

    def start_and_train(self):
        self.Manne_init(self.Args)
        self.do_everything()
        

    def createWidgets(self):

        self.DATATEXT = Label(self, text='Minimum Length of Dataset')
        self.DATATEXT.pack()
        self.DATATEXT.place(relx=0.35,rely=0.05)

        self.EPOCHTEXT = Label(self, text='Number of Iterations to Train the Network')
        self.EPOCHTEXT.pack()
        self.EPOCHTEXT.place(relx=0.35,rely=0.15)
        
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.pack()
        self.QUIT.place(relx=0.45,rely=0.65)

        self.model_name = Entry(self)
        self.model_name.pack()
        self.model_name.place(relx=0.4,rely=0.52)
        self.label = Label(self,text='Output Model Name')
        self.label.pack()
        self.label.place(relx=0.25,rely=0.52)

        self.track1_name = Entry(self)
        self.track1_name.pack()
        self.track1_name.place(relx=0.40,rely=0.32)
        self.label_1 = Label(self,text='Drum Audio')
        self.label_1.pack()
        self.label_1.place(relx=0.30,rely=0.32)

        self.track3_name = Entry(self)
        self.track3_name.pack()
        self.track3_name.place(relx=0.40,rely=0.42)
        self.label_3 = Label(self,text='Bass Audio')
        self.label_3.pack()
        self.label_3.place(relx=0.30,rely=0.42)

        self.START = Button(self)
        self.START["text"] = "START"
        self.START["fg"]   = "green"
        self.START["command"] =  lambda: self.start_and_train()
        self.START.pack()
        self.START.place(relx=0.45,rely=0.6)

        self.LOAD = Button(self)
        self.LOAD["text"] = "LOAD DRUMS"
        self.LOAD["fg"]   = "black"
        self.LOAD["command"] =  lambda: self.process_drums()
        self.LOAD.pack()
        self.LOAD.place(relx=0.6,rely=0.32)

        self.LOAD = Button(self)
        self.LOAD["text"] = "LOAD BASS"
        self.LOAD["fg"]   = "black"
        self.LOAD["command"] =  lambda: self.process_bass()
        self.LOAD.pack()
        self.LOAD.place(relx=0.6,rely=0.42)

    def createButtons(self):
        global chroma_val
        self.datasize = IntVar()
        self.datasize.set(2500)
        NOTE_OPTIONS = [
        ('one minute',2500),
        ('four minutes',10000),
        ('ten minutes',25000)
        ]
        xx = 1.3

        for text, val in NOTE_OPTIONS:
            b = Radiobutton(self, text=text, value=val, variable=self.datasize)
            b.pack()
            b.place(relx=xx/5.,rely=0.1)
            xx+=1

    def createEpochButtons(self):
        global chroma_val
        self.epochs = IntVar()
        self.epochs.set(50)
        NOTE_OPTIONS = [
        ('50 iterations',50),
        ('150 iterations',150),
        ('300 iterations',300)
        ]
        xx = 1.3

        for text, val in NOTE_OPTIONS:
            b = Radiobutton(self, text=text, value=val, variable=self.epochs)
            b.pack()
            b.place(relx=xx/5.,rely=0.2)
            xx+=1


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

        Frame.__init__(self, master,width=1000, height=800)
        self.pack()
        self.createWidgets()
        self.createButtons()
        self.createEpochButtons()
        recorded_scales = []
        self.Args = Args()


global app 
global num_latents
num_latents = 10
root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
