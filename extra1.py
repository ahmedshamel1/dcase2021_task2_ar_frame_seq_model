import os
import sys
import numpy as np
#import sox
import soundfile as sf
from utils.tqdm_wrapper import tqdm
from common import com
from torch.utils.data import random_split
import librosa ,librosa.display
import librosa.core
import librosa.feature
import glob
import matplotlib.pyplot as plt



#pathAudio = "C:/Users/ahmed/Desktop/audio aug"
#files = librosa.util.find_files(pathAudio, ext=['wav']) 
#files = np.ndarray(files)
#for y in files: 
   # data = librosa.load(y, sr = 16000,mono = True)   
    #data = data[0] 
     # sf.write(''+str(y), data, samplerate=16000)    
#for i in range(199):
    #librosa.load(section_00_source_train_normal_0183_strength_1_ambient.wav, sr=None)
#for filename in glob.glob(os.path.join(path, '*.wav')):
 #   librosa.load(*.wav, sr=None, mono=mono)
 #D:\deep3\dcase2021_task2_ar_frame_seq_model\dev_data\valve\train\section_00_source_train_normal_0000_pattern_00_no_pump.wav
path="D:/deep3/dcase2021_task2_ar_frame_seq_model/dev_data/ToyCar/train/section_00_source_train_normal_0000_A1Spd28VMic1.wav"
y,sr= librosa.load(path,sr=None,mono=True) 
y = librosa.effects.pitch_shift(y,sr=16000,n_steps=3.5)
# plot time domainnnnnnnnnnnnnnnnnnnn
#y=y[12000:16000]
#duration = len(y)/sr
#time = np.arange(0,duration,1/sr)
#plt.plot(time,y)
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('ToyCar after pitch shifting.wav')
#plt.show()

#plot frequency domainnnnnnnnnnnnnnnn
y=y[12000:16000]
X = np.fft.fft(y)
X_mag = np.absolute(X)
f = np.linspace(0, sr, len(X_mag))
left_mag = X_mag[:int(len(X_mag)/2)]
left_freq = f[:int(len(f)/2)]
plt.plot(left_freq, left_mag)
plt.title("Discrete-Fourier Transform", fontdict=dict(size=15))
plt.xlabel("Frequency", fontdict=dict(size=12))
plt.ylabel("Magnitude", fontdict=dict(size=12))
plt.show()
#==============================================
#f_bins = int(len(X_mag))
#plt.plot(f[:f_bins], X_mag[:f_bins])
#plt.xlabel("Frequency (Hz)")
#plt.title("ToyCar bef pitch shifting.wav")
#plt.show()



#mel_signal = librosa.feature.melspectrogram(y,sr)
#spectrogram = np.abs(mel_signal)
#power_to_db = librosa.power_to_db(spectrogram)
#librosa.display.specshow(power_to_db, sr=sr, x_axis="time", y_axis="mel")
#plt.colorbar(label="dB")
#plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
#plt.xlabel('Time', fontdict=dict(size=15))
#plt.ylabel('Frequency', fontdict=dict(size=15))
#plt.show()

  