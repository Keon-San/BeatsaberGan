import networkDefinition as nd
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pickle
import librosa
import scipy.io.wavfile as wavfile
import os
import json
from pydub import AudioSegment
import random
import scipy

def getSpectrogram(wavFile, secondsIn):
    #Read the wavfile
    sr, songData = wavfile.read(wavFile)
    songData = np.mean(songData, axis=1)
    #Make the spectrogram - 1000 is the best resolution I found that gives a reasonable balance of accuracy
    test, times, sxx = scipy.signal.spectrogram(songData, sr, nperseg=1000)
    #Figure out the time step per position (that's times[1] minus times[0]), divide the seconds by it to get the position, then ceiling by flooring it then adding 1
    posInSxx = (int(secondsIn//(times[1]-times[0]))+1)
    #This is an extra bit that'll be set because there's a chance that my position is super near the start or end
    emptyBit = None
    #I grouped the bits for being past the start and past the end - it'll break if my song is less than 4 or 5 seconds, but if my song is that short I have bigger issues
    
    #posInSxx = posInSxx.astype(int)
    if posInSxx < 99 or posInSxx > (len(sxx[0])-100):
        try:
            emptyBit = np.zeros((501, max(100-posInSxx, 100-len(sxx[0])+posInSxx)))
        except:
            emptyBit = None
    spectrogramArray = None
    #This bit just takes the empty bits, and the spectrogram, to get the correct section of it into a variable
    if not (posInSxx < 99 or posInSxx > len(sxx[0])-100):
       spectrogramArray = sxx.take([*range(max(0, posInSxx-100), min(len(sxx[0]), posInSxx+100))], axis=1)
    elif posInSxx < 100:
       spectrogramArray = np.concatenate((emptyBit, sxx.take([*range(max(0, posInSxx-100), min(len(sxx[0]), posInSxx+100))], axis=1)), axis=1)
    else:
        spectrogramArray = np.concatenate((sxx.take([*range(max(0, posInSxx-100),min(len(sxx[0]), posInSxx+100))], axis=1), emptyBit), axis=1)

    return spectrogramArray

