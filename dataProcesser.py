import json
import os
from pydub import AudioSegment
import pickle
import numpy
import sys
import networkDefinition as nd
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import librosa
import scipy.io.wavfile as wavfile
import random
import scipy

directory = 'processedData'

translateArray = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18], [], [19, 20, 21, 22, 23, 24, 25, 26, 27]]

for filename in os.listdir(directory):
    for finalName in os.listdir(os.path.join(directory, filename)):
        if ".egg" in finalName or ".ogg" in finalName:
            if (".ogg" not in finalName):    
                os.rename(os.path.join(os.path.join(directory, filename), finalName), os.path.join(os.path.join(directory, filename), finalName[:-4] + ".ogg"))

for filename in os.listdir(directory):
    audioLength = 0
    bpm = 0
    for finalName in os.listdir(os.path.join(directory, filename)):
        if finalName != "ExpertPlusStandard.dat" and finalName != "ExpertPlus.dat" and finalName != "info.dat" and ".egg" not in finalName and ".ogg" not in finalName and ".pickle" not in finalName and ".wav" not in finalName:
            os.remove(os.path.join(os.path.join(directory, filename), finalName))
            continue
        elif finalName == "info.dat":
            testFile = open(os.path.join(os.path.join(directory, filename), finalName))
            beatpm = json.load(testFile)['_beatsPerMinute']
            bpm = beatpm
            json.dump(beatpm, open(os.path.join(os.path.join(directory, filename), "bpm.pickle"), 'w'))
            #os.remove(os.path.join(os.path.join(directory, filename), finalName))
        elif ".egg" in finalName or ".ogg" in finalName:

            #print(filename.translate(non_bmp_map))
            #print(finalName.translate(non_bmp_map))
            #print(os.path.join(os.path.join(directory, filename.translate(non_bmp_map)), finalName.translate(non_bmp_map)))
                
            print("Doing an ffmpeg")
            audio = AudioSegment.from_ogg(os.path.join(os.path.join(directory, filename), finalName[:-4] + ".ogg"))
            audioLength = len(audio)/1000
            json.dump(audioLength, open(os.path.join(os.path.join(directory, filename), "audioLength.pickle"), 'w'))
            audio.export(os.path.join(os.path.join(directory, filename), "song.wav"), format="wav")

            os.remove(os.path.join(os.path.join(directory, filename), finalName[:-4] + ".ogg"))
            
            continue
        elif (finalName == "ExpertPlusStandard.dat" or finalName == "ExpertPlus.dat"):
            testFile = open(os.path.join(os.path.join(directory, filename), finalName))

            non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

            print(filename.translate(non_bmp_map))
            print(finalName.translate(non_bmp_map))
            print(os.path.join(os.path.join(directory, filename.translate(non_bmp_map)), finalName.translate(non_bmp_map)))

            listOfFiles = os.listdir(os.path.join(directory, filename))

            audioFileName = ""

            for file in listOfFiles:
                if ".egg" in file or ".ogg" in file:
                    audioFileName = file
                    break
            audiSegi = None
            infoFile = None
            if audioLength == 0:
                audiSegi = AudioSegment.from_ogg(os.path.join(os.path.join(directory, filename), audioFileName[:-4] + ".ogg"))
                audioLength = len(audiSegi)/1000

            if bpm == 0:
                infoFile = open(os.path.join(os.path.join(directory, filename), "info.dat"))
                bpm = json.load(infoFile)['_beatsPerMinute']

            data = json.load(testFile)['_notes']

            currentLocation = 0

            slices = []

            currentSlice = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]

            for x in numpy.arange(0, bpm*(audioLength/60), 0.0625):
                for y in range(currentLocation, len(data)):
                    if abs(data[y]['_time'] - x) < 0.03125:
                        #print(str(data[y]['_lineLayer']) + " " + str(data[y]['_lineIndex']) + " " + str(data[y]['_type']) + " " + str(data[y]['_cutDirection']))
                        currentSlice[data[y]['_lineLayer']][data[y]['_lineIndex']] = translateArray[data[y]['_type']][data[y]['_cutDirection']]
                    else:
                        slices.append(currentSlice)
                        currentSlice = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
                        currentLocation = y
                        break
            json.dump(slices, open(os.path.join(os.path.join(directory, filename), "map.pickle"), "w"))
            testFile.close()
            os.remove(os.path.join(os.path.join(directory, filename), finalName))
            infoFile.close()
            #audiSegi.close()
            continue

def getBPM(directory, fileName, finalName, audioFile):
    testFile = open(os.path.join(os.path.join(directory, filename), finalName))
    beatpm = json.load(testFile)['_beatsPerMinute']
    if beatpm == None or beatpm == 0:
        audio = AudioSegment.from_ogg(os.path.join(os.path.join(directory, filename), audioFile[:-4] + ".ogg"))
        audio = audio.set_channels(1)
        sr = audio.frame_rate
        data = np.nan_to_num(np.array(audio.get_array_of_samples('f')))
        return librosa.beat.beat_track(y=data.astype(float), sr=sr)
    return json.load(testFile)['_beatsPerMinute']
