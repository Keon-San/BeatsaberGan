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

def getBPM(directory, fileName, finalName, audioFile):
    try:
        testFile = open(os.path.join(os.path.join(directory, filename), finalName))
        beatpm = json.load(testFile)
        return beatpm
    except IOError:
        audio = AudioSegment.from_wav(os.path.join(os.path.join(directory, filename), audioFile))
        audio = audio.set_channels(1)
        sr = audio.frame_rate
        data = np.nan_to_num(np.array(audio.get_array_of_samples('f')))
        print("lib")
        bpm, _ = librosa.beat.beat_track(y=data.astype(float), sr=sr)
        return bpm

class song:
    def __init__(self, bpm, maps, wavFile, lenAudio):
        self.wavFile = wavFile
        self.bpm = bpm
        self.datas = []
        self.maps = []
        self.lenAudio = lenAudio
        for x in range(0, 64):
            self.maps.append([[0,0,0,0], [0,0,0,0], [0,0,0,0]])
        self.maps = self.maps + maps
        for x in range(0, len(maps)):
            self.makeData(x)
    
    def makeData(self, sliceNumber):
        self.datas.append(trainableData(self.maps[sliceNumber+64], self.maps[sliceNumber:sliceNumber+64], sliceNumber*(1/16), self.bpm, self.wavFile, self.lenAudio))

    def length(self):
        return len(self.maps)-64
    def get(self, loc):
        return self.maps[loc+64]

    def getData(self, loc):
        return self.datas[loc]

class trainableData:
    def __init__(self, mainSlice, extraSlices, time, bpm, wavFile, lenAudio):
        self.mainSlice = mainSlice
        self.extraSlices = extraSlices
        self.time = time
        self.wavFile = wavFile
        self.secondsIn = (time/bpm)*60
        self.bpm = bpm
        self.lenAudio = lenAudio
    
device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = nd.FeatureExtractor(501, 200).to(device)
generator = nd.Generator().to(device)
discriminator = nd.Discriminator().to(device)

lr = 1

genOptimizer = optim.Adam(generator.parameters(), lr=lr)
discrimOptimizer = optim.Adam(discriminator.parameters(), lr=lr)
extractorOptimizer = optim.Adam(extractor.parameters(), lr=lr)

songs = []

directory = 'processedData'

for filename in os.listdir(directory):
    bpm = getBPM(directory, filename, "bpm.pickle", "song.wav")
    lenAudio = json.load(open(os.path.join(os.path.join(directory, filename), "audioLength.pickle"), 'r'))
    songs.append(song(bpm, json.load(open(os.path.join(os.path.join(directory, filename), "map.pickle"), 'r')), os.path.join(os.path.join(directory, filename), "song.wav"), lenAudio))

length = 0
for x in songs:
    length += x.length()

maxNum = 0

for x in songs:
    maxNum += x.length()

while True:
    random.shuffle(songs)
    for x in songs:
        for y in range(0, x.length()):
            #Grab data from the dataset - this is REAL data, not generated data
            data = x.datas[y]
            #Read the wavfile
            sr, songData = wavfile.read(data.wavFile)
            songData = np.mean(songData, axis=1)
            #Make the spectrogram - 1000 is the best resolution I found that gives a reasonable balance of accuracy
            _, times, sxx = scipy.signal.spectrogram(songData, sr, nperseg=1000)
            #Figure out the time step per position (that's times[1] minus times[0]), divide the seconds by it to get the position, then ceiling by flooring it then adding 1
            posInSxx = (int(data.secondsIn//(times[1]-times[0]))+1)
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
            #Zero the gradients
            extractorOptimizer.zero_grad()
            discrimOptimizer.zero_grad()
            genOptimizer.zero_grad()
            #Run the data through the feature extractor, then the disciminator, then take loss, and train just the discriminator - that's because the feature extractor is optimized for the generator, not the discriminator
           
            features = extractor(torch.tensor(spectrogramArray).unsqueeze(0).unsqueeze(0).to(device).float())
            discriminatorGuess = discriminator(torch.tensor(data.mainSlice).unsqueeze(0).to(device).float(), features.detach(), torch.tensor(data.extraSlices).unsqueeze(0).to(device).float())
    
            loss = F.cross_entropy(discriminatorGuess.squeeze(0), torch.tensor([1]).to(device))
            loss.backward(retain_graph=True)
            print("Loss Discrim 1 " + str(loss.item()))
            discriminator.to("cpu")
            discrimOptimizer.step()
            discriminator.to(device)

            #This is a somewhat crude but not that bad method for grabbing random data from the dataset - I grab a random number within the data, then iterate until I get to the song that contains it, then I grab it
            data = None
            loc = random.randint(0, maxNum)
            for b in songs:
                if loc > b.length():
                    loc -= b.length()
                    continue
                else:
                    data = b.getData(loc)
                    break

            #This whole section is basically the above section - I'll put this in a function soon
            sr, songData = wavfile.read(data.wavFile)
            songData = np.mean(songData, axis=1)
            _, times, sxx = scipy.signal.spectrogram(songData, sr, nperseg=1000)
            posInSxx = (int(data.secondsIn//(times[1]-times[0]))+1)
            emptyBit = None
            if posInSxx < 99 or posInSxx > (len(sxx[0])-100):
                try:
                    emptyBit = np.zeros((501, max(100-posInSxx, 100-len(sxx[0])+posInSxx)))
                except:
                    emptyBit = None
            spectrogramArray = None
            if not (posInSxx < 99 or posInSxx > len(sxx[0])-100):
                spectrogramArray = sxx.take([*range(max(0, posInSxx-100), min(len(sxx[0]), posInSxx+100))], axis=1)
            elif posInSxx < 100:
                spectrogramArray = np.concatenate((emptyBit, sxx.take([*range(max(0, posInSxx-100), min(len(sxx[0]), posInSxx+100))], axis=1)), axis=1)
            else:
                spectrogramArray = np.concatenate((sxx.take([*range(max(0, posInSxx-100),min(len(sxx[0]), posInSxx+100))], axis=1), emptyBit), axis=1)
            #Zero gradients
            extractorOptimizer.zero_grad()
            discrimOptimizer.zero_grad()
            genOptimizer.zero_grad()
            #This section is different. Firstly, I use torch.normal to get the random input to my generator. I don't train the discriminator here - I do that later cause they have different goals
            features = extractor(torch.tensor(spectrogramArray).unsqueeze(0).unsqueeze(0).to(device).float()).squeeze(0)
            generatedData = generator(torch.normal(0, 1, (1, 100)).to(device).float(), features, torch.tensor(data.extraSlices).unsqueeze(0).to(device).float())
            discrimantorResultForGen = discriminator(generatedData.squeeze(0), features, torch.tensor(data.extraSlices).unsqueeze(0).to(device).float())
            lossForGen = F.cross_entropy(discrimantorResultForGen.squeeze(0), torch.tensor([1]).to(device))
            print("Loss for gen " + str(lossForGen.item()))
            lossForGen.backward()
            generator.to("cpu")
            genOptimizer.step()
            extractor.to("cpu")
            extractorOptimizer.step()
            extractor.to(device)
            generator.to(device)
            #And now I do the work for the discriminator - specifically, I only optimize the discriminator, not the features or generator
            extractorOptimizer.zero_grad()
            discrimOptimizer.zero_grad()
            genOptimizer.zero_grad()
            discrimantorResultForDiscrim = discriminator(generatedData.squeeze(0).detach(), features.detach(), torch.tensor(data.extraSlices).unsqueeze(0).to(device).float())
            lossForDiscrim = F.cross_entropy(discriminatorGuess.squeeze(0), torch.tensor([0]).to(device))
            print("Loss for discrim 2 " + str(lossForDiscrim.item()))
            lossForDiscrim.backward()
            discriminator.to("cpu")
            discrimOptimizer.step()
            discriminator.to(device)
