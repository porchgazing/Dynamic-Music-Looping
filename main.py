# %% Dependencies
import numpy as np 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import random

# %% Algorithm Class

class MixSongs:
    def __init__ (self, songOnePath, songOneType, songTwoPath, songTwoType, outputPath, 
                winDen, hopDen, fMax, nMels,
                nSimilarSegments, startThreshold,
                probJump, totalJumps):

        self.songOnePath = songOnePath
        self.songOneType = songOneType
        self.songTwoPath = songTwoPath
        self.songTwoType = songTwoType
        self.outputPath = outputPath
        self.winDen = winDen
        self.hopDen = hopDen
        self.fMax = fMax
        self.nMels = nMels
        self.nSimilarSegments = nSimilarSegments
        self.startThreshold = startThreshold
        self.probJump = probJump
        self.totalJumps = totalJumps

    # Helper Functions
    def retrieveSong(self,file_path, file_type):
        # ---------------Inputs-----------------
        # file_path: relative path to music file
        # file_type: mp3 ,wav, etc.
        # --------------Outputs-----------------
        # rawAudio: normalized audio as a np array
        # fs: sampling frequency

        audio = AudioSegment.from_file(file_path, format=file_type)
        audio = audio.split_to_mono()[0]
        rawAudio = np.array(audio.get_array_of_samples())
        rawAudio = np.array(rawAudio / np.iinfo(rawAudio[0]).max).astype(np.float32)
        rawAudio[rawAudio == 0] = 1e-10
        fs = audio.frame_rate
        return rawAudio,fs

    def findOnset(self,rawAudio, fs, hopSize, fMax, nMels):
        # ---------------Inputs-----------------
        # # rawAudio: normalized audio as a np array
        # fs: sampling frequency
        # hopSize: the number of samples in each hop
        # fMax: maximum frequency for Mel Scale
        # nMels: number of bins in Mel Frequencies
        # --------------Outputs------------------ 
        # tempo: the determined overall tempo of the song
        # beats: the starting onset of each note
        # beatsRange: enumerating the index of the beats in a np array

        onsetEnv = librosa.onset.onset_strength(y=rawAudio, sr=fs,
                                        hop_length = hopSize,
                                         aggregate=np.median,
                                         fmax=fMax, n_mels=nMels)
        tempo, beats = librosa.beat.beat_track(y=rawAudio, sr=fs, 
                                            hop_length = hopSize,
                                            onset_envelope = onsetEnv)
        beatsRange = np.arange(len(beats))
        return tempo, beats, beatsRange

    def findChroma(self,rawAudio, fs, hopSize, winSize):
        # ---------------Inputs-----------------
        # rawAudio: normalized audio as a np array
        # fs: sampling frequency
        # hopSize: the number of samples in each hop 
        # winSize: the number of samples in each window
        # --------------Outputs------------------ 
        # chromaFeatues: chromaFeatures for the given rawAudio

        return librosa.feature.chroma_stft(y=rawAudio, sr = fs,
                                             hop_length = hopSize, win_length = winSize)

    # Functions for my approach

    def calcSegmentsCost(self):
        # This function will calculate the similarity of each segment calculated from onset detection
        # Each segment in song 1 will be compared in each segment in song2
        # Similarity is calculated from the normalized last value in the DTW output

        self.songOne, self.fsOne = self.retrieveSong(self.songOnePath, self.songOneType)
        self.winSize = self.fsOne // self.winDen
        self.hopSize = self.winSize // self.hopDen
        self.tempoOne, self.beatsOne, self.beatsOneRange = self.findOnset(self.songOne,
                                                                        self.fsOne,
                                                                        self.hopSize,
                                                                        self.fMax,
                                                                        self.nMels)
        self.chromaOne = self.findChroma(self.songOne, self.fsOne, 
                                        self.hopSize, self.winSize)
                                           
        self.songTwo, self.fsTwo = self.retrieveSong(self.songTwoPath, self.songTwoType)
        self.tempoTwo, self.beatsTwo, self.beatsTwoRange = self.findOnset(self.songTwo,
                                                                        self.fsTwo,
                                                                        self.hopSize,
                                                                        self.fMax,
                                                                        self.nMels)
        self.chromaTwo = self.findChroma(self.songTwo, self.fsTwo, 
                                        self.hopSize, self.winSize)
        
        self.costs = ()
        prevBeatOne = 0
        prevBeatTwo = 0

        print("Calculating the similarity of segments between both songs...")

        for beatOne in self.beatsOne:
            
            segmentOne = self.chromaOne[:,prevBeatOne:beatOne]

            for beatTwo in self.beatsTwo:
                segmentTwo = self.chromaTwo[:,prevBeatTwo:beatTwo]
                dist,_ = librosa.sequence.dtw(X=segmentOne, Y=segmentTwo, metric='cosine')
                dist = dist[-1,-1] / (segmentOne.shape[1] + segmentTwo.shape[1]) 
                self.costs = self.costs + (beatOne,beatTwo,dist)
                prevBeatTwo = beatTwo

            prevBeatOne = beatOne
            prevBeatTwo = 0 
        
        print("Calculated the similarity of segments between both songs!")

    def evalSegmentsCost(self):
        # This function will evaluate the cost function and keep the segments that are under 
        # the threshold input
        # nSimilarSegments: The number of possible pairs of similar segments
        # threshold: the upper limit of the cost output (aka 0.01 corresponds to 99% similar)
        #            (0<= threshold <=1)
        # startThreshold = The starting point of the threshold and decreases till it finds the optimal
        #                  threshold that meets the nSimilarSegments constraint

        self.similarSegments = np.array(self.costs).reshape(-1,3)  

        nSegments = self.similarSegments.shape[0]
        threshold = self.startThreshold
        print("Finding suitable cost threshold for both songs...")

        while (nSegments > self.nSimilarSegments):
            tempCosts = self.similarSegments[self.similarSegments[:,2] < threshold]
            nSegments = tempCosts.shape[0]
            threshold -= 0.0001

        self.similarSegmentsBoth = tempCosts[:,0:2] 

        np.random.shuffle(self.similarSegmentsBoth) # shuffling before we remove the bias from large number of same beats
        self.similarSegmentsBoth = np.array([self.similarSegmentsBoth[np.where(self.similarSegmentsBoth[:, 0] # only one unique beat allowed in
                                  == val)[0][0]] for val in np.unique(self.similarSegmentsBoth[:, 0])])       # first segment
        self.similarSegmentsBoth = np.array([self.similarSegmentsBoth[np.where(self.similarSegmentsBoth[:, 1] # only one unique beat allowed in 
                                 == val)[0][0]] for val in np.unique(self.similarSegmentsBoth[:, 1])])        # second segment
        print("Found similar segments that are this similar: ", 1-threshold)

    def genJumpSequences(self):
        # This function will calculate the order of the segments in the remix of both songs
        # If a beat is in the similar segments it is eligible to possibly jump to the similar beat in the next song
        # nJumps: the number of jumps between the songs before the remix ends
        # probJump: the denominator of the likelihood of jumping to other song (1 / probJump)

        self.jumpSequences = ()
        nJumps = 0
        self.jumpSequences = self.jumpSequences + (self.beatsOne[0],'songOne')

        print("Building random jumps between similar segments in both songs...")

        while nJumps < self.totalJumps:
            if self.jumpSequences[-1] == 'songOne':
                if self.jumpSequences[-2] in self.similarSegmentsBoth[:,0]:
                    randInt = random.randrange(0,self.probJump,1)
                    if randInt < 1:
                        chosenJump = self.similarSegmentsBoth[self.similarSegmentsBoth[:,0]==self.jumpSequences[-2], 1]
                        beatIdx = np.argwhere(self.beatsTwo == chosenJump)[0][0] 
                        nextBeat = self.beatsTwo[(beatIdx+1) % (len(self.beatsTwo))]
                        self.jumpSequences = self.jumpSequences + (nextBeat, 'songTwo')
                        nJumps += 1
                        
                else:
                    beatIdx = np.argwhere(self.beatsOne == self.jumpSequences[-2])[0][0]
                    self.jumpSequences = self.jumpSequences + (self.beatsOne[(beatIdx+1) % (len(self.beatsOne))],
                                                                    'songOne')
            if self.jumpSequences[-1] == 'songTwo':
                if self.jumpSequences[-2] in self.similarSegmentsBoth[:,1]:
                    randInt = random.randrange(0,self.probJump,1)
                    if randInt < 1:
                        chosenJump = self.similarSegmentsBoth[self.similarSegmentsBoth[:,1]==self.jumpSequences[-2], 0]
                        beatIdx = np.argwhere(self.beatsOne == chosenJump)[0][0] 
                        nextBeat = self.beatsOne[(beatIdx+1) % (len(self.beatsOne))] 
                        self.jumpSequences = self.jumpSequences + (nextBeat, 'songOne')
                        nJumps += 1 
                else:
                    beatIdx = np.argwhere(self.beatsTwo == self.jumpSequences[-2])[0][0]
                    self.jumpSequences = self.jumpSequences + (self.beatsTwo[(beatIdx+1) % (len(self.beatsTwo))],
                                                                    'songTwo')

    def genRemix(self):
        # This function will use the sequence of beats to generate the remix with the rawAudio from both songs
        # Will save the remix in the relative path given by outputPath

        jumpSequencesArr = np.array(self.jumpSequences)
        remixSong = ()
        self.beatSamplesOne = librosa.frames_to_samples(self.beatsOne, hop_length = self.hopSize)
        self.beatSamplesTwo = librosa.frames_to_samples(self.beatsTwo, hop_length = self.hopSize)

        for beat in range(0,len(jumpSequencesArr),2):
            beatInt = int(jumpSequencesArr[beat])
            songNum = jumpSequencesArr[beat + 1]
            if songNum == 'songOne':
                beatIdx = np.argwhere(self.beatsOne == beatInt)[0][0] - 1
                if beatIdx != 0:
                    prevBeatIdx = beatIdx - 1
                    segment = self.songOne[self.beatSamplesOne[prevBeatIdx]:self.beatSamplesOne[beatIdx]].flatten()
                    remixSong = remixSong + (segment,)
                else:
                    segment = self.songOne[0:self.beatSamplesOne[0]].flatten()
                    remixSong = remixSong + (segment,)

            if songNum == 'songTwo':
                beatIdx = np.argwhere(self.beatsTwo == beatInt)[0][0] - 1
                if beatIdx != 0:
                    prevBeatIdx = beatIdx - 1
                    segment = self.songTwo[self.beatSamplesTwo[prevBeatIdx]:self.beatSamplesTwo[beatIdx]].flatten()
                    remixSong = remixSong + (segment,)
                else:
                    segment = self.songTwo[0:self.beatSamplesTwo[0]].flatten()
                    remixSong = remixSong + (segment,)

        remixSong = np.concatenate(remixSong)
        remixSongInt16 = np.int16(remixSong * 32767)
        newSong = AudioSegment(data = remixSongInt16.tobytes(),
                                sample_width = 2,
                                frame_rate = self.fsOne,
                                channels = 1)
        newSong.export(self.outputPath, format="mp3")

        print("Exported remix of songs in provided output path!")

    def visualize(self, beatOne, beatTwo):
        # This function will plot the chroma features and DTW of them given the similar beats in
        # self.SimilarSegmentsBoth
        
        # Plotting Chroma
            if beatOne != self.beatsOne[0]:
                prevBeatOne = self.beatsOne[np.argwhere(self.beatsOne == beatOne)[0][0]-1]
                chromaSegmentOne = self.chromaOne[:,prevBeatOne:beatOne]
                beatsOneTime = librosa.frames_to_time([prevBeatOne, beatOne-1],
                                                    sr=self.fsOne, hop_length=self.hopSize)
            else:
                chromaSegmentOne = self.chromaOne[:,0:self.beatsOne[0]]
                beatsOneTime = librosa.frames_to_time([0,beatOne-1],
                                                    sr=self.fsOne, hop_length=self.hopSize)

            if beatTwo != 0:
                prevBeatTwo = self.beatsTwo[np.argwhere(self.beatsTwo == beatTwo)[0][0]-1]
                chromaSegmentTwo = self.chromaTwo[:,prevBeatTwo:beatTwo]
                beatsTwoTime = librosa.frames_to_time([prevBeatTwo, beatTwo-1],
                                                    sr=self.fsOne, hop_length=self.hopSize)
            else:
                chromaSegmentTwo = self.chromaTwo[:,0:self.beatsTwo[0]]
                beatsTwoTime = librosa.frames_to_time([0,beatTwo-1],
                                                    sr=self.fsOne, hop_length=self.hopSize)
            
            timeOne = np.linspace(beatsOneTime[0], beatsOneTime[1], chromaSegmentOne.shape[1])
            timeTwo = np.linspace(beatsTwoTime[0], beatsTwoTime[1], chromaSegmentTwo.shape[1])
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            print(beatsOneTime)

            fig, axs = plt.subplots(2,1, figsize=(12,12))

            grid = axs[0].pcolormesh(timeOne,notes,chromaSegmentOne, shading='nearest', cmap='Greys')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_title('Chromagram for Segment in the First Song')

            grid = axs[1].pcolormesh(timeTwo,notes,chromaSegmentTwo, shading='nearest', cmap='Greys')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_title('Chromagram for Segment in the Second Song')


            # Plotting DTW
            D, wp = librosa.sequence.dtw(X=chromaSegmentOne, Y=chromaSegmentTwo, metric='cosine')

            fig, ax = plt.subplots(figsize=(8,8))

            grid = ax.pcolormesh(timeTwo,timeOne, D, shading='nearest', cmap='Greys')
            ax.plot(timeTwo[wp[:,1]], timeOne[wp[:,0]], marker='o', color='r')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Time (s)')
            ax.set_title('DTW of the Chroma Features for Both Segments')

# %% Executing code
songOnePath = "audio/good.mp3"
songOneType = "mp3"
songTwoPath = "audio/business.mp3"
songTwoType = "mp3"
outputPath =  "audio/remix.mp3"
winDen = 25
hopDen = 2
fMax = 8000
nMels = 256
nSimilarSegments = 700
startThreshold = 0.5
probJump = 750
totalJumps = 10
mixer = MixSongs(songOnePath ,songOneType, songTwoPath, songTwoType, outputPath,
                winDen, hopDen, fMax, nMels,
                nSimilarSegments, startThreshold,
                probJump, totalJumps) 
mixer.calcSegmentsCost()
mixer.evalSegmentsCost()
mixer.genJumpSequences()
mixer.genRemix()
segments = mixer.similarSegmentsBoth[3,:]
mixer.visualize(int(segments[0]), int(segments[1]))