
#Dataset_preprocess
#Dependencies: pandas, numpy, librosa, soundfile



#This is for the raw+spetrogram versions, might need separate for opensmile
#Call the dataset_preprocess() like so : dataset_preprocess emodb 16000
#dataset names are emodb iemocap ravdess and savee - lower case
#Run this script from the root folder containing the four dataset folders in their original strucuture
#Ie .\this_script.py
#     /EmoDB/
#     /SAVEE/
#etc. Slashes need to be changed to forward slashes if running on windows. But the training process requires linux anyway */ 
import sys
import os 
import pandas as pd
import numpy as np
import librosa as lr
import librosa.display
import soundfile as sf
#Arguments 
#python dataset_preproces.py emodb 16000 4 y

#takeargs
which_dataset = sys.argv[1]
SAMPLE_RATE = sys.argv[2]
SAMPLE_DURATION = sys.argv[3]
z_score = sys.argv[4] #Flag for performing z-score normalisation at the data preprocessing stage. This should be y or n 



#Dataset preprocessing 
# Go through dataset and convert sample rate, Trim or pad to a uniform duration, normalise via zero mean and unit variance 
# Create the Mel spectrogram and MFCCs from the duration-adjusted and normalised samples
# Split dataset into training/validation/test 

#To preprocess I iterate through a CSV, The csv's are provided with this script so they do not need to be rebuilt, however they include 
# directory paths and so this is why the original structure of the dataset needs to be preserved
# The CSV needs to be in same root folder as this script 
#Should specify in readme these folders should be for IEMOCAP "IEMOCAP_full_release", etc, to avoid confusion. 


if which_dataset == "emodb":
     DATASET_PATH = "EmoDB/wav"
     dataset_name = "EmoDB" # Simply so the cmdline argument need not be caps-sensitive, but this way it preserves the capitalisation of the original dataset. 
     csv_path = "emodb.csv"        
     
elif which_dataset == "iemocap":
     DATASET_PATH = "IEMOCAP_full_release"
     dataset_name = "IEMOCAP"
     csv_path = "iemocap.csv"
     
elif which_dataset == "ravdess":
     DATASET_PATH = "RAVDESS"
     dataset_name = "RAVDESS"
     csv_path = "ravdess.csv"
     
elif which_dataset == "savee":
     DATASET_PATH = "SAVEE/AudioData"
     dataset_name = "SAVEE"
     csv_path = "savee.csv"
     
     
else:
     print("Incorrect dataset provided, options are: emodb iemocap ravdess savee")


#Setting up
dataset_path = './' + DATASET_PATH
if z_score == 'y':
     out_path = '/' + dataset_name + 'norm_and_fixedduration/'
else: 
     out_path = '/' + dataset_name + '_fixedduration/'

df = pd.read_csv(csv_path)



# Defining the functions for dataset preprocessing
def norm_script():     

     #Z-score normalisation
     def listwavs(dataframe):
          list_wavs = []
          for file in dataframe['file']:
               audio_file_path = os.path.join(dataset_path,file[4:])   
               print("audio file path: ", audio_file_path)
               x,_ = lr.load(audio_file_path, sr=SAMPLE_RATE)
               list_wavs.append(x)
          return list_wavs
     
     #Now we compute the mean and std from the training data in order to fit
     globalaudio = np.concatenate(listwavs(pd.read_csv(csv_path +"train.csv")))
     mean = np.mean(globalaudio)
     std = np.std(globalaudio)
     print("Progress: global values for z-score normalisation calculated")


     #Trim Length EmoDB has 2-3 seconds for most utterances, sometimes 5s or so rarely so so trimming down to 3s should be simple and sufficient
     def trim_wave(wave): #
          duration = SAMPLE_DURATION * sr # So duration in cmdline arguments passed in seconds, is turned into number of samples based on sample rate
          trimmed_wave = wave[0:duration]
          return trimmed_wave



     def pad_wave(wave):
          sr = SAMPLE_RATE 
          duration = SAMPLE_DURATION * sr # So duration in cmdline arguments passed in seconds, is turned into number of samples based on sample rate
          padding = int(duration - len(wave))
          padded_wave = np.pad(wave, (0,padding),'constant')
          return padded_wave
     
     def save_output(wave,filename):
    
          # Write out audio as 24bit PCM WAV
          filename = os.path.join(out_path,filename)
          sf.write(filename, wave, SAMPLE_RATE, subtype='PCM_24')


     #Applying functions

     for file in os.listdir(dataset_path):
          print("file =", file)

     audio_file = os.path.join(dataset_path,file)        
     print("audio file = ", audio_file)
     y,sr = lr.load(audio_file,sr=SAMPLE_RATE)  
     #Normalise via zero mean and 1 unit variance if z-score at this stage is selected
     if z_score == 'y':
          y_norm =  (y - mean) / std
     else: 
          y_norm = y #Use the original sample instead of the normalised
     if lr.get_duration(y=y,sr=sr) > SAMPLE_DURATION:
        trimmed_wave = trim_wave(y_norm)
        save_output(trimmed_wave,file)
        print(file," saved")
     else: 
        padded_wave = pad_wave(y_norm)
        save_output(padded_wave,file)
        print(file," saved")


     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'duration_adjusted'] = os.path.join(out_path, str(file))        
          df.to_csv(which_dataset + "preprocessed.csv")

     print("Progress: Z-score normalisation and fixing of sample duration completed")


     #Creating Mel and MFCCs. 

     #Constants
     FRAME_WIDTH = 512 # increase to 512 now 
     NUM_SPECTROGRAM_BINS = 512 # 512 is recommended for speech (default is 2048 and suited for music)
     NUM_MEL_BINS = 128
     LOWER_EDGE_HERTZ = 80.0 # Human speech is not lower
     UPPER_EDGE_HERTZ = 7600.0 # Higher is inaudbile to humans   
     SAMPLE_RATE = 16000
     N_MFCC = 40

     dataset_path = out_path  #Take the trimmed/padded sound files 
     mel_path = out_path + "mel/"
     mfcc_path = out_path + "mfccs/"

     for file in os.listdir(dataset_path):
          audio_file = os.path.join(dataset_path,file)    
          print(str(audio_file))
          samples, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    
          #Create spectrogram
          sgram = librosa.stft(samples,n_fft=NUM_SPECTROGRAM_BINS)  


          # use the mel-scale instead of raw frequency on
          sgram_mag, _ = librosa.magphase(sgram)
          mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, n_fft= FRAME_WIDTH,
                                                       sr=sample_rate,fmin=LOWER_EDGE_HERTZ,fmax=UPPER_EDGE_HERTZ,
                                                       n_mels = NUM_MEL_BINS)
          librosa.display.specshow(mel_scale_sgram)

          # use the decibel scale to get the final Mel Spectrogram, as the human hear perceives loudness this way
          mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min) 
          #Now we should have an actual mel spectrogram
          np.save(mel_path + str(file)[:-4],mel_sgram)
          mfccs = librosa.feature.mfcc(S=mel_sgram,sr=SAMPLE_RATE,n_mfcc=N_MFCC)
          np.save(mfcc_path + str(file)[:-4],mfccs)      

     #Write path to the mel-spectrograms and mffcs for each utterance to the appropriate row in the CSV
     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'mel_spectrogram'] = os.path.join(mel_path,str(file)[4:-4]+".npy")
          df.loc[index, 'MFCCs'] = os.path.join(mfcc_path, str(file)[4:-4]+".npy")

     df.to_csv(which_dataset + "preprocessed_with_mel_mfcc.csv")

     # Splitting the data

norm_script()