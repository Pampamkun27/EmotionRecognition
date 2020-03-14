import librosa
import soundfile
import os, glob, pickle
import numpy as np
import sys
import pickle as pck
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from librosa.core import istft
import joblib
import argparse
import time

emotions={
  'neutral':'neutral',
  'calm':'calm',
  'happy':'happy',
  'sad':'sad',
  'angry':'angry',
  'fear':'fearful',
  'disgust':'disgust',
  'surprised':'surprised',
  'bored':'bored'
}

observed_emotions=['sad', 'angry', 'fearful','happy','neutral']


def arg_parse():
	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
	parser.add_argument("-file", dest = 'filee', help = 
                        "File atau direktori data yang akan diuji",
                        default = "MLPlibrosa\datatest_*\*.wav", type = str)
	parser.add_argument("-modelname", dest = 'modelname', help = "Model atau bobot hasil training",default = "modelweight.sav", type = str)

	return parser.parse_args()

def extract_feature(file_name, mfcc, lpc, mel):
    X, sample_rate = librosa.load(file_name)
    result = np.array([])
    if mfcc:
        stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
            result = np.hstack((result, mfccs))
        if lpc:
            lpc = (librosa.lpc(X,16))
            result = np.hstack((result, lpc))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

def feature(file):
    feature = extract_feature(file, mfcc=True, lpc=False, mel=False)
    feature_str = str(feature)
    return feature_str

def load_data(filename):
    x, y = [], []
    for file in glob.glob(filename):
        file_name, ext = os.path.splitext(file)
        file_only = os.path.basename(file_name)
        emotion = emotions[file_only.split("_")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, lpc=False, mel=False)
        feature = extract_feature(file, mfcc=True, lpc=False, mel=False)
        feature_str = str(feature)
        x.append(feature)
        y.append(emotion)
        #print(feature)
    return np.array(x), np.array(y)

start = time.time()

args = arg_parse()
datatest = args.filee
x_test, y_test = load_data(datatest)

filename = args.modelname
loaded_model = pck.load(open(filename, 'rb'))
if loaded_model is not None:
	print("Neural Network berhasil dibangun...")
#result = loaded_model.score(X_test, Y_test)
y_pred=loaded_model.predict(x_test)

end = time.time()

i = 0
for file in glob.glob(datatest):
    print("Data yang digunakan ",file, " dan hasilnya ", y_pred[i])
    i+=1
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("jumlah data ", i)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("{:25s}: {:2.3f}".format("Running time...", end - start, " seconds"))