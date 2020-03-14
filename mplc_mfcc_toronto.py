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
np.set_printoptions(threshold=sys.maxsize)

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
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

#DataFlair - Emotions in the TORONTO dataset
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
#DataFlair -  to observe
observed_emotions=['sad', 'angry', 'fearful','happy','neutral']

#DataFlair - Load the data and extract features for each sound file
def load_data(test_size=0.001):
    x, y = [], []
    for file in glob.glob('MLPlibrosa/datatrain_*/*.wav'):
        file_name, ext = os.path.splitext(file)
        file_only = os.path.basename(file_name)
        emotion = emotions[file_only.split("_")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, lpc=False, mel=False)
        feature_str = str(feature)
        x.append(feature)
        y.append(emotion)
        #print(feature)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=0)

#untuk 8:2 -> kfolds 5
#untuk 9:1 -> kflods 10
#7:3 -> 
#untuk 6:4 -> kflods 12

#load_data()
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=load_data()

#DataFlair - Get the shape of the training and testing datasets
#print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
#print(f'Features extracted: {x_train.shape[1]}')

#DataFlair - Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', epsilon=1e-08, hidden_layer_sizes=(10), learning_rate='adaptive', max_iter=1000, verbose=False)
#epsilon : nilai ambang batas kesalahan, aplha = angka pembelajaran/learning rate

#Cross-Validating the results
#kf= KFold(n_splits=4)
#cross_val = model

#DataFlair - Train the model based on 10 folds
#for train_indices, test_indices in kf.split(x_train):
model.fit(x_train,y_train)
filename = 'modelweight.sav'
error = pck.dump(model, open(filename, 'wb'))
if error is None:
  print("Model berhasil disimpan dengan nama ", filename)
   # y_pred=model.predict(x_test)
   # print("Training set score: %f" % model.score(x_train, y_train))
   # print("Test set score: %f" %model.score(x_test, y_test))
   # accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
   # print("Accuracy: {:.2f}%".format(accuracy*100))
   # target_names =['sad', 'angry', 'fearful','happy','neutral']
   # k = 0
   # for i in y_test:
   #   if (y_pred[k] != i):
   #     print("dataset = ", y_pred[k], " hasil uji = ", i)
   #   k += 1   
 #  for k in i:
 #       print("Dataset berkata = ", y_test[i][k]," dan hasilnya adalah ", y_pred[i][k])
 #   print(classification_report(y_pred,y_test, target_names=target_names))
    