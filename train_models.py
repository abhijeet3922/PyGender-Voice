#train_models.py

import os
import cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features

#path to training data
source   = "D:\\pygender\\train_data\\youtube\\male\\"   
#path to save trained model
dest     = "D:\\pygender\\"         
files    = [os.path.join(source,f) for f in os.listdir(source) if 
             f.endswith('.wav')] 
features = np.asarray(());

for f in files:
    sr,audio = read(f)
    vector   = get_MFCC(sr,audio)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag',
        n_init = 3)
gmm.fit(features)
picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"

# model saved as male.gmm
cPickle.dump(gmm,open(dest + picklefile,'w'))
print 'modeling completed for gender:',picklefile
