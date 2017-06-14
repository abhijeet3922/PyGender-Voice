# PyGender-Voice
It uses popular ML technique GMM to train gender detector models. 

Data-set:

Training corpus : It has been developed from YouTube videos and consists of 5 minutes of speech for each gender, spoken by 5 distinct male and 5 female speakers (i.e, 1 minute/speaker).

Test corpus: It has been extracted from "AudioSet", a large scale manually annotated corpus recently released by Google this year. The subset constructed from it contains 558 female only speech utterances and 546 male only speech utterances. All audio files are of 10 seconds duration and are sampled at 16000 Hz.

The documentation/tutorial for this repository can be read from this blog.

https://appliedmachinelearning.wordpress.com/2017/06/14/voice-gender-detection-using-gmms-a-python-primer/


# Installation

You need to install only these (tested with):
1. Install Anaconda 64 bit Python 2.7 version. (https://www.continuum.io/downloads) 
2. pip install python_speech_features. (for extracting MFCC features)

Also, Download the data-set from the provided link in the beginning of blog.

Note : Directory path used for train and test corpus in code train_models.py and test_gender.py needs to be properly set depending upon the path where you download the data-set.

