import os
import numpy as np
import librosa
import scipy.io.wavfile as wav
from sklearn.svm import SVC

def getMfcc(filename):
    y, sr = librosa.load(filename)
    return librosa.feature.mfcc(y=y, sr=sr)

ACTORS = 24
TRAIN_RATIO = 0.7

x_train = []
y_train = []
x_test = []
y_test = []

for actor in range(1, ACTORS + 1):
    path = "archive/Actor_{:02}".format(actor)
    print(path)
    for pathname, dirnames, filenames in os.walk(path):
        cnt = 0
        num = len(filenames)
        for filename in filenames:
            if filename.endswith('.wav'):
                mfcc = getMfcc(os.path.join(pathname, filename))
                label = np.full((mfcc.shape[1], ), actor, dtype=np.int)
                if cnt < num * TRAIN_RATIO: # train
                    x_train.append(mfcc.T)
                    y_train.append(label)
                else: # test
                    x_test.append(mfcc.T)
                    y_test.append(actor)
                cnt += 1

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

clf = SVC(C=1, gamma=1e-4)
clf.fit(x_train, y_train)
print('Learning Done.')

correct = 0
test_num = len(x_test)
for i in range(test_num):
    predict = clf.predict(x_test[i])
    pre_actor = np.argmax(np.bincount(predict))
    if pre_actor == y_test[i]:
        correct += 1

acc = correct / test_num
print(acc)