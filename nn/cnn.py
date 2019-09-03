from keras.models import Sequential

import sys
sys.path.append("/home/antonio/Dropbox/learning/dl4cv")

from dl4cvtools.nn import parseCNNstr

def sequentialCNN(cnnstr, classes = -1):
    model = Sequential()
    parseCNNstr(cnnstr, model)

    return model     
