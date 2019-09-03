from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras import backend
    



def buildMiniVGG(width, height, depth, classes):
    if backend.image_data_format() != "channels_first":
        inputShape = (height, width, depth)
        chani = -1
    else:
        inputShape = (depth, height, width)
        chani = 1

    model = Sequential()
    
    model.add(Conv2D(32,
                     (3, 3),
                     padding = "same",
                     input_shape = inputShape)
                    )
    
    model.add( Activation("relu") )
    model.add(BatchNormalization(axis = chani))
    model.add( Conv2D(32, (3, 3), padding = "same") )
    model.add( Activation("relu") )
    model.add( BatchNormalization(axis = chani) )
    model.add(  MaxPooling2D( pool_size = (2, 2) )  )
    model.add( Dropout(0.25) )         
    model.add( Conv2D(64, (3, 3), padding = "same") )
    model.add( Activation("relu") )
    model.add( BatchNormalization(axis = chani) )
    model.add( Conv2D(64, (3, 3), padding = "same") )
    model.add( Activation("relu") )
    model.add( BatchNormalization(axis = chani) )
    model.add(  MaxPooling2D( pool_size = (2, 2) )  )
    model.add( Dropout(0.25) )
    model.add( Flatten() )
    model.add( Dense(512) )
    model.add( Activation("relu") )
    model.add( BatchNormalization() )
    model.add( Dropout(0.5) )
    model.add( Dense(classes) )
    model.add( Activation("softmax") )

    return model