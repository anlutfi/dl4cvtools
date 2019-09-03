from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

from keras import backend

import sys
sys.path.append("/home/antonio/Dropbox/learning/dl4cv")

from dl4cvtools.visualisation import simpleReport

def addFC(model, classes):
    model.add( Flatten() )
    model.add( Dense(classes) )
    
def addConv2d(model,
              k,
              ksize,
              width = None,
              height = None,
              depth = None,
              padding = "same"
             ):
    
    ksize = ksize.replace('(', '').replace(')', '').split(',')
    
    input_shape = (None
                   if width is None
                   else ( ( int(depth), int(height), int(width) )
                          if backend.image_data_format() == "channels_first"
                          else ( int(height), int(width), int(depth) )
                        )
                  )
        
    if input_shape is None:
        model.add(Conv2D(int(k),
                         ( int(ksize[0]), int(ksize[1]) ),
                         padding = padding
                        )
                 )
    else:
        model.add(Conv2D(int(k),
                         ( int(ksize[0]), int(ksize[1]) ),
                         padding = padding,
                         input_shape = input_shape
                        )
                 )

def addPooling2d(model, pool_size, strides):
    pool_size = pool_size.replace('(', '').replace(')', '').split(',')
    pool_size = ( int(pool_size[0]), int(pool_size[1]) )
    strides = strides.replace('(', '').replace(')', '').split(',')
    strides = ( int(strides[0]), int(strides[1]) )
    model.add( MaxPooling2D(pool_size = pool_size, strides = strides) )

layerfs = {"conv2d": (lambda model,
                             k,
                             ksize,
                             width = None,
                             height = None,
                             depth = None,
                             padding = "same": addConv2d(model,
                                                         k,
                                                         ksize,
                                                         width,
                                                         height,
                                                         depth,
                                                         padding
                                                        )
                     ),
           "bn": lambda model, axis: model.add(BatchNormalization(axis = int(axis[0]))),
           "do": lambda model, rate: model.add(Dropout(int(rate[0]))),
           "flatten": lambda model: model.add(Flatten()),
           "fc": lambda model, classes: model.add( Dense(int(classes)) ),
           "pool": lambda x: x,
           "pool2d": lambda model, pool_size, strides: addPooling2d(model, pool_size, strides),
           "act": lambda model, activator: model.add( Activation(activator) ),
           "relu": lambda model: model.add( Activation("relu") ),
           "softmax": lambda model: model.add( Activation("softmax") )
          }
             
sequentials = ['>', '=>', '->']
blanks = [' ', '\t']
multiplier = '*'
enclosure = ('[', ']')
argseps = ('{', '}', ';')           

def parseCNNstr(cnnstr, model, report = simpleReport):
    cnnstr = cnnstr.lower()
    for seq in sequentials[1:]:
        cnnstr = cnnstr.replace(seq, sequentials[0])
        
    for blank in blanks:
        cnnstr = cnnstr.replace(blank, '')
        
    i = cnnstr.find(multiplier)
    while i != -1:
        partial = cnnstr[:i].rpartition(enclosure[0])[2].replace(enclosure[1], '')
        nextseq = i + cnnstr[i:].find(sequentials[0])
        try:
            factor = int(cnnstr[i + 1:].partition(sequentials[0])[0])
            partial = (sequentials[0] + partial) * (factor - 1)
        except:
            raise Exception("invalid CNN architecture string, non-integer multiplier")
            
        cnnstr = (cnnstr[:i].replace(enclosure[0], '').replace(enclosure[1], '')
                  + partial
                  + cnnstr[nextseq:]
                 )
        i = cnnstr.find(multiplier)
        
    layers = cnnstr.split(sequentials[0])
    
    for layer in layers:
        (layertype, _, _) = layer.partition(argseps[0])
        if layertype not in list(layerfs.keys()):
            raise Exception( "invalid CNN architecture string, non-existent layer {}".format(layertype) )
            
    for layer in layers:
        (layertype, _, args) = layer.partition(argseps[0])
        argsl = args.partition(argseps[1])[0].split(argseps[2])
        args = tuple(argsl) if argsl != [''] else ()
        print( "{}  ---  {}  ---  {}".format(layertype, argsl, args) )
        layerfs[layertype](model, *args)
        
    report(model)
    return
            
            
            
            
            
            
            
            
            
            
    
