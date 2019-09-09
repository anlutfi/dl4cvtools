from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

import sys 
sys.path.append("/home/antonio/Dropbox/learning/dl4cv")

from dl4cvtools.nn import sequentialCNN

def buildLeNet(width, height, depth, classes, waitinput = True):
    cnnstr = ("conv2d{20; (5, 5); "
              + str(width) + "; "
              + str(height)+ "; "
              + str(depth)
              + "} > relu > pool2d{(2, 2); (2, 2)} > conv2d{50; (5, 5)} > relu > pool2d{(2, 2); (2, 2)} > flatten > fc{500} > relu > fc{"
              + str(classes)
              + "} > softmax"
             )

    model = sequentialCNN(cnnstr)
    if waitinput:
        _ = input("Press a key to continue: ")
    return model

model = buildLeNet(28, 28, 1, 10)

print("[INFO] accessing MNIST...")
dataset = datasets.fetch_openml("mnist_784")
data = dataset.data

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)

else:
    data = data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY) = train_test_split(data / 255,
                                                  dataset.target.astype("int"),
                                                  test_size = 0.25,
                                                  random_state = 42
                                                 )
                                                  
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)                                                  
                                                  
print("[INFO] compiling model...")
model.compile(loss = "categorical_crossentropy",
              optimizer = SGD(lr = 0.01),
              metrics = ["accuracy"]
             )
             
print("[INFO] training network...")
H = model.fit(trainX,
              trainY,
              validation_data = (testX, testY),
              batch_size = 128,
              epochs = 20,
              verbose = 1
             )
             
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size = 128)
print(classification_report(testY.argmax(axis = 1),
                            predictions.argmax(axis = 1),
                            target_names = [str(x) for x in le.classes_]
                           )
     )
     
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
              
              
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
