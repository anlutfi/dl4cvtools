
import sys 
sys.path.append("/home/antonio/Dropbox/learning/dl4cv")

from dl4cvtools.nn import sequentialCNN

def buildLeNet(width, height, depth, classes, waitinput = True):
    cnnstr = ("conv2d{20; (5, 5); "
              + str(width) + "; "
              + str(height) + "; "
              + str(depth)
              + "} > relu > pool2d{(2, 2); (2, 2)} > conv2d{50; (5, 5)} > relu > pool2d{(2, 2); (2, 2)} > flatten > fc{500} > relu > fc{"
              + str(classes)
              + "} > softmax"
             )

    model = sequentialCNN(cnnstr)
    if waitinput:
        _ = input("Press a key to continue: ")
    return model
              
              
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
                                                  
