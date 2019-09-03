import numpy as np
import cv2
import os

def simpleDatasetLoader(imgpaths, verbose = 0, preprocessors = []):
    """simpleDatasetLoader(imgpaths, verbose = 0, preprocessors = [])

        returns a tuple (data, labels), both np.array, containing datapoints
        and their respective labels.

        imgpaths is the directory in which the folders of each class(label) are,
                 each containing their images

        verbose is the frequency of progress reporting during loading.
                at each verbose images, a status message will be displayed.
                0 means silent

        preprocessors is a list of functions, each receives an image
                      as a single parameter and returns a processed image
                      for each image loaded from the dataset,
                      all the functions in preprocessors will be applied,
                      in order.
                      Examples of preprocessors are at dl4cvtool.preprocessors
    """

    data = []
    labels = []
    
    for (i, imgpath) in enumerate(imgpaths):
        # assuming dataset/{label}/{image}
        img = cv2.imread(imgpath)
        label = imgpath.split(os.path.sep)[-2]
        
        for preprocessor in preprocessors:
            img = preprocessor(img)
            
        data.append(img)
        labels.append(label)
        
        if verbose > 0 and i % verbose == 0:
            print(  "[INFO] processed {}/{}".format( i, len(imgpaths) )  )
            
    return ( np.array(data), np.array(labels) )

        
