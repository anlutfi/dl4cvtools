from keras.preprocessing.image import img_to_array

def imageToArrayPreprocessor(dataformat = None):
    """imageToArrayPreprocessor(dataformat = None)

       returns a function f(img), that receives an image img
       and flattens it into an array according to data_format
    """
    return lambda img: img_to_array(img, data_format = dataformat)
