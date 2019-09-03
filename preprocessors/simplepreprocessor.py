import cv2

def simplePreprocessor(width, height = -1, interpolation = cv2.INTER_AREA):
    """simplePreprocessor(width, height = -1, interpolation = cv2.INTER_AREA)

       returns a function f(img), that resizes an image img
       to a width X height resolution,
       using the interpolation algorithm provided

       if height is not provided, the new size will be the square width X width
    """

    if height == -1:
        height = width
        
    return lambda img: cv2.resize(img, (width, height), interpolation)
