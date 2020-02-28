import imutils
import cv2

def resizeFixedAspect(targetw, targeth = -1, inter = cv2.INTER_AREA):
    """resizeFixedAspect(img, targetw, targeth = -1)

       returns a function that receives an image IMG,
       that resizes it to TARGETW X TARGETH resolution,
       without distorting the initial contents.

       if targeth is not provided, image will be resized
       to a targetw x targetw square
    """
    
    if targeth == -1:
        targeth = targetw
        
    def preprocess(img):
        (h, w) = img.shape[:2]

        if w < h:
            img = imutils.resize(img, width = targetw, inter = inter)
            delta = (img.shape[0] - targeth) // 2
            img = img[delta:delta + targeth, :]

        else:
            img = imutils.resize(img, height = targeth, inter = inter)
            delta = (img.shape[1] - targetw) // 2
            img = img[:, delta:delta + targetw]
        
        return img
    
    return preprocess