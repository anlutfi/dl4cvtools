import imutils
import cv2

def resizeFixedAspect(targetw, targeth = -1):
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

        if h > w:
            img = imutils.resize(img, width = targetw)

        else:
            img = imutils.resize(img, height = targeth)

        padh = int( (targeth - h) / 2 )
        padw = int( (targetw - w) / 2 )
        
        img = cv2.copyMakeBorder(img,
                                 padh,
                                 padh,
                                 padw,
                                 padw,
                                 cv2.BORDER_REPLICATE
                                )

        img = cv2.resize( img, (targetw, targeth) )

        return img
    
    return preprocess