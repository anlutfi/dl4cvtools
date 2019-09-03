import cv2
import numpy as np

def showImgLabel(label, img = None, imgpath = ''):
    """showImgLabel(label, img = None, imgpath = '')

       displays an image IMG with a text LABEL on the upper left corner

       if img is None, loads image will be loaded from imgpath
    """
    assert (img is not None or imgpath != '')

    if img is None:
        img = cv2.imread(imgpath)

    cv2.putText(img,
                "Label: {}".format(label),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                3
               )
    cv2.imshow("Labeled image", img)