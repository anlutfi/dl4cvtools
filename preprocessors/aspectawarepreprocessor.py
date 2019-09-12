import imutils
import cv2

def aspectAwarePreprocessor(targetw,
                            targeth = -1,
                            interpolation = cv2.INTER_AREA
                           ):

    if targeth == -1:
        targeth = targetw

    def preprocess(img):
        (h, w) = img.shape[:2]

        deltaw = deltah = 0

        if w < h:
            img = imutils.resize(img,
                                 width = targetw,
                                 inter = interpolation
                                )
            deltah = (img.shape[0] - targeth) // 2

        else:
            img = imutils.resize(img,
                                 height = targeth,
                                 inter = interpolation
                                )
            deltaw = (img.shape[1] - targetw) // 2

        (h, w) = img.shape[:2]
        img = img[deltah:h - deltah, deltaw:w - deltaw]

        return cv2.resize(img,
                          (targetw, targeth),
                          interpolation = interpolation
                         )

    return preprocess
