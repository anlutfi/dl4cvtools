from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img


import numpy as np

import argparse

from os.path import join
from os.path import sep

def simpleAug(samples,
              img = None,
              imgpath = None,
              outpath = None,
              verbose = True,
              rotation_range = 30,
              width_shift_range = 0.1,
              height_shift_range = 0.1,
              shear_range = 0.2,
              zoom_range = 0.2,
              horizontal_flip = True,
              fill_mode = "nearest",
              save_format = 'jpg'
             ):
    """def simpleAug(img = None,
                  samples,
                  imgpath = None,
                  outpath = None,
                  verbose = True,
                  rotation_range = 30,
                  width_shift_range = 0.1,
                  height_shift_range = 0.1,
                  shear_range = 0.2,
                  zoom_range = 0.2,
                  horizontal_flip = True,
                  fill_mode = "nearest",
                  save_format = 'jpg'
                 ):

        Applies simple transformations to an image for data augmentation. Returns a list of samples transformed versions of the original image

        img is the image. If it's set to None, then it will be loaded from imgpath

        outpath is the filename of the output image, if it's to be saved to disk

        verbose set to True displays the function's progress

        the remaining parameters are the transformation ranges, to be passed on to keras.preprocessing.image.ImageDataGenerator. Refer to its documentation
    """

    if img is None:
        img = load_img(imgpath)
    
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    augger = ImageDataGenerator(rotation_range = rotation_range,
                                width_shift_range = width_shift_range,
                                height_shift_range = height_shift_range,
                                shear_range = shear_range,
                                zoom_range = zoom_range,
                                horizontal_flip = horizontal_flip,
                                fill_mode = fill_mode
                               )

    imagegen = augger.flow(img,
                           batch_size = 1,
                           save_to_dir = outpath,
                           save_prefix = (imgpath.split(sep)[-1].split('.')[0]
                                          if imgpath is not None
                                          else ''
                                         ),
                           save_format = save_format
                          )

    result = []

    for i in range(samples):
        newimg = imagegen.next()
        result.append(newimg)
    
    return result


