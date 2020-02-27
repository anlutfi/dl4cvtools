"""dl4cvtools.preprocessors

   collection of function wrappers
   that return functions that receive a single image, perform some processing,
   and return the result
"""

# simple preprocessor wrapper for image resizing
from .simplepreprocessor import *

# preprocessor wrapper for image resizing without distoting
from .resizefixedaspect import *

# preprocessor wrapper for image resizing keeping aspect ratio and cropping
from .aspectawarepreprocessor import *

# wrapper for corverting an image to an array
from .imagetoarray import *

# importing data augmentation tools
from .data_augmentation import *


