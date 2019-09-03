"""dl4cvtools.preprocessors

   collection of function wrappers
   that return functions that receive a single image, perform some processing,
   and return the result
"""

# simple preprocessor wrapper for image resizing
from .simplepreprocessor import *

# wrapper for corverting an image to an array
from .imagetoarray import *
