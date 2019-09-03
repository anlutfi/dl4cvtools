# __init__.py
"""dl4cvtools.dataloaders
   
   functions that load datasets to memory
"""


# as the name suggests, load an entire dataset to memory, without any management
# not suitable for large sets.
from .simpleimageloader import *
