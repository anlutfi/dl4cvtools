"""dl4cvtools.alerts
   Collection of useful functions to trigger a response for specific conditions
   that might have occured in one of the entry-points
   of keras.callbacks' BaseLogger, as its extended by the
   classes in dl4cvtools.callbacks. 

   Functions here receive, as arguments, at least a dictionary with the training history and
   a function to call if the criteria for triggering the alert is met.

   However, it can receive any other number of arguments

   A wrapper is suggested as it follows, where h is the history dictionary:

   wrapper = lambda h: alertFunction(h, arg1, arg2, ...)
"""

from .alerts import *