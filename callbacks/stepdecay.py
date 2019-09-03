import numpy as np

def stepDecay(alpha0, factor, stepfrequency):
    return lambda epoch: float(alpha0
                               * (factor
                                  ** np.floor(epoch
                                              / stepfrequency
                                             )
                                 )
                              )