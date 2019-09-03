from keras.callbacks import BaseLogger
import json
import os
import multiprocessing
from time import sleep
import signal

import sys
sys.path.append("/home/antonio/Dropbox/learning/dl4cv")

from dl4cvtools.visualisation import jsonLivePlot


class TrainingMonitor(BaseLogger):
    """Class to monitor, save, and plot a model's trining performance

       It logs training and cross-validation losses and accuracies
    """

    def __init__(self,
                 jsonpath,
                 figurepath = '',
                 plottitle = '',
                 resumeat = 0,
                 plotF = jsonLivePlot
                ):
        """def __init__(self,
                 jsonpath,
                 figurepath = '',
                 plottitle = '',
                 resumeat = 0,
                 plotF = jsonLivePlot
                )

           creates a TrainingMonitor

           jsonpath -> path to save a json dictionary containing training
                       and cross-validation losses and accuracies
                       over each epoch
           
           figurepath -> path to save a plot of the same information

           plottitle -> title to be used in the plot window

           resumeat -> starting epoch, for models that have been
                       interrupted mid-training

           plotF -> function to be used for plotting.
                    plotting is done in a separate process,
                    so it needs to implement safe-guards.
                    it must be of the format:
                    plotF(jsonpath, filelock, plottitle, figurepath)
                    jsonpath, plottitle  and figurepath are the same as above
                    
                    filelock is a multiprocessing.Lock(),
                    to prevent racing problems over the json file
        """
        
        super(TrainingMonitor, self).__init__()
        # path to save a plot of the same information
        self.figurepath = figurepath
       
        # starting epoch, for models that have been
        # interrupted mid-training
        self.resumeat = resumeat
       
        # path to save a json dictionary containing training
        # and cross-validation losses and accuracies
        # over each epoch
        self.jsonpath = jsonpath        
        
        #dictionary containing training data
        self.h = {}

        # multiprocessing lock for json file
        self.filelock = multiprocessing.Lock()

        # parallel process to plot training data
        self.plotter = multiprocessing.Process(target = plotF,
                                               args = (self.jsonpath,
                                                       self.filelock,
                                                       plottitle,
                                                       figurepath
                                                      )
                                              )



    def on_train_begin(self, logs = {}):
        """on_train_begin(self, logs = {})
           
           starts parallel plotting process
           and loads training history, in case of a resumed training
        """

        self.plotter.start()
        self.plotterpid = self.plotter.pid
        
        if os.path.exists(self.jsonpath):
            self.h = json.loads( open(self.jsonpath).read() )

            if self.resumeat > 0:
                self.h = [self.h[k][:self.resumeat]
                          for k in self.h.keys()
                         ]

        leaffolder = self.jsonpath.rpartition('/')[0]
        if not os.path.exists(leaffolder):
            os.makedirs(leaffolder) 



    def on_epoch_end(self, epoch, logs = {}):
        """on_epoch_end(self, epoch, logs = {})
           
           logs last epoch performance to a json file
        """

        for (key, value) in logs.items():
            if key not in self.h or self.h[key] is None:
                self.h[key] = []
            self.h[key].append(value)

        with self.filelock:
            f = open(self.jsonpath, 'w')
            f.write(json.dumps(self.h))
            f.close()
        

    def on_train_end(self, logs = {}):
        """on_train_end(self, logs = {})
           
           at the end of training, kills the plotting process.
           the plotter is responsible for saving images to disk.
        """

        os.kill(self.plotterpid, signal.SIGQUIT)
        self.plotter.join()
