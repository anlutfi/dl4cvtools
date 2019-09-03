from keras.callbacks import BaseLogger
import json
import os
import multiprocessing
from time import sleep
import signal


class AlertingMonitor(BaseLogger):
    """AlertingMonitor

       class that calls alerting functions at given points during training.
       these functions, in turn, evaluate aspects of the model
       and trigger appropriate alerts
    """
    def __init__(self,
                 jsonpath,
                 alertfs = []
                ):
        """__init__(self,
                 jsonpath,
                 alertfs = []
                )
           
           creates an AlertingMonitor

           jsonpath -> path to load a json dictionary containing training
                       and cross-validation losses and accuracies
                       over each epoch

           alertfs -> list containing functions that receive a single argument:
                      the dictionary loaded by a json at jsonpath.
                      each of these functions analyze the dictionary and
                      decide to emit an alert or not
        """

        super(AlertingMonitor, self).__init__()
        self.jsonpath = jsonpath
        self.alertfs = alertfs    
        


    def on_epoch_end(self, epoch, logs = {}):
        """on_epoch_end(self, epoch, logs = {})

           called after each epoch, opens a separate process and calls each
           of the evaluating functions in self.alertfs
        """

        parentpid = os.getpid()
        os.fork()
        
        # if it's the parent process (which is also where model training happens),
        # return
        if os.getpid() == parentpid:
            return
        
        # if it's the new process, 
        # iterate through alert functions in self.alertfs,
        else:
            try:
                with open(self.jsonpath, 'rb') as f:
                    h = json.loads( f.read() )
                
                alertps = []
                for alert in self.alertfs:
                    alertps.append(multiprocessing.Process(target = alert,
                                                           args = (h.copy(),)
                                                          )
                                  )
                    alertps[-1].start()
                
                for process in alertps:
                    process.join()#TODO make it so it kills the processes
            
            except (FileExistsError, FileNotFoundError):
                pass

            os._exit(0)
        
       

