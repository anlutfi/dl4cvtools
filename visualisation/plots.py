import matplotlib.pyplot as plt
import numpy as np
import os
import json
from time import sleep
import signal

def showPlot():
    plt.show()


def scatterPlot(xs,
                ys,
                plottitle = "plot",
                marker = "o",
                c = None,
                s = 30,
                outfname = ''
               ):
    plt.style.use("ggplot")
    plt.figure()
    plt.title(plottitle)
    plt.scatter(xs, ys, marker = marker, c = c, s = s)

    if outfname != '':
        plt.savefig(outfname)


def xOverYPlot(xs,
               ys,
               plottitle = "plot",
               xlabel = "x",
               ylabel = "y",
               outfname = ''
              ):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(xs, ys)
    plt.title(plottitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if outfname != '':
        plt.savefig(outfname)


def modelLossOverTimePlot(h, title, outfname = ''):
    """modelLossOverTimePlot(h, title, outfname = '')

       plots loss and accuracy over epochs of a model training.

       h is the dictionary containing progress data
       title is the plot's title
       outfname is the path to save the plot as a png image
    """
    plt.style.use("ggplot")
    plt.figure()
    for key in h.keys():
        plt.plot(np.arange(0, len(h[key])),
                 h[key],
                 label = key
                )

    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    if outfname != '':
        if not os.path.exists(outfname.rpartition('/')[0]):
            os.makedirs(outfname.rpartition('/')[0])
        plt.savefig(outfname)
        plt.close()

    else:
        plt.show(block = False)


def plotFromJson(d = None, jsonpath = '', title = '', outfname = ''):
    """plotFromJson(d = None, jsonpath = '', title = '', outfname = '')

       plots a line graph, with each list entry in dictionary d as one line

       d is the dictionary to be plotted
       jsonpath is a pathe to a json file to be loaded, if d is not provided
       title is the title of the plot
       outfname is the path to save the plot as a png image
    """

    if d is None and jsonpath != '':
        try:
            with open(jsonpath, 'rb') as f:
                d = json.loads(f.read())
        except FileNotFoundError:
            return
    
    plt.style.use("ggplot")
    plt.figure()
    
    for key in d.keys():
        plt.plot(np.arange(0, len(d[key])),
                 d[key],
                 label = key
                )
    
    plt.title(title if title != '' else jsonpath)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    plt.title(title if title != '' else jsonpath)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    plt.show(block = False)  
    
    if outfname != '':
        if not os.path.exists(outfname.rpartition('/')[0]):
            os.makedirs(outfname.rpartition('/')[0])
        plt.savefig(outfname)
        plt.close()
                  


def jsonLivePlot(jsonpath,
                 lock,
                 title = '',
                 outfname = '',
                 refreshrate = 15
                ):
    """jsonLivePlot(jsonpath,
                 lock,
                 title = '',
                 outfname = '',
                 refreshrate = 15
                )
       
       monitors a json file containing a dictionary and plots its contents live,
       every time it detects new information.
       THIS HAS TO BE CALLED BY A NEW PROCESS

       jsonpath is the path of the json file to be plotted

       lock is a multiprocessing.Lock, to prevent racing problems
            when accessing the file at jsonpath

       title is the title of the plot
       
       outfname is the path to save the plot as a png image

       refreshrate is the period, in seconds, that jsonLivePlot will check
                   for changes on the json file
    """

    # define behaviour for when SIGQUIT or SIGINT are received
    def sighandler(sig, stack):
        lock.acquire()
        with open(jsonpath, 'rb') as f:
            buff = f.read()
        try:
            while True:
                lock.release()
        except AssertionError:
            pass
        except ValueError:
            pass
            
        plot(json.loads(buff))
        if outfname != '':
            if not os.path.exists(outfname.rpartition('/')[0]):
                os.makedirs(outfname.rpartition('/')[0])
            plt.savefig(outfname)
        
        plt.close('all')
        
        os.kill(os.getpid(), signal.SIGKILL)
        sleep(1000)

    signal.signal(signal.SIGQUIT, sighandler)
    
    # internal plotting function
    def plot(h):
        plt.clf()
        for key in h.keys():
            plt.plot(np.arange(0, len(h[key])),
                     h[key],
                     label = key
                    )
        plt.title(title if title != '' else jsonpath)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show(block = False)
        plt.pause(0.1)

    plt.style.use("ggplot")
    plt.ion()
    plt.figure()
    
    currstr = ''
    buff = ''
    
    try:
        # for every represhrate seconds, load and plot file
        while True:
            lock.acquire()
            try:
                with open(jsonpath, 'rb') as f:
                    buff = f.read()
                lock.release()
                    
                if currstr != buff:
                    plot(json.loads(buff))
                    currstr = buff
            except FileNotFoundError:
                lock.release()
                pass

            sleep(refreshrate)
    except KeyboardInterrupt:
        sighandler(signal.SIGINT, 0)
