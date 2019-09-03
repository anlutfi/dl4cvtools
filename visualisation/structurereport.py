from colorama import Fore as f
from keras.utils import plot_model
import os

def simpleReport(model):
    """simpleReport(model)

       display, on screen, the layers of the model,
       each with their more relevant attributes
    """
    
    pllt = {'header': f.LIGHTGREEN_EX,
            'modelname': f.LIGHTCYAN_EX,
            'layertitle': f.YELLOW,
            'reset': f.RESET
           }
    
    print(pllt['header']
          + "\n\n******************** MODEL DESCRIPTION *******************"
          + pllt['modelname']
          + "\n\n    Model name: "
          + model.get_config()['name']
          + pllt['reset']
         )
    
    putfield = lambda s: print('    ' + s)
    
    layers = model.get_config()['layers']
    for i in range(len(layers)):
        layerclass = layers[i]['class_name']
        config = layers[i]['config']
        layername = config['name']
        
        print(pllt['layertitle']
              + "\nLayer {}: {} --> {}".format(i,
                                               layerclass,
                                               layername
                                              )
              + pllt['reset']                                        
             )
        
        layerclass = layerclass.lower()
        if 'conv' in layerclass:
            try:
                putfield( "Input shape: {}".format(config['batch_input_shape']) )
            except KeyError:
                pass
                
            putfield( "Filters: {}".format(config['filters']) )
            putfield( "Kernel size: {}".format(config['kernel_size']) )
            putfield( "Strides: {}".format(config['strides']) )
            
        elif 'activation' in layerclass:
            putfield( "Activation: {}".format(config['activation'].upper()) )
            
        elif 'pooling' in layerclass:
            putfield( "Pool size: {}".format(config['pool_size']) )
            putfield( "Strides: {}".format(config['strides']) )
            
        elif 'dense' in layerclass:
            putfield( "units: {}".format(config['units']) )

        elif 'batchnormalization' in layerclass:
            putfield( "Axis: {}".format(config['axis']) ) 
            putfield( "Momentum: {}".format(config['momentum']) ) 
            putfield( "Epsilon: {}".format(config['epsilon']) )

        elif 'dropout' in layerclass.lower():
            putfield( "Rate: {}".format(config['rate']) )

        else:
            putfield("LAYER TYPE NOT CONTEMPLATED BY simpleReport, REFACTOR TO INCLUDE")         
        
                
            
    print(pllt['header']
          + "\n\n******************************************************\n\n\n\n"
          + pllt['reset']
         )



def plotModelLayers(model, outfname, makepath = True):
    """plotModelLayers(model, outfname)

       uses keras plotting function to plot a diagram
       of MODEL's layers and save it as OUTFNAME

       if outfname is in a non-existing directory
       and MAKEPATH is set to true, the path will be created
    """

    if not os.path.exists(outfname.rpartition('/')[0]):
        if makepath:
            os.makedirs(outfname.rpartition('/')[0])
        
        else:
            print("Non-existing directory {} set makepath = True to create it.".format(outfname)
                 )
            return

    plot_model(model, to_file = outfname, show_shapes = True)    

