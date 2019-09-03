"""dl4cvtools.callbacks

    classes that extend keras.callbacks BaseLogger
    to be used to keep a close look on a model during its training
"""

from .stepdecay import *

# TrainingMonitor monitors the training process in real-time,
# saving performance information in json files and plotting losses and accuracy
from .trainingmonitor import *

# AlertingMonitor monitor's the model's training,
# checking if certain conditions are met and, then, triggering alerts
from .alertingMonitor import AlertingMonitor
