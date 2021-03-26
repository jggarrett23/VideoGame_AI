
def call_imports():
    import tensorflow as tf
    import keras
    import numpy as np
    import cv2
    import os
    from random import shuffle
    import urllib.request
    import imutils
    from tqdm import tqdm
    import pickle
    import sklearn as sk
    import matplotlib.pyplot as plt
    import re
    from datetime import datetime
    from keras import backend as K

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.optimizers import SGD
    from keras.layers.normalization import BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam
    from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
    from keras.layers.advanced_activations import LeakyReLU 
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard,TerminateOnNaN
    
if __name__ == "__main__":
    call_imports()
