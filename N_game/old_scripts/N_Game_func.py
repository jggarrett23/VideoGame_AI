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
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard,TerminateOnNaN
from keras.applications.vgg16 import VGG16
from keras.models import Model

def original_labels(img_path):
    original_labels_list = []
    for file_type in [img_path]:
        for img in os.listdir(file_type+'\\originals'):
            original_labels_list.append(img.split('.')[0])
    
    #remove repeated labels
    merged_labels = original_labels_list
    for i,label in enumerate(original_labels_list):
         merged_labels[i] = re.sub('[0-9]+', '',label) #remove number from label
    merged_labels = list(set(merged_labels))
    merged_labels.sort()
    
    return original_labels_list,merged_labels

"""Convolution Neural Networks require a binary array the 
length of the number of images needed to be classified
as a label, with 1 corresponding to image of interest"""
def label_img(image,labels_list):

    #merge repeated labels
    new_labels = labels_list
    for i,label in enumerate(labels_list):
        new_labels[i] = re.sub('[0-9]+', '',label) #remove number from label
    new_labels = list(set(new_labels))
    new_labels.sort()

    for output,label in enumerate(new_labels):
        word_label = image.split('.')[0]

        if label in word_label:
            label_array= keras.utils.to_categorical(output,num_classes=len(new_labels)) #populates image position with 1
            return label_array

def create_train_data(train_dir,labels,img_size=60):
    training_data = []
    training_labels = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img,labels)
        path = os.path.join(train_dir,img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size,img_size))
        img = img.reshape([img_size,img_size,3])
        training_data.append(img)
        training_labels.append(label) #use for sentdex tutorial

    training_data,training_labels= sk.utils.shuffle(training_data,training_labels)
    training_data = np.array(training_data)

    training_data = training_data.astype('float32')

    #training_data = training_data.reshape(training_data.shape[0],img_size,img_size,1)
    np.save('train_data.npy', training_data)
    np.save('train_labels.npy',training_labels)
    return training_data,training_labels

def process_test_data(test_dir,labels,img_size=60):
    testing_data = []
    testing_labels = []
    for img in tqdm(os.listdir(test_dir)):
        label = label_img(img,labels)
        path = os.path.join(test_dir,img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img,(img_size,img_size))
        img = img.reshape([img_size,img_size,3])
        testing_data.append(img) #use for sentdex tutorial
        testing_labels.append(label)


    #testing_data,testing_labels= sk.utils.shuffle(testing_data,testing_labels)
    testing_data = np.array(testing_data)

    testing_data = testing_data.astype('float32')

    np.save('test_data.npy', testing_data)
    np.save('test_labels.npy',testing_labels)
    return testing_data,testing_labels

# Returns a convolution neural network model
def create_CNN(input_data,cnn_nodes = 32, full_nodes = 512, output_nodes = 30,
                 hidden_layers = 6,fully_connected = 1,reg=0.0005,learn_rate=1e-4):
    
    #have to always clear keras session whenever running a new model
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    reg = keras.regularizers.l2(reg)
    
    #check dimensions of input data, must correspond to that of one image (3D)
    if np.ndim(input_data) > 3:
        input_data = input_data[1] #removes fourth dimension
    
    model = Sequential()

    #input layer
    model.add(Conv2D(cnn_nodes, (3, 3), kernel_regularizer = reg,
                     input_shape=([input_data.shape[0],input_data.shape[1],input_data.shape[2]])))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    
    multiplier = 2
    for iLayers in range(hidden_layers):
        #hidden layers
        model.add(Conv2D(cnn_nodes*multiplier, (3, 3), kernel_regularizer = reg))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        multiplier += 2 #powers of 2 increasing hidden layer CNN size


    model.add(Flatten())

    # Fully connected layer
    model.add(Dropout(0.25))
    model.add(Dense(full_nodes))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_nodes))

    model.add(Activation('softmax'))
    
    model.compile(optimizer=Adam(lr=learn_rate, decay = 1e-6),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model_name = 'N_game_'+str(cnn_nodes)+'cnnStart_'+str(full_nodes)+'full_'+str(hidden_layers)+'hidden_'+str(len(model.layers))+'TotalLayers.h5'

    return model,model_name


def train_model(model,model_path,checkpoint_dir,num_ep=5,train_data=True,training_dataNorm=None,training_labels=None,
                testing_dataNorm=None,testing_labels=None,use_generator=False,train_generator=None,validation_generator=None,
               gen_trainSamples=1000,gen_valSamples=1000):
    
    model_name = model_path.split('\\')[-1] # get model name from designated save path
    
    logdir = '.\\log\\'+model_name
    
    #If using image generator to augment data
    if use_generator is True:
        model_name = model_name.split('.')[0]+'_Generator.'+model_name.split('.')[1]
        model_path = model_path.split('.')[0]+'_Generator.'+model_path.split('.')[1]
        
    # Establish callbacks for model
    checkpoint_path = checkpoint_dir+model_name[:-3]+"_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    
    cp_callback = ModelCheckpoint(checkpoint_path,monitor='val_acc',save_best_only=True,verbose=0,mode='auto')
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=round(num_ep*.75), verbose=1, 
                                              mode='auto', baseline=None, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=3, min_lr=1e-6)
    nan_term = TerminateOnNaN()

    tensorboard = TensorBoard(log_dir = logdir,histogram_freq=0,write_graph=False,
                                             write_grads=True,update_freq='epoch')

    callback_list = [cp_callback,early_stop,reduce_lr,nan_term,tensorboard]
    
    
    # check if model already exists. If it does, load weights of best epoch and continue training
    epNum = []
    epVal_acc = []
    if os.path.exists('{}'.format(model_path)):
        for cp in os.listdir(checkpoint_dir):
            if model_name[:-3] in cp:
                epNum.append(int(cp[-12:-10]))
                epVal_acc.append(float(cp[-9:-5]))
        if epVal_acc:
            best_ep = epNum[np.argmax(epVal_acc)]
            model.load_weights(checkpoint_path.format(epoch=best_ep,val_acc=max(epVal_acc)))

            num_ep = best_ep+num_ep

            print('Weights loaded!')
        else: 
            print('Weights not found')
            best_ep = 0
            return('Error with code')
    else:
        best_ep = 0 # If model not found will train from scratch
        if train_data is True:
            print('No version of model found, proceeding to train new model from scratch...')
        else: 
            print('No version of model found.')

    if train_data is True:
        if use_generator is True:
            model.fit_generator(train_generator,epochs=num_ep,initial_epoch=best_ep,
                                 steps_per_epoch= gen_trainSamples,
                                 validation_data = validation_generator,
                                 validation_steps = gen_valSamples,
                                 verbose=1,callbacks=callback_list)
        else:
            model.fit(x=training_dataNorm,y=np.array(training_labels),epochs=num_ep,verbose=1,shuffle=True,
                       validation_data = (testing_dataNorm,test_labels), initial_epoch=best_ep,
                       callbacks=callback_list)
    model.save(model_path)
    
    return model

def custom_results_plot(model):
    history = model.history.history
    x = model.history.epoch
    plt.style.use("ggplot")
    plt.figure(figsize=(10,10))
    #plt.subplot(1,2,1)
    plt.plot(x,history["loss"],'r',label="train_loss")
    plt.plot(x,history["acc"],'g',label="train_acc")
    plt.plot(x,history["val_loss"],'r--',label="val_loss")
    plt.plot(x,history["val_acc"],'g--',label="val_acc")
    plt.title("Training/Val Loss and Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    
    #plt.subplot(1,2,2)
    #keras.utils.plot_model(kmodel)
    plt.show()
    
def create_ownVGG16(input_data,num_output,reg=0.0005,learn_rate=1e-4):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    reg = keras.regularizers.l2(reg)
    
    #check dimensions of input data, must correspond to that of one image (3D)
    if np.ndim(input_data) > 3:
        input_data = input_data[1] #removes fourth dimension
    
    model = Sequential()
    
    model.add(keras.layers.InputLayer(input_shape=([input_data.shape[0],input_data.shape[1],input_data.shape[2]]),name='input_2'))
    n = 64
    for i in range(1,6):
        model.add(Conv2D(n,(1,1), activation='relu', name='block'+str(i)+'_conv1'))
        model.add(Conv2D(n,(1,1), activation='relu', name='block'+str(i)+'_conv2'))
        model.add(BatchNormalization(axis=-1))
        model.add(MaxPooling2D(pool_size=(2,2), name='block'+str(i)+'_pool'))
        if n == 512:
            pass
        else:
            n = n*2
    model.add(keras.layers.Flatten(name='flatten'))
    #model.add(Dropout(0.25))
    model.add(keras.layers.Dense(4096,name='fc1'))
    model.add(keras.layers.Dense(4096,name='fc2'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(keras.layers.Dense(num_output,activation = 'softmax', name='predicitions'))
    
    model.compile(optimizer=Adam(lr=learn_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model_name = 'N_game_VGG.h5'

    return model,model_name

def data_generator(img_path):
    mean = 0
    for img in tqdm(os.listdir(img_path)):
        positive = cv2.imread(img_path+'\\'+str(img))
        for angle in np.arange(5,360,10):
            rotated = imutils.rotate_bound(positive, angle)
            cv2.imwrite(img_path+'_rotated\\'+str(img)[:-4]+'_'+str(angle)+'.JPG',rotated)
            for s in np.arange(20,600,20):
                gaus = cv2.randn(rotated,mean,int(s))
                cv2.imwrite(img_path+'\\originals_rotated\\'+str(img)[:-4]+'_'+str(angle)+'_gauss_sd_'+str(s)+'.JPG',gaus)
                
def preCreated_VGG(input_dim,num_classes,pre_trained=True,include_topLayers=False,freezeLayers=-3):
    
    if pre_trained:
        mod_weights ='imagenet'
    else:
        mod_weights = None
        
    
    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights=mod_weights, include_top=include_topLayers)

    if pre_trained:
        #freeze all but the last n layers of VGG16 
        for layer in model_vgg16_conv.layers[:freezeLayers]:
            layer.trainable = False

    #Create your own input format
    input_tensor = Input(shape=(input_dim),name = 'image_input')

    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input_tensor)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    #Create your own model 
    my_model = Model(input=input_tensor, output=x)
    
    return my_model