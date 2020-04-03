import retro
import gym
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,TensorBoard,TerminateOnNaN
from keras.layers.advanced_activations import LeakyReLU
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import runai.ga 
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras
import re
import itertools

env = retro.make('DoubleDunk-Atari2600')


"""Test trained model on double dunk
Model keeps predicting doing the same action over and over"""

act_combN = 2**env.action_space.n
all_actComb = list(map(list, itertools.product([0, 1], repeat=env.action_space.n)))

action_dict = dict(zip(range(act_combN),all_actComb))

#Load model
modelDir = 'D:\\VideoGame_AI\\OpenAI_Tutorials\\models\\'
past_models = os.listdir(modelDir)
pastMod_avgReward = []
agent = []
if past_models:
    for mod in past_models:
        start = mod.find('max')+4
        end = mod.find('avg')
        temp = re.findall(r"[-+]?\d*\.\d+|\d+", mod[start:end]) 
        avgRew  = list(map(float, temp)) 
        pastMod_avgReward.append(avgRew[0])

    best_model = past_models[np.argmax(pastMod_avgReward)]

agent = keras.models.load_model(modelDir+best_model, compile = False)

print(f"\nLoading Model: {best_model}\n")

testing_episodes = 2
show_game = True



for iEp in range(testing_episodes+1):
	observation_raw = env.reset()

	#resize observations
	observation = cv2.resize(observation_raw,(agent.input_shape[1], agent.input_shape[2]))
	
	#add dimension for prediction
	observation = np.expand_dims(observation,0)

	step = 1
	done = False
	while not done:
		if show_game:
			env.render()

		raw_action = agent.predict(observation/255)

		#clip actions between 0 and 1
		action_idx = np.argmax(raw_action)

		#turn action into int array
		action = np.asarray(action_dict[action_idx], dtype = 'int8')

		raw_newState, reward, done, info = env.step(action)

		step += 1


		#if the agent hasn't moved in 10 steps do a random action
		if step > 2 and np.array_equal(prev_action, action):
			raw_newState, reward, done, info = env.step(env.action_space.sample())
			#print('Doing Random Action')
			step = 1

		#same preprocessing needed as intitially did for first observation
		new_state = cv2.resize(raw_newState, (agent.input_shape[1], agent.input_shape[2]))

		new_state = np.expand_dims(new_state,0)

		observation = new_state

		prev_action = action


env.close()

