import gym
from gym import spaces
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2
import os
import sys
import imutils
import re
import pickle
import pyautogui
from PIL import ImageGrab, Image
from difflib import SequenceMatcher
import time
import win32gui, win32con, win32com.client
import pyautogui
from pynput.keyboard import Key, Controller
import traceback

# need to include in script for pytesseract to work
pytesseract.pytesseract.tesseract_cmd = 'D:\\VideoGame_AI\\third_party\\Tesseract-OCR\\tesseract.exe'

# need a buffer of time between key press and release (seconds)
key_releaseBuffer = 0.25

class CustomEnv(gym.Env):
    metadata={'render.modes':['human']}
    
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # bounds for image grab. Note: This is for when the game is fullscreen
        left_x_bound = 430
        top_y_bound = 170
        right_x_bound = 1485
        bottom_y_bound = 900
        self.whole_screen_bounds =(left_x_bound,top_y_bound,right_x_bound,bottom_y_bound)

        # bounds for amount of time left. Necessary for reward
        self.time_left_bounds = (560,170,1000,200)

        # bounds for level completion message
        self.lvlComplete_msgBounds = (845,290,1070,310)

        # threshold for end level message detection
        self.lvlComplete_msgThresh = 0.80

        # shape of the observation
        self.observation_height = 530
        self.observation_width = 370
        self.observation_channels = 3
        
        
        # agent can go left,right, space
        self.action_space = spaces.Discrete(3)
        
        # input will be an image. for now we will say the shape is (60,60,3)
        self.observation_space = spaces.Box(low=0, high=255,
                                           shape=(self.observation_height,
                                                  self.observation_width,
                                                  self.observation_channels), 
                                            dtype=np.uint8)
        
    def reset(self):
        # press K to kill agent and reset
        keyboard.press('k')
        time.sleep(key_releaseBuffer)
        keyboard.release('k')
        
        # press Z to continue
        keyboard.press(action_lookup[2])
        time.sleep(key_releaseBuffer)
        keyboard.release(action_lookup[2])
        
        # press Z to start
        keyboard.press(action_lookup[2])
        time.sleep(key_releaseBuffer)
        keyboard.release(action_lookup[2])
        
        # capture the screen on restart
        observation, lvl_initialTime = self.screen_capture()
        
        # save so we can use for the reward function
        self.lvl_initialTime = lvl_initialTime
        
        self.prev_stepTime = self.lvl_initialTime
        
        return observation
    
    def step(self, action):
        action_key = action_lookup[action]
        
        # key press
        keyboard.press(action_key)
        time.sleep(key_releaseBuffer)
        keyboard.release(action_key)
        
        # capture updated screen
        observation, lvl_time = self.screen_capture()
        
        # determine if we are done. Capture end of level message and see how close
        self.end_level = self.detect_end_levelMessage()
        
        done = bool(self.end_level != 0)

        # Two rewards here:
        # One for completing the level
        lvl_completeReward = self.end_level
        
        # Another for the amount of time it took to complete the level (ratio)
        # If the agent has obtained a yellow square, which increases its current level time
        # then set the time reward equal to the ratio of current time to intial level time
        if lvl_time > self.prev_stepTime:
            time_rewardRatio = lvl_time/self.lvl_initialTime
        else:
            time_rewardRatio = 0
        
        self.prev_stepTime = lvl_time
        
        reward = time_rewardRatio+lvl_completeReward
        
        info = {}
        
        return observation, reward, done, info
    
    def screen_capture(self):
        
        # capture the screen on restart
        screen = np.array(ImageGrab.grab(bbox=self.whole_screen_bounds))
        
        # resize the screen
        observation = cv2.resize(screen,(self.observation_height,self.observation_width))
        
        # Use the time rectangle to get the intial amount of time 
        # the agent has to complete the level 
        time_capture = np.array(ImageGrab.grab(bbox=self.time_left_bounds))
        
        time_captureGS = cv2.cvtColor(time_capture,cv2.COLOR_BGR2GRAY)
        
        gray = cv2.bitwise_not(time_captureGS)
        time_captureThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, -2)

        contours, hierarchy = cv2.findContours(time_captureThresh, 
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = np.squeeze(contours)

        # use distance equation to get length of rectangle
        # find the maximum distance of all the points in the contours
        try:
            distances = []
            old_points = []
            for i,p1 in enumerate(contours):
                old_points.append(i) #prevent redudancy
                for j,p2 in enumerate(contours):
                    if i != j and j not in old_points:
                        dist = np.linalg.norm(p1-p2,axis=-1)
                        distances.append((i,j,dist))
            time_rect_length = max(distances, key=lambda item:item[2])[2]
        except:
            time_rect_length = self.prev_stepTime

        # convert extracted time to float
        lvl_time = round(time_rect_length,2)
        
        return observation, lvl_time
    
    def detect_end_levelMessage(self):
        
        # determine if the end level message is present
        message_capture = np.array(ImageGrab.grab(bbox=self.lvlComplete_msgBounds))
        
        message_gs = cv2.cvtColor(message_capture,cv2.COLOR_BGR2GRAY)
        
        extracted_text = pytesseract.image_to_string(Image.fromarray(message_gs),
                            config = '--psm 7',
                            lang='eng')
        
        message_text = extracted_text.rstrip()
        
        # determine how similar the text is to the true message
        levelComplete_message = 'level complete! press JUMP to continue'
        
        # get similarity between level complete text and extracted text
        lvlComplete_similarity = round(SequenceMatcher(None,
            message_text,levelComplete_message).ratio(),2)
        
        # do same for end of game message
        lvlFail_message = 'ouch... press JUMP to continue.'

        lvlFail_similarity = round(SequenceMatcher(None,
            message_text,lvlFail_message).ratio(),2)

        if (lvlComplete_similarity > lvlFail_similarity) and (lvlComplete_similarity >= self.lvlComplete_msgThresh) :
            end_level = 1
        elif (lvlFail_similarity > lvlComplete_similarity) and (lvlFail_similarity >= self.lvlComplete_msgThresh) :
            end_level = -1
        else:
            end_level = 0

        return end_level
    
    def close(self):
        hwnd = win32gui.FindWindow(None,'Adobe Flash Player 11')

        # close the game window
        win32gui.PostMessage(hwnd,win32con.WM_CLOSE,0,0)

    
    
# action dictionary. Note: 0x4B is mapped onto the 'K' key
action_lookup = {
    0 : Key.left,
    1 : Key.right,
    2 : 'z',
}     


if __name__ == '__main__':
    
    # start the game
    os.startfile('D:\\Nv2-PC.exe')
    first_game_start = 1
    
    time.sleep(1) # wait for game to load
    keyboard = Controller()
    
    # switch focus to the game window so actions register
    hwnd = win32gui.FindWindow(None,'Adobe Flash Player 11')

    # do this to avoid errors
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')

    # set focus on window
    win32gui.SetForegroundWindow(hwnd)

    # maximize window
    tup = win32gui.GetWindowPlacement(hwnd)
    if tup[1] != win32con.SW_SHOWMAXIMIZED:
        win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

    
    if first_game_start:
        # click on play game and first level
        play_game_xy = (1244,427)

        select_lvl_xy = (549,323)

        pyautogui.click(x = play_game_xy[0], y = play_game_xy[1])
        pyautogui.click(x = select_lvl_xy[0], y = select_lvl_xy[1])

        first_game_start = 0
    
    
    env = CustomEnv()
    obs = env.reset()
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs,rew,done,info = env.step(action)
        time.sleep(1.03) # wait enough time for key release to register action

    print(rew)
    