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
from PIL import Image
from difflib import SequenceMatcher
import time
import win32gui, win32con, win32com.client
import pyautogui
from pynput.keyboard import Key, Controller

from gym.envs.registration import register

import mss
import mss.tools

# need to include in script for pytesseract to work
pytesseract.pytesseract.tesseract_cmd = 'D:\\VideoGame_AI\\third_party\\Tesseract-OCR\\tesseract.exe'

keyboard = Controller()

# need a buffer of time between key press and release (seconds)
key_releaseBuffer = 0.5

sct = mss.mss()

class N_Game_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(N_Game_Env, self).__init__()

        # bounds for amount of time left. Necessary for reward
        self.time_left_bounds = (162, 145, 850, 170) # OMEN (85, 79, 886, 102)

        # bounds for level completion message
        self.lvlComplete_msgBounds = (363, 240, 595, 260) # OMEN (392, 199, 566, 213)

        # threshold for end level message detection
        self.lvlComplete_msgThresh = 0.80

        # shape of the observation
        self.observation_height = 84
        self.observation_width = 84
        self.observation_channels = 3

        # action dictionary. Note: 0x4B is mapped onto the 'K' key
        self.action_lookup = {
            0: 0x27,  # left arrow
            1: 0x25,  # right arrow
            2: 0x5A,  # z key
            3: (0x27, 0x5A),  # left + z
            4: (0x25, 0x5A),  # right + z
        }

        # agent can go left, right, jump
        self.action_space = spaces.Discrete(len(self.action_lookup))

        # input will be an image. for now we will say the shape is (84,84,3)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_height,
                                                   self.observation_width,
                                                   self.observation_channels),
                                            dtype=np.uint8)
        # for navigating menus
        self.down_arrowcode = 0x28

        self.lvl_initialTime = None

        self.prev_stepTime = None

    def reset(self):

        # press Z to continue
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                             self.action_lookup[2], 0)
        time.sleep(.1)
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                             self.action_lookup[2], 0)

        # press Z to start
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                             self.action_lookup[2], 0)
        time.sleep(.1)
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                             self.action_lookup[2], 0)

        # capture the screen on restart
        observation, lvl_initialTime = self.screen_capture()

        # save so we can use for the reward function
        self.lvl_initialTime = lvl_initialTime

        self.prev_stepTime = self.lvl_initialTime

        # for later versions of gym default behavior is to return obs and info
        info = {}

        return observation, info

    def step(self, action):
        action_key = self.action_lookup[action]

        # key press
        if action < 3:
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                                 action_key, 0)
            time.sleep(key_releaseBuffer)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                                 action_key, 0)
        else:
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                                 action_key[0], 0)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                                 action_key[1], 0)
            time.sleep(key_releaseBuffer)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                                 action_key[0], 0)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                                 action_key[1], 0)

        # capture updated screen
        observation, lvl_time = self.screen_capture()

        # get around the run out of time menu start by just pressing 'k'
        if lvl_time < 20:
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                                 0x4B, 0)
            time.sleep(key_releaseBuffer)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                                 0x4B, 0)

        # Two rewards here:

        # One for the amount of time it took to complete the level (ratio)
        # If the agent has obtained a yellow square, which increases its current level time
        # then set the time reward equal to the ratio of current time to intial level time
        if lvl_time > self.prev_stepTime:
            time_rewardRatio = lvl_time / self.lvl_initialTime
        else:
            time_rewardRatio = 0

        self.end_level = self.detect_end_levelMessage()

        done = bool(self.end_level != 0)

        # Second reward for completing the level
        lvl_completeReward = self.end_level

        self.prev_stepTime = lvl_time

        #reward = time_rewardRatio + lvl_completeReward

        reward = lvl_completeReward

        # restart level if we beat it
        if self.end_level == 1:
            self.completed_lvl_restart()

        # cap the reward to prevent outliers when saving mean model
        if reward < 0:
            reward = 0

        truncation = 0
        info = {}
        return observation, reward, done, truncation, info

    def screen_capture(self):

        game_window_bounds = win32gui.GetWindowRect(self.gameWindow_handle)

        # use to crop the image we are grabbing
        left_bound = game_window_bounds[0] + 35
        top_bound = game_window_bounds[1] + 85
        right_bound = game_window_bounds[2] - 35
        bottom_bound = game_window_bounds[3] - 50

        game_window_bounds = (left_bound, top_bound,
                              right_bound, bottom_bound)

        # capture the screen on restart
        screen = sct.grab(game_window_bounds)

        # mss returns BGRA, so need to convert to RGB
        screen = Image.frombytes('RGB', screen.size, screen.bgra, 'raw', 'BGRX')

        screen = np.array(screen)

        # resize the screen
        observation = cv2.resize(screen, (self.observation_height, self.observation_width))

        # Use the time rectangle to get the intial amount of time 
        # the agent has to complete the level 
        time_capture = np.array(sct.grab(self.time_left_bounds))

        # use BGRA2GRAY for sct captures, BGR2GRAY for ImageGrab
        time_captureGS = cv2.cvtColor(time_capture, cv2.COLOR_BGRA2GRAY)

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
            for i, p1 in enumerate(contours):
                old_points.append(i)  # prevent redudancy
                for j, p2 in enumerate(contours):
                    if i != j and j not in old_points:
                        dist = np.linalg.norm(p1 - p2, axis=-1)
                        distances.append((i, j, dist))
            time_rect_length = max(distances, key=lambda item: item[2])[2]
        except:
            time_rect_length = self.prev_stepTime

        # convert extracted time to float
        lvl_time = round(time_rect_length, 2)

        return observation, lvl_time

    def detect_end_levelMessage(self):

        # determine if the end level message is present
        message_capture = np.array(sct.grab(self.lvlComplete_msgBounds))

        # use BGRA2GRAY for sct captures, BGR2GRAY for ImageGrab
        message_gs = cv2.cvtColor(message_capture, cv2.COLOR_BGRA2GRAY)

        extracted_text = pytesseract.image_to_string(Image.fromarray(message_gs),
                                                     config='--psm 7',
                                                     lang='eng')

        message_text = extracted_text.rstrip()

        # determine how similar the text is to the true message
        levelComplete_message = 'level complete! press JUMP to continue'

        # get similarity between level complete text and extracted text
        lvlComplete_similarity = round(SequenceMatcher(None,
                                                       message_text, levelComplete_message).ratio(), 2)

        # do same for end of game message
        lvlFail_message = 'ouch... press JUMP to continue.'

        lvlFail_similarity = round(SequenceMatcher(None,
                                                   message_text, lvlFail_message).ratio(), 2)

        if (lvlComplete_similarity > lvlFail_similarity) and (lvlComplete_similarity >= self.lvlComplete_msgThresh):
            end_level = 1
        elif (lvlFail_similarity > lvlComplete_similarity) and (lvlFail_similarity >= self.lvlComplete_msgThresh):
            end_level = -1
        else:
            end_level = 0

        return end_level

    def close(self):
        # close the game window
        win32gui.PostMessage(self.gameWindow_handle, win32con.WM_CLOSE, 0, 0)

    def start_game(self, game_path: str ='D:\\VideoGame_AI\\N_game\\Nv2-PC.exe'):

        window_w = 967
        window_h = 844

        window_x = -8
        window_y = 0

        # view menu position for first screen
        screen_viewX = 54 # OMEN 72
        screen_viewY = 35 # OMEN 51

        os.startfile(game_path)

        time.sleep(1)  # wait for game to load

        # do this to avoid errors
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')

        hwnd = win32gui.FindWindow(None, 'Adobe Flash Player 11')

        cx = window_w - window_x
        cy = window_h - window_y

        win32gui.SetWindowPos(hwnd, win32con.HWND_TOP,
                              window_x, window_y,
                              cx, cy, 0)

        win32gui.SetWindowText(hwnd, 'NGame')
        new_hwnd = win32gui.FindWindow(None, 'NGame')

        # click on screen to show all. That way window can be smaller
        # REDUNDANT
        #pyautogui.click(screen_viewX, screen_viewY)
        #time.sleep(.25)
        #pyautogui.click(screen_viewX, screen_viewY + 50)

        # highlights play game
        for i in range(2):
            win32gui.SendMessage(new_hwnd, win32con.WM_KEYDOWN, self.down_arrowcode, 0)
            time.sleep(.05)
            win32gui.SendMessage(new_hwnd, win32con.WM_KEYUP, self.down_arrowcode, 0)

        # selects play game
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYDOWN,
                             self.action_lookup[2], 0)
        time.sleep(.1)
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYUP,
                             self.action_lookup[2], 0)

        # highlights level 00 game
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYDOWN,
                             self.action_lookup[0], 0)
        time.sleep(.05)
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYUP,
                             self.action_lookup[0], 0)

        # selects level 00 game
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYDOWN,
                             self.action_lookup[2], 0)
        time.sleep(.1)
        win32gui.SendMessage(new_hwnd, win32con.WM_KEYUP,
                             self.action_lookup[2], 0)

        # click on 100% to get rid of lines
        pyautogui.click(screen_viewX, screen_viewY)
        time.sleep(.25)
        pyautogui.click(70, 74)

        self.gameWindow_handle = new_hwnd

        # moves mouse off screen
        pyautogui.click(1542, 185)

    def completed_lvl_restart(self):
        # press escape
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                             0x1B, 0)
        time.sleep(0.5)
        win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                             0x1B, 0)

        # press down key twice to highlight play game
        for i in range(2):
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN, self.down_arrowcode, 0)
            time.sleep(.05)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP, self.down_arrowcode, 0)

        #pyautogui.click(826, 346)
        #time.sleep(0.05)
        #pyautogui.click(62, 225)

        for i in range(2):
            # press Z twice to continue
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYDOWN,
                                 self.action_lookup[2], 0)
            time.sleep(0.05)
            win32gui.SendMessage(self.gameWindow_handle, win32con.WM_KEYUP,
                                 self.action_lookup[2], 0)

        # moves mouse off screen
        pyautogui.click(1542, 185)