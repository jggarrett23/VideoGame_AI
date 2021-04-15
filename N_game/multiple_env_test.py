import os
import pyautogui

import time
import win32gui, win32con, win32com.client
import pyautogui
import multiprocessing as mp

shell = win32com.client.Dispatch("WScript.Shell")
shell.SendKeys('%')

key_codes = {}
key_codes['r_arrow'] = 0x25
key_codes['l_arrow'] = 0x27
key_codes['z'] = 0x5A
key_codes['d_arrow'] = 0x28

# start the game
window_list = []
def start_multiple_ngames(game_number: int, window_w: int,
    window_h: int, window_x: int, window_y: int, screen_viewX: int,
    screen_viewY: int) -> int:
    

    os.startfile('D:\\Nv2-PC.exe')
    
    time.sleep(.2)
    hwnd = win32gui.FindWindow(None,'Adobe Flash Player 11')
    
    win32gui.SetWindowPos(hwnd,win32con.HWND_TOP,
                          window_x,window_y,
                          window_w,window_h,0)
    
    win32gui.SetWindowText(hwnd,'NGame'+str(game_number))
    new_hwnd = win32gui.FindWindow(None,'NGame'+str(game_number))
    
    
    # click on first screen to show all. That way window can be smaller
    pyautogui.click(screen_viewX,screen_viewY)
    time.sleep(.25)
    pyautogui.click(screen_viewX,screen_viewY+50)
    
    
    # highlights play game
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYDOWN,key_codes['d_arrow'],0)
    time.sleep(.1)
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYUP,key_codes['d_arrow'],0)
    
    # selects play game
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYDOWN,key_codes['z'],0)
    time.sleep(.1)
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYUP,key_codes['z'],0)
    
    # selects first level 
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYDOWN,key_codes['r_arrow'],0)
    time.sleep(.1)
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYUP,key_codes['r_arrow'],0)
    
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYDOWN,key_codes['z'],0)
    time.sleep(.01)
    win32gui.SendMessage(new_hwnd,win32con.WM_KEYUP,key_codes['z'],0)
    
    return new_hwnd


## CURRENTLY NOT WORKING, SERIAL MIGHT BE BEST BET

if __name__ == '__main__':

    # window size and location settings
    window_w = 910
    window_h = 514

    window_x = 284
    window_y = 0

    # view menu position for first screen
    screen_viewX = 348
    screen_viewY = 50

    for game_numb in range(2):

        cx = window_w-window_x
        cy = window_h-window_y


        p = mp.Process(target=start_multiple_ngames,
            args=(game_numb, cx, cy, window_x, window_y, 
                screen_viewX, screen_viewY, ))
        p.start()

        window_x = window_w 
        window_w += cx

        screen_viewX += cx
    

    p.join()