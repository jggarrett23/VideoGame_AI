import numpy as np 
from PIL import ImageGrab
import cv

def screen_record(): 
    while True:
        # 800x600 windowed mode
        printscreen_pil =  ImageGrab.grab(bbox=(0,40,800,640))
        printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype='uint8')\
        .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
        cv2.imshow('window',cv2.cvtColor(printscreen_numpy, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break