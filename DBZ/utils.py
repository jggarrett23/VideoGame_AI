import win32gui
import win32process
import win32ui
import win32con
import numpy as np
from PIL import Image
import time
import pyautogui
import vgamepad


def get_process_id_by_window(partial_title):
    """
    Get the process ID of a window whose title contains a specific substring.
    :param partial_title: Substring to search for in the window title.
    :return: Process ID if a matching window is found, else None.
    """
    target_pid = None

    def enum_windows_callback(hwnd, _):
        nonlocal target_pid
        # Get the window title
        window_title = win32gui.GetWindowText(hwnd)

        # Check if the window title contains the partial title
        if partial_title in window_title:
            win32gui.SetWindowText(hwnd, 'DBZ BT3')
            # Get the process ID associated with the window
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            target_pid = pid
            return False  # Stop enumeration once the target is found

    # Enumerate all top-level windows
    win32gui.EnumWindows(enum_windows_callback, None)

    return target_pid


def capture_screen(hwnd, bound_deltas=None):
    game_window_bounds = win32gui.GetWindowRect(hwnd)

    width = game_window_bounds[2] - game_window_bounds[0]
    height = game_window_bounds[3] - game_window_bounds[1]

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (game_window_bounds[2], game_window_bounds[3]),
               dcObj, (0, 0), win32con.SRCCOPY)

    # convert bitmap to something we can use
    bmpdata = dataBitMap.GetBitmapBits(True)

    pil_im = Image.frombuffer('RGB', (width, height),
                              bmpdata, 'raw', 'BGRX', 0, 1)
    screen = np.array(pil_im)

    if any(bound_deltas):
        screen = screen[bound_deltas[0]:-bound_deltas[1], bound_deltas[2]:-bound_deltas[3], :]

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return screen


def press_key(keys, hold_time=0.1):

    for key in keys:
        pyautogui.keyDown(key)

    time.sleep(hold_time)

    for key in keys:
        pyautogui.keyUp(key)


def press_controller_button(gamepad, buttons, hold_time=0.1):
    for button in buttons:
        if isinstance(button, tuple) or isinstance(button, str):
            if 'LeftTrigger' in button:
                gamepad.left_trigger_float(value_float=1.0)
            elif 'RightTrigger' in button:
                gamepad.right_trigger_float(value_float=1.0)
        else:
            gamepad.press_button(button=button)

    gamepad.update()
    time.sleep(hold_time)

    for button in buttons:
        if isinstance(button, tuple) or isinstance(button, str):
            if 'LeftTrigger' in button:
                gamepad.left_trigger_float(value_float=0.0)
            elif 'RightTrigger' in button:
                gamepad.right_trigger_float(value_float=0.0)
        else:
            gamepad.release_button(button=button)

    gamepad.update()