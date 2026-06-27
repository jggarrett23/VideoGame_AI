import ctypes
import os
import subprocess
import time

import cv2
import numpy as np
import pymem
import pyautogui
import vgamepad
import win32con
import win32gui
import win32process
import win32ui
from PIL import Image

PW_RENDERFULLCONTENT = 2

# Ordered menu_screenshots/ steps: (filename, button action, press count).
# Each screen is matched against a live capture; once matched, its button
# pattern is fired to advance toward the fight screen.
MENU_TEMPLATE_STEPS = [
    ('Start Screen_1.png',     'start', 1),
    ('Start Screen_2.png',     'cross', 1),
    ('Menu_3.png',              'down', 3),
    ('Menu_4.png',             'cross', 1),
    ('Versus_5.png',           'cross', 1),
    ('Versus_6.png',           'cross', 1),
    ('Character Select_7.png', 'cross', 6),
]


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
    # Pin PCSX2 to the top of the z-order so the desktop BitBlt reads its GPU
    # frame rather than whatever window happens to be in front of it.
    _SWP_FLAGS = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, _SWP_FLAGS)

    rect = win32gui.GetWindowRect(hwnd)
    x, y = rect[0], rect[1]
    width  = rect[2] - rect[0]
    height = rect[3] - rect[1]

    hdesktop = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hdesktop)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (width, height), dcObj, (x, y), win32con.SRCCOPY)

    bmpdata = dataBitMap.GetBitmapBits(True)
    pil_im = Image.frombuffer('RGB', (width, height), bmpdata, 'raw', 'BGRX', 0, 1)
    screen = np.array(pil_im)

    if bound_deltas and any(bound_deltas):
        screen = screen[bound_deltas[0]:-bound_deltas[1], bound_deltas[2]:-bound_deltas[3], :]

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hdesktop, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    # Restore normal z-order so PCSX2 doesn't permanently float above everything.
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, _SWP_FLAGS)

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


def _enum_windows_with_title(title: str) -> list[int]:
    """Return handles of all visible windows matching title."""
    matches: list[int] = []
    win32gui.EnumWindows(
        lambda hwnd, _: matches.append(hwnd) if win32gui.GetWindowText(hwnd) == title else None,
        None,
    )
    return matches


def launch_pcsx2(pcsx2_exe: str, iso_path: str, window_title: str = 'Slot: 0',
                 timeout: int = 120, num_instances: int = 1) -> list[int]:
    """Launch PCSX2 until num_instances windows with window_title exist.

    Returns a list of window handles (one per instance), in the order they appeared.
    Already-running instances count toward the target.
    """
    existing = _enum_windows_with_title(window_title)
    handles  = list(existing)

    while len(handles) < num_instances:
        subprocess.Popen([pcsx2_exe, iso_path])
        deadline = time.time() + timeout
        while time.time() < deadline:
            current = _enum_windows_with_title(window_title)
            new = [h for h in current if h not in handles]
            if new:
                handles.extend(new)
                break
            time.sleep(2)
        else:
            raise TimeoutError(
                f'PCSX2 window "{window_title}" did not appear within {timeout}s '
                f'(have {len(handles)}/{num_instances})'
            )

    return handles[:num_instances]


def wait_for_game_load(pm: pymem.Pymem, base_health_ptr_addr: int, timeout: int = 200,
                       skip_gamepad=None, skip_btn=None) -> int:
    """Wait for the game to finish loading past the intro trailer.

    Sleeps 10 s first so PCSX2's window becomes interactive, then polls
    base_health_ptr_addr every 0.5 s. While waiting, presses skip_btn every
    3 s if a gamepad is provided — pass XUSB_GAMEPAD_START to skip publisher
    intro screens. Returns the resolved health address (ptr + 0xA4).
    """
    time.sleep(10)  # let PCSX2 render its first frame before inputs register
    deadline = time.time() + timeout
    last_skip = 0.0
    while time.time() < deadline:
        now = time.time()
        if skip_gamepad is not None and skip_btn is not None and now - last_skip >= 3.0:
            press_controller_button(skip_gamepad, (skip_btn,), 0.1)
            last_skip = now
        try:
            ptr = pm.read_int(base_health_ptr_addr)
            if ptr != 0:
                return ptr + 0xA4
        except Exception:
            pass
        time.sleep(0.5)

    raise TimeoutError(f'Game did not load within {timeout}s')


def load_menu_templates(folder: str, size: tuple = (160, 120)) -> list:
    """Load the menu_screenshots/ reference images used by navigate_to_fight.

    Each image is grayscaled and resized to a common size so it can be
    correlated against a live capture regardless of minor crop differences
    between how the screenshots were taken and how capture_screen crops the
    live window.
    """
    templates = []
    for filename, action, count in MENU_TEMPLATE_STEPS:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f'Menu template not found: {path}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        templates.append({'name': filename, 'action': action, 'count': count, 'image': gray})
    return templates


def match_menu_template(frame: np.ndarray, templates: list, threshold: float = 0.90):
    """Return (template, score) for the best-matching menu template, or (None, score).

    Uses normalized cross-correlation (cv2.matchTemplate) between the live
    frame and each reference image, both resized to the same size.
    """
    size = templates[0]['image'].shape[::-1]  # (w, h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    best, best_score = None, threshold
    for tpl in templates:
        score = float(cv2.matchTemplate(gray, tpl['image'], cv2.TM_CCOEFF_NORMED)[0, 0])
        if score >= best_score:
            best, best_score = tpl, score
    return best, best_score


def navigate_to_fight_memory(
    pm: pymem.Pymem,
    memory_addresses: dict,
    gamepad: vgamepad.VX360Gamepad,
    action_lookup: dict,
    timeout: float = 180,
) -> None:
    """Navigate DBZ menus to the fight screen using memory address polling.

    Replaces CV template matching. Polls start_game_screen, continue_game, and
    menu_options addresses (resolved in hook_memory_codes) to gate each button press.
    """
    _START_LOADING = 2218774995  # memory loading — wait, don't press
    _START_TRAILER = 2147483648  # trailer running — press Start to skip
    _START_MENU    = 14          # title screen ready

    start_btn = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_START
    cross_btn = action_lookup['A']
    down_btn  = action_lookup['Dpad_Down']
    deadline  = time.time() + timeout

    # Phase 1: poll start_game_screen until title screen (14).
    # Print value when it matches known intermediate states.
    print("navigate: waiting for title screen (start_game_screen == 14)...")
    while pm.read_int(memory_addresses['start_game_screen']) != 14:
        
        press_controller_button(gamepad, (start_btn,), 0.1)
        time.sleep(0.3)

    time.sleep(2.0)
    press_controller_button(gamepad, (start_btn,), 0.1)
    time.sleep(0.3)

    # Phase 2: press Start once to enter main menu
    print("navigate: entering main menu...")
    press_controller_button(gamepad, (start_btn,), 0.1)
    time.sleep(1.5)

    # Phase 3: press Down until continue_game == 1065353216, then press Cross
    print("navigate: navigating to Continue...")
    while pm.read_int(memory_addresses['continue_game']) != 0:
        press_controller_button(gamepad, down_btn, 0.1)
        time.sleep(0.1)

    press_controller_button(gamepad, cross_btn, 0.1)
    time.sleep(3.0)
    
    # Phase 4: press Down until menu_options == 144 (Duel highlighted), then press Cross
    print("navigate: navigating to Duel...")
    for _ in range(3):
        press_controller_button(gamepad, down_btn, 0.1)
        time.sleep(2.0)
    
    press_controller_button(gamepad, cross_btn, 0.1)
    time.sleep(3.0)

    # Phase 5: advance through versus setup and character select
    for _ in range(9):
        press_controller_button(gamepad, cross_btn, 0.1)
        time.sleep(3.5)

    time.sleep(2.0)
     
    # advance to fight start. first skip past intro, then pause game
    while pm.read_int(memory_addresses['fight_pause_menu']) != 1:
        press_controller_button(gamepad, start_btn, 0.1)
        time.sleep(0.5)

    print("navigate: done.")
    
    # continue battle
    press_controller_button(gamepad, cross_btn, 0.1)
    time.sleep(0.1)


def navigate_to_fight(gamepad: vgamepad.VX360Gamepad, action_lookup: dict, hwnd: int,
                      templates: list, capture_bounds=(40, 20, 15, 15),
                      threshold: float = 0.90, timeout: float = 120,
                      poll_interval: float = 0.5, after_action: float = 1.5,
                      retry_after: float = 3.0) -> None:
    """Navigate DBZ menus to the fight screen via CV template matching (no window focus needed).

    Each loop iteration captures the live screen and matches it against
    templates from load_menu_templates(). Whichever reference screen clears
    `threshold` determines the button pattern fired next. Driving off what's
    actually on screen (rather than a fixed button/timing script) makes this
    robust to PCSX2 already being mid-session — e.g. left on a different
    menu from a previous run — since a step only fires once its expected
    screen is genuinely showing.

    If the same template keeps matching for longer than `retry_after`
    seconds (its button press apparently didn't register), the action is
    retried.
    """
    start_btn = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_START
    button_map = {
        'start': (start_btn,),
        'cross': action_lookup['A'],
        'down':  action_lookup['Dpad_Down'],
    }
    final_template_name = MENU_TEMPLATE_STEPS[-1][0]

    last_fired, last_fired_at = None, 0.0
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = capture_screen(hwnd, bound_deltas=capture_bounds)
        tpl, score = match_menu_template(frame, templates, threshold)

        if tpl is None:
            time.sleep(poll_interval)
            continue

        if tpl['name'] == last_fired and time.time() - last_fired_at < retry_after:
            time.sleep(poll_interval)
            continue

        buttons = button_map[tpl['action']]
        for _ in range(tpl['count']):
            press_controller_button(gamepad, buttons, hold_time=0.1)
            if tpl['count'] > 1:
                time.sleep(0.5)

        last_fired, last_fired_at = tpl['name'], time.time()

        if tpl['name'] == final_template_name:
            return  # character select confirmed -> fight is starting

        time.sleep(after_action)

    raise TimeoutError('navigate_to_fight: no menu template matched within the timeout')