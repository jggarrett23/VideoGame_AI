# TODO: Detect end of fight message.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymem
from utils import capture_screen, press_key, press_controller_button, navigate_to_fight_memory
import sched
import time
import win32ui, win32gui, win32process
import cv2
import matplotlib
import matplotlib.pyplot as plt
import easyocr
import threading
import vgamepad as vg

matplotlib.use('TkAgg')

# need a buffer of time between key press and release (seconds)
key_releaseBuffer = 0.5

scheduler = sched.scheduler(time.time, time.sleep)
# reader moved to DBZ_Env.__init__ so each instance owns its own EasyOCR model (thread-safe for multi-env)


class DBZ_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    PCSX2_EXE = r'D:\PCSX2 1.6.0\pcsx2.exe'
    ISO_PATH  = r'D:\PCSX2 1.6.0\Dragon Ball Z - Budokai Tenkaichi 3 (USA) (En,Ja).iso'
    MENU_TEMPLATES_DIR = r'D:\VideoGame_AI\DBZ\menu_screenshots'

    def __init__(self, game_window_title=None, observation_size=128, observation_buffer_size=4,
                 health_threshold=100, full_health=40000, navigate=True, env_idx: int = 0):
        super(DBZ_Env, self).__init__()

        # create virtual xbox controller
        self.gamepad = vg.VX360Gamepad()

        xbox_controller_button_names = [
            'Dpad_Up', 'Dpad_Down', 'Dpad_Left', 'Dpad_Right', 'Y', 'X', 'B', 'A', 'L1', 'R1',
            'L2', 'R2'
        ]

        xbox_controller_buttons = [
            vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_Y, vg.XUSB_BUTTON.XUSB_GAMEPAD_X, vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            vg.XUSB_BUTTON.XUSB_GAMEPAD_A, vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER, vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            ('LeftTrigger', ), ('RightTrigger', )
        ]

        self.action_lookup = dict(zip(xbox_controller_button_names, xbox_controller_buttons))

        # get combo actions
        dpad_codes = [self.action_lookup[x] for x in xbox_controller_button_names if 'Dpad' in x]
        combo_punches_codes = []
        for code in dpad_codes:
            combo_punches_codes.append((code, self.action_lookup['X']))

        # L2 + Circle, L2 + D-Pad Up + Circle, L2 + Triangle, L2 + D-Pad Up + Triangle, L2 + D-Pad Down + Triangle
        combo_specials_keys = [('L2', 'B'), ('L2', 'Dpad_Up', 'B'),
                               ('L2', 'Y'), ('L2', 'Dpad_Up', 'Y'),
                               ('L2', 'Dpad_Down', 'Y'), ('L2', 'A', 'L2', 'A')
                               ]
        combo_specials_codes = []
        for combo in combo_specials_keys:
            combo_codes = []
            for k in combo:
                combo_codes.append(self.action_lookup[k])
            combo_specials_codes.append(tuple(combo_codes))
        combos = combo_punches_codes + combo_specials_codes

        for i in range(len(combos)):
            self.action_lookup[f'Combo_{i}'] = combos[i]

        self.action_keys = list(self.action_lookup.keys())
        self.action_space = spaces.Discrete(len(self.action_lookup))

        # shape of the observation
        # change this to be the size desired
        self.observation_height = observation_size
        self.observation_width = observation_size
        self.observation_channels = 1

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.observation_channels, self.observation_height, self.observation_width),
            dtype=np.uint8
        )

        # bounds for cropping game window during capture
        # top, bottom, left, right,
        self.capture_bounds = [40, 20, 15, 15]

        # thresholds for resetting game
        self.health_threshold = health_threshold
        self.full_health = full_health
        self.special_attack_ki_thresh = 60040

        self.game_window_title = game_window_title
        self.game_window_handle = None
        self.env_idx = env_idx

        self.reader = easyocr.Reader(['en'])

        # read memory for reset signals
        self.pm = pymem.Pymem()  # Instantiate pymem without arguments
        self.memory_addresses = {}

        self.hook_memory_codes()

        if navigate:
            navigate_to_fight_memory(
                self.pm, self.memory_addresses,
                self.gamepad, self.action_lookup,
            )

        self.player_health = self.pm.read_int(self.memory_addresses['player_health'])
        self.opp_health = self.pm.read_int(self.memory_addresses['opponent_health'])
        self.player_ki = self.pm.read_int(self.memory_addresses['player_ki'])
        self.prev_action = None

        self.observation_buffer_size = observation_buffer_size
        self.observation_buffer = np.zeros(
            (self.observation_buffer_size, self.observation_height, self.observation_width),
            dtype=np.float32)

        self.frame_skip = 4
        self.frame_cnt = 0

        # distance to opponent for melee to land
        self.player_dist_threshold = 1131883873

    def hook_memory_codes(self):
        matches = []
        win32gui.EnumWindows(
            lambda hwnd, _: matches.append(hwnd) if win32gui.GetWindowText(hwnd) == self.game_window_title else None,
            None
        )
        self.game_window_handle = matches[self.env_idx]
        _, pid = win32process.GetWindowThreadProcessId(self.game_window_handle)

        if pid:
            try:
                # Attach to the process using the process name instead of PID
                self.pm.open_process_from_id(pid)  # Open the process by PID

                print(f"Successfully attached to process with PID: {pid}")

                base_address = pymem.process.module_from_name(self.pm.process_handle, 'pcsx2.exe').lpBaseOfDll

                # player and opponent health
                player_health_address = base_address + 0x01243984
                self.memory_addresses['player_health'] = self.pm.read_int(player_health_address) + 0xA4

                opp_health_address = base_address + 0x01243988
                self.memory_addresses['opponent_health'] = self.pm.read_int(opp_health_address) + 0x6A4

                # player ki
                player_ki_address = base_address + 0x01243984
                self.memory_addresses['player_ki'] = self.pm.read_int(player_ki_address) + 0xB0

                # start menu (for detecting end game)
                start_address = base_address + 0x012439A8
                self.memory_addresses['start'] = self.pm.read_int(start_address) + 0x2C8

                # attack initiated (any player)
                damage_address = base_address + 0x012439A0
                damage_address = self.pm.read_int(damage_address) + 0xC9C
                self.memory_addresses['damage_address'] = damage_address

                # WARNING not reliable
                opp_attack_address = base_address + 0x1199F50
                self.memory_addresses['opp_attack_address'] = opp_attack_address

                player_dist_address = base_address + 0x0124398C
                player_dist_address = self.pm.read_int(player_dist_address) + 0x464
                self.memory_addresses['player_opp_dist_address'] = player_dist_address

                # Start Game Menu
                # address = 14 when at menu screen, 2147483648 when trailer running, 2218774995 when memory loading, 2081454128 if space not pressed fast enough at menu
                start_game_screen = base_address + 0x123F2B8
                start_game_screen = self.pm.read_int(start_game_screen) + 0x1C8
                self.memory_addresses['start_game_screen'] = start_game_screen

                # Continue game (continue = 1065353216)
                continue_game = base_address + 0x0124072C
                continue_game = self.pm.read_int(continue_game) + 0xAE4
                self.memory_addresses['continue_game'] = continue_game

                # Menu navigation
                # Duel option = 144
                menu_options = base_address + 0x011EA1D4
                menu_options = self.pm.read_int(menu_options) + 0xB54
                self.memory_addresses['menu_options'] = menu_options

                fight_pause_menu = base_address + 0x00752408
                fight_pause_menu = self.pm.read_int(fight_pause_menu) + 0x178
                
                # 1 = continue battle 2 = view skills 3 = return to character select 4 = return to main menu
                self.memory_addresses['fight_pause_menu'] = fight_pause_menu

                end_fight_menu = base_address + 0x0123F284
                for code in [0x570, 0x6B4, 0xFE4]:
                    end_fight_menu = self.pm.read_int(end_fight_menu) + code

                # 1 = fight again 2 = return to character select 3 = return to vs 4 = return to main menu
                self.memory_addresses['fight_again'] = end_fight_menu

                
            except pymem.exception.ProcessNotFound:
                print(f"Process with PID {pid} not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Failed to get process ID")

    def sample_n_process_screen(self):
        screen = capture_screen(self.game_window_handle, bound_deltas=self.capture_bounds)
        observation = cv2.resize(screen, (self.observation_height, self.observation_width),
                                 interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = np.asarray(observation, dtype=float) / 255.

        # update observation buffer
        if not self.frame_cnt or not self.frame_cnt % self.frame_skip or self.frame_cnt < self.frame_skip:
            self.observation_buffer = np.append(self.observation_buffer, observation[np.newaxis, :], axis=0)
            self.observation_buffer = np.delete(self.observation_buffer, 0, axis=0)

        self.frame_cnt += 1

        return

    def calculate_reward(self, action_key):

        reward = 0.0

        health_reward = True
        special_attack_reward = True
        block_reward = False      # disabled: opp_attack_address is unreliable
        attack_dist_reward = False  # disabled: player_dist_threshold int/float mismatch unverified
        ki_reward = True

        current_player_health = self.pm.read_int(self.memory_addresses['player_health'])
        current_opp_health = self.pm.read_int(self.memory_addresses['opponent_health'])

        current_player_ki = self.pm.read_int(self.memory_addresses['player_ki'])

        if health_reward:
            # scale by full_health so a 2000 HP hit yields ~0.25 reward
            opp_damage    = max(0, self.opp_health    - current_opp_health)    / self.full_health
            player_damage = max(0, self.player_health - current_player_health) / self.full_health
            reward = 5.0 * (opp_damage - player_damage)

        if ki_reward:
            # reward ki generation
            if current_player_ki > self.player_ki and self.prev_action == 'L2':
                reward += 0.15

            # penalize holding counter and losing key
            if current_player_ki < self.player_ki and action_key == 'B':
                reward -= 0.4

        if special_attack_reward:
            # if action was a special but not enough ki, penalize
            if current_player_ki < self.special_attack_ki_thresh:
                if action_key in ['Combo_6', 'Combo_7', 'Combo_8']:
                    reward -= 0.2
                # else:
                #     reward += 0.2  # disabled: fires on nearly every step (ki starts at 0),
                #                    # drowning out the health differential signal

            if action_key == 'Combo_9' and current_player_ki < 2*self.special_attack_ki_thresh:
                reward -= 0.3

        if block_reward:
            # amount of damage in opp attack
            opp_attack_val = self.pm.read_int(self.memory_addresses['damage_address'])
            opp_attack_detected = self.pm.read_int(self.memory_addresses['opp_attack_address'])

            # reward blocking attacks/penalize random blocking
            if opp_attack_val and opp_attack_detected and action_key == 'B':
                reward += 0.6
            elif action_key == 'B':
                reward -= 0.3

        if attack_dist_reward:
            # penalize attacking when too far from opponent
            player_opp_dist = self.pm.read_int(self.memory_addresses['player_opp_dist_address'])
            if action_key in ['X', 'Combo_1', 'Combo_2', 'Combo_3', 'Combo_4'] and player_opp_dist >= self.player_dist_threshold:
                reward -= 0.3

        # small step penalty to discourage passivity
        reward -= 0.005

        # clamp reward values
        if reward >= 1:
            reward = 0.95
        elif reward <= -1:
            reward = -0.95

        done = False
        # big reward value if win or lose
        if current_player_health <= self.health_threshold:
            reward = -1
            done = True
        elif current_opp_health <= self.health_threshold:
            reward = 1
            done = True

        # update health and ki trackers
        self.player_health = current_player_health
        self.opp_health = current_opp_health
        self.player_ki = current_player_ki
        self.prev_action = action_key

        return reward, done

    def reset(self, seed=0, options=None):

        # check if start window is open for "Fight again option
        if self.pm.read_int(self.memory_addresses['start']) and self.pm.read_int(self.memory_addresses['fight_again'] == 1):
            press_controller_button(gamepad, self.action_keys['A'])
            time.sleep(0.1)
        else:
            self.pm.write_int(self.memory_addresses['player_health'], self.full_health)
            self.pm.write_int(self.memory_addresses['opponent_health'], self.full_health)

        # reset observation buffer
        self.observation_buffer = np.zeros(
            (self.observation_buffer_size, self.observation_height, self.observation_width),
            dtype=np.float32)

        self.frame_cnt = 0

        # concatenate first screen shot to observation buffer
        self.sample_n_process_screen()

        info = {}

        return self.observation_buffer, info

    def step(self, action_idx, key_hold_time=0.1):
        action_key = self.action_keys[action_idx]
        action_inputs = self.action_lookup[action_key]

        #thread = threading.Thread(
        #    target=press_controller_button,
        #    args=(self.vController, action_input, key_hold_time)
        #)
        #thread.start()

        press_controller_button(gamepad=self.gamepad, buttons=action_inputs, hold_time=key_hold_time)

        # capture updated state
        self.sample_n_process_screen()
        reward, done = self.calculate_reward(action_key)

        if done:
            self.reset()

        truncation = False
        info = {}
        return self.observation_buffer, reward, done, truncation, info

    def render(self) -> None:
        pass


"""
controller_kbrd_mapping_dict = {
            'Start': 'Space',
            'Select': ']',
            'D-Pad Up': 'T',
            'D-Pad Down': 'I',
            'D-Pad Left': 'Y',
            'D-Pad Right': 'U',
            'Triangle': 'Z',
            'Square': 'X',
            'Circle': 'C',
            'Cross': 'V',
            'L1': 'Q',
            'R1': 'W',
            'L2': 'E',
            'R2': 'R'
        }
        
# mapped to the same values in dict above
        keyboard_codes = {
            'Space': 0x20,
            ']': 0xDD,
            'T': 0x54,
            'I': 0x49,
            'Y': 0x59,
            'U': 0x55,
            'Z': 0x5A,
            'X': 0x58,
            'C': 0x43,
            'V': 0x56,
            'Q': 0x51,
            'W': 0x57,
            'E': 0x45,
            'R': 0x52
        }
        
# key_combos = [['E', 'C'], ['E', 'T', 'C'],
        #              ['E', 'Z'], ['E', 'T', 'Z'],
        #              ['E', 'I', 'Z']
        #              ]
"""