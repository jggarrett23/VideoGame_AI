import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import base64
import time
from typing import Optional

import cv2
import numpy as np
import win32gui
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict

from Custom_DBZ_Game import DBZ_Env
from utils import capture_screen, launch_pcsx2, load_menu_templates, navigate_to_fight, press_controller_button

PCSX2_EXE = r'D:\PCSX2 1.6.0\pcsx2.exe'
ISO_PATH = r'D:\PCSX2 1.6.0\Dragon Ball Z - Budokai Tenkaichi 3 (USA) (En,Ja).iso'
MENU_TEMPLATES_DIR = r'D:\VideoGame_AI\DBZ\menu_screenshots'
WINDOW_TITLE = 'Slot: 0'

mcp = FastMCP('pcsx2-dbz')


class GameSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    env: Optional[DBZ_Env] = None
    hwnd: Optional[int] = None
    connected: bool = False


session = GameSession()


def _require_connection() -> Optional[str]:
    if not session.connected or session.env is None:
        return "Not connected. Call connect_to_game() first."
    return None


@mcp.tool()
def launch_game(num_envs: int = 1) -> str:
    """Launch PCSX2 with DBZ. Starts num_envs instances if not already running."""
    try:
        handles = launch_pcsx2(PCSX2_EXE, ISO_PATH, WINDOW_TITLE, num_instances=num_envs)
        session.hwnd = handles[0]
        return f"PCSX2 launched. {len(handles)} instance(s). Handles: {handles}"
    except TimeoutError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error launching PCSX2: {e}"


@mcp.tool()
def connect_to_game() -> str:
    """Attach to a running PCSX2 process and resolve all game memory addresses.
    PCSX2 must already be running with DBZ loaded and past the title screen.
    """
    try:
        env = DBZ_Env(game_window_title=WINDOW_TITLE, navigate=False)
        if not env.memory_addresses:
            return "Failed to resolve memory addresses. Is PCSX2 running with DBZ loaded?"
        session.env = env
        session.hwnd = env.game_window_handle
        session.connected = True
        addrs = list(env.memory_addresses.keys())
        return f"Connected. Resolved addresses: {addrs}"
    except Exception as e:
        return f"Error connecting: {e}"


@mcp.tool()
def read_game_state() -> dict:
    """Read current game state from PCSX2 memory.
    Returns player_health, opponent_health, player_ki, distance, damage_value, opp_attack_value.
    """
    err = _require_connection()
    if err:
        return {"error": err}
    env = session.env
    try:
        state = {
            "player_health": env.pm.read_int(env.memory_addresses["player_health"]),
            "opponent_health": env.pm.read_int(env.memory_addresses["opponent_health"]),
            "player_ki": env.pm.read_int(env.memory_addresses["player_ki"]),
            "player_opp_distance": env.pm.read_int(env.memory_addresses["player_opp_dist_address"]),
            "damage_value": env.pm.read_int(env.memory_addresses["damage_address"]),
            "opp_attack_value": env.pm.read_int(env.memory_addresses["opp_attack_address"]),
            "start_menu_active": bool(env.pm.read_int(env.memory_addresses["start"])),
        }
        return state
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def capture_frame() -> dict:
    """Capture the current game screen and return it as a base64-encoded PNG string.
    The image is the raw (uncropped) window capture.
    """
    err = _require_connection()
    if err:
        return {"error": err}
    try:
        hwnd = session.hwnd or win32gui.FindWindow(None, WINDOW_TITLE)
        frame = capture_screen(hwnd)
        _, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf).decode("utf-8")
        h, w = frame.shape[:2]
        return {"image_base64": b64, "width": w, "height": h, "format": "png"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_actions() -> list[str]:
    """Return all available action names that can be passed to send_action()."""
    err = _require_connection()
    if err:
        return [f"ERROR: {err}"]
    return session.env.action_keys


@mcp.tool()
def send_action(action_name: str, hold_time: float = 0.1) -> str:
    """Send a controller action to the game.

    Args:
        action_name: One of the action names returned by list_actions().
        hold_time: How long to hold the button(s) in seconds (default 0.1).
    """
    err = _require_connection()
    if err:
        return err
    env = session.env
    if action_name not in env.action_lookup:
        valid = ", ".join(env.action_keys)
        return f"Unknown action '{action_name}'. Valid actions: {valid}"
    try:
        buttons = env.action_lookup[action_name]
        press_controller_button(env.gamepad, buttons, hold_time=hold_time)
        return f"Sent action '{action_name}' (hold_time={hold_time}s)"
    except Exception as e:
        return f"Error sending action: {e}"


@mcp.tool()
def navigate_to_fight_screen() -> str:
    """Navigate DBZ menus from wherever the game currently is to the fight screen.
    Uses CV template matching — works regardless of which menu the game is on.
    Blocks until the fight screen is reached or the 120s timeout expires.
    """
    err = _require_connection()
    if err:
        return err
    env = session.env
    try:
        templates = load_menu_templates(MENU_TEMPLATES_DIR)
        navigate_to_fight(
            env.gamepad,
            env.action_lookup,
            env.game_window_handle,
            templates,
            capture_bounds=env.capture_bounds,
        )
        return "Reached fight screen."
    except TimeoutError as e:
        return f"Timeout: {e}"
    except Exception as e:
        return f"Error navigating menus: {e}"


@mcp.tool()
def reset_episode() -> dict:
    """Reset the current fight. If 'Fight Again' is on screen, presses it.
    Otherwise writes full health to both characters directly in memory.
    Returns the shape of the new observation buffer.
    """
    err = _require_connection()
    if err:
        return {"error": err}
    try:
        obs, info = session.env.reset()
        return {"observation_shape": list(obs.shape), "info": info}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
