"""Standalone game launcher — run this before train_agent.py.

Uses the MCP server's launch_game() to start PCSX2, then polls memory
until the game reaches a playable state (health pointer is readable).
"""
import argparse
import sys
import time
import pymem
import pymem.process
import win32gui
import win32process
from Custom_DBZ_Game import DBZ_Env
from utils import _enum_windows_with_title

WINDOW_TITLE = 'Slot: 0'
FULL_HEALTH = 40_000
POLL_TIMEOUT = 120


def _is_instance_playable(hwnd: int) -> bool:
    """Return True if the PCSX2 window at hwnd has a readable health value."""
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    if not pid:
        return False
    try:
        pm = pymem.Pymem()
        pm.open_process_from_id(pid)
        base = pymem.process.module_from_name(pm.process_handle, 'pcsx2.exe').lpBaseOfDll
        ptr = pm.read_int(base + 0x01243984)
        health = pm.read_int(ptr + 0xA4)
        return 0 < health <= FULL_HEALTH
    except Exception:
        return False
    


def wait_for_playable_state(num_envs: int = 1, window_title: str = WINDOW_TITLE,
                            timeout: int = POLL_TIMEOUT) -> bool:
    """Block until num_envs windows with window_title are all in a playable state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        handles = _enum_windows_with_title(window_title)
        ready = [h for h in handles if _is_instance_playable(h)]
        if len(ready) >= num_envs:
            return True
        time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PCSX2 for DBZ training")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of PCSX2 instances to launch")
    args = parser.parse_args()

    from mcp_pcsx2_server import launch_game

    print(f"Launching {args.num_envs} PCSX2 instance(s) via MCP server...")
    result = launch_game(num_envs=args.num_envs)
    print(result)

    """
    print("Waiting for all instances to reach a playable state...")
    if wait_for_playable_state(num_envs=args.num_envs):
        print(f"All {args.num_envs} instance(s) ready. You can now run train_agent.py.")
    else:
        print(f"ERROR: not all instances reached a playable state within {POLL_TIMEOUT}s.",
              file=sys.stderr)
        sys.exit(1)
    """

if __name__ == "__main__":
    main()
