"""Standalone game launcher — run this before train_agent.py.

Uses the MCP server's launch_game() to start PCSX2, then polls memory
until the game reaches a playable state (health pointer is readable).
"""
import sys
import time
import pymem
import pymem.process
import win32gui
import win32process

WINDOW_TITLE = 'Slot: 0'
FULL_HEALTH = 40_000
POLL_TIMEOUT = 120


def wait_for_playable_state(window_title: str = WINDOW_TITLE, timeout: int = POLL_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = win32gui.FindWindow(None, window_title)
        if not hwnd:
            time.sleep(2)
            continue
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if not pid:
            time.sleep(2)
            continue
        try:
            pm = pymem.Pymem()
            pm.open_process_from_id(pid)
            base = pymem.process.module_from_name(pm.process_handle, 'pcsx2.exe').lpBaseOfDll
            ptr = pm.read_int(base + 0x01243984)
            health = pm.read_int(ptr + 0xA4)
            if 0 < health <= FULL_HEALTH:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    from mcp_pcsx2_server import launch_game

    print("Launching PCSX2 via MCP server...")
    result = launch_game()
    print(result)

    print("Waiting for game to reach a playable state...")
    if wait_for_playable_state():
        print("Game ready. You can now run train_agent.py.")
    else:
        print("ERROR: game did not reach a playable state within 120s.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
