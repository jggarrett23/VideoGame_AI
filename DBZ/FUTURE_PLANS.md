# Future Plans

## Option 2: PINE IPC MCP Server (PCSX2 1.7+ / PCSX2-QT)

### What is PINE?
PINE (PCSX2 Interprocess Network Extension) is a built-in IPC protocol introduced in PCSX2 1.7.x
(nightly/dev builds and the current stable PCSX2-QT). It exposes game memory and emulator control
via a named pipe on Windows (`\\.\pipe\pcsx2.{slot}`) or Unix socket on Linux/macOS.

Unlike the current Option 1 approach (which reads PCSX2 process memory directly via pymem), PINE
is an official, documented API that doesn't require admin privileges or Cheat Engine address scanning.

### What PINE exposes
- `MsgRead8 / MsgRead16 / MsgRead32 / MsgRead64` — read PS2 memory at any address
- `MsgWrite8 / MsgWrite16 / MsgWrite32 / MsgWrite64` — write PS2 memory
- `MsgVersion` — get emulator version string
- `MsgStatus` — get emulator state (running / paused / etc.)
- `MsgPause` / `MsgResume` / `MsgReset` — control emulation
- `MsgGetGameTitle` / `MsgGetGameID` / `MsgGetGameUUID` — game identification
- Save/load states (in newer builds)

### Why it's better than Option 1
| Concern | Option 1 (pymem) | Option 2 (PINE) |
|---|---|---|
| Address stability | Pointer chains re-scanned via Cheat Engine | Same PS2 RAM addresses across emulator versions |
| Admin rights | Requires process memory access (often admin) | Named pipe, no special privileges |
| Emulator control | None (read/write only) | Pause, resume, reset, save/load state |
| Cross-platform | Windows only | Windows + Linux + macOS |

### Migration steps
1. **Upgrade PCSX2**: Switch from 1.6.0 to PCSX2-QT (current stable, 2.x). Verify the same
   `Dragon Ball Z - Budokai Tenkaichi 3 (USA)` ISO boots and pointer addresses still apply in PS2 RAM.
2. **Re-scan memory addresses**: The Cheat Engine pointer chains in `Custom_DBZ_Game.py` are
   offsets into PCSX2's host process. PINE exposes PS2 RAM directly, so the pointer chain base
   offsets will change. Use Cheat Engine on PCSX2-QT to re-derive them, then store as PS2 RAM
   offsets rather than host process offsets.
3. **Install a PINE client library**: Use [`pine-client`](https://github.com/GovanifY/pine) (C++)
   or write a thin Python wrapper around the named pipe protocol
   (`\\.\pipe\pcsx2.0` by default on Windows). The protocol is documented in the PCSX2 repo.
4. **Replace `hook_memory_codes()` in `Custom_DBZ_Game.py`**: Instead of pymem + module base
   address resolution, open the named pipe and send PINE read requests.
5. **Extend the MCP server** (Option 1's `mcp_pcsx2_server.py`): Add tools for
   `pause_emulator`, `resume_emulator`, `save_state`, `load_state` using PINE's control messages.

### Resources
- PINE documentation: https://github.com/PCSX2/pcsx2/blob/master/pcsx2/IPC.h
- PINE protocol reference: https://github.com/GovanifY/pine
- PCSX2-QT release: https://pcsx2.net/downloads/
