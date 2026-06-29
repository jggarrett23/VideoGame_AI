# Future Plans

## ForecastDQN

- Change GRU to Transformer to capture longer context windows before passing Z to forecasting head.
- Track actions across all frames instead of the last one for a better control model.

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

---

## Option 3: Cheat Engine MCP Server

### Problem it solves

All game-state memory addresses in `Custom_DBZ_Game.py` are hardcoded pointer chains discovered manually via Cheat Engine (e.g. `base_address + 0x01243984` → dereference → `+ 0xA4` for player health). If the PCSX2 version changes, the ISO region changes, or a new game variable is needed (e.g. stun state, super armor, character ID), the entire scan process must be redone by hand. An MCP server for Cheat Engine lets Claude perform value scans, pointer scans, and address discovery autonomously.

### Architecture

Two components communicate over HTTP on localhost:

```
Claude (MCP client)
    │
    ▼
mcp_cheat_engine_server.py   ← FastMCP server, same pattern as mcp_pcsx2_server.py
    │  HTTP requests to :9096
    ▼
ce_http_bridge.lua            ← Lua script loaded inside Cheat Engine
    │  Cheat Engine Lua API
    ▼
Cheat Engine (running, attached to pcsx2.exe)
```

Cheat Engine 7.x ships with a built-in Lua engine and HTTP server support. The Lua bridge script registers handlers for scan, filter, read, write, and pointer-scan operations and serves them over a local port.

### Lua bridge script (`ce_http_bridge.lua`)

Loaded via Cheat Engine → Table → Show Cheat Table Lua Script → Execute, or placed in CE's `autorun/` folder.

Key Lua API calls used:
- `createMemScan()` / `firstScan()` / `nextScan()` — value scans
- `createAddressList()` / `getAddressList()` — result enumeration
- `getPointerScanner()` / `PointerScan()` — pointer chain discovery
- `readInteger(addr)` / `writeInteger(addr, val)` — direct memory R/W
- `openProcess(name)` — attach to target process

The HTTP server portion uses CE's Lua `socket` library (bundled) to listen on `127.0.0.1:9096` and respond to JSON-encoded requests.

### MCP server (`mcp_cheat_engine_server.py`)

Follows the same `FastMCP` pattern as `mcp_pcsx2_server.py`. All tools send HTTP POST requests to the Lua bridge and return JSON results.

**Tools:**

| Tool | Parameters | Description |
|------|-----------|-------------|
| `attach_to_process` | `process_name: str` | Attach CE to a running process (e.g. `"pcsx2.exe"`) |
| `first_scan` | `value: int`, `scan_type: str` | Start a new scan (`"exact"`, `"greater"`, `"less"`, `"unknown"`) |
| `next_scan` | `value: int`, `condition: str` | Filter results (`"exact"`, `"increased"`, `"decreased"`, `"changed"`, `"unchanged"`) |
| `get_scan_results` | `limit: int = 50` | Return current scan result addresses + current values |
| `read_address` | `address: int`, `data_type: str` | Read memory (`"int32"`, `"float"`, `"int16"`) |
| `write_address` | `address: int`, `value: int`, `data_type: str` | Write memory |
| `pointer_scan` | `address: int`, `max_depth: int = 4`, `max_offset: int = 0x1000` | Run pointer scan for a given dynamic address; returns static pointer chains |
| `save_address` | `name: str`, `address: int` | Add address to CE address list (for export) |
| `export_cheat_table` | `path: str` | Save `.CT` file to disk for later import |
| `get_address_list` | — | Return all addresses currently in CE's address list |

### Intended workflow for re-scanning addresses

The current addresses were found by:
1. In-game, player at full health (40000) → scan for `40000`
2. Take damage → filter "decreased"
3. Repeat until 1–2 candidates remain
4. Do pointer scan on the candidate → get stable chain relative to `pcsx2.exe` base

With the MCP server, Claude can do this autonomously:
```
attach_to_process("pcsx2.exe")
first_scan(40000, "exact")          # player at full health
# [user takes damage in game]
next_scan(0, "decreased")           # health went down
get_scan_results()                  # inspect candidates
pointer_scan(candidate_address)     # find stable chain
save_address("player_health", chain_base)
export_cheat_table("dbz_addresses.CT")
```

This produces a `.CT` file that can also be imported manually into CE for inspection, and the discovered chains can be copied back into `Custom_DBZ_Game.py`'s `hook_memory_codes()`.

### Files to create

| File | Purpose |
|------|---------|
| `ce_http_bridge.lua` | Lua HTTP server to load inside Cheat Engine |
| `mcp_cheat_engine_server.py` | FastMCP server with tools listed above |

Register in `.claude/settings.json` as a second MCP server alongside `pcsx2-dbz`:
```json
"cheat-engine": {
    "command": "python",
    "args": ["D:/VideoGame_AI/DBZ/mcp_cheat_engine_server.py"]
}
```

### Dependencies / prerequisites

- Cheat Engine 7.4+ (ships with Lua 5.4 and socket library)
- CE must be running and attached to the target process before MCP tools are called
- No new Python packages needed — the MCP server only makes `http.client` requests to the Lua bridge

### Limitations

- CE's Lua HTTP server is single-threaded; concurrent tool calls will queue
- Pointer scans can take 30–120 s depending on depth/offset range; MCP tool should return a job ID and poll for completion
- Writing memory via this server bypasses the safety checks in `pymem`; restrict write tools to known address ranges
