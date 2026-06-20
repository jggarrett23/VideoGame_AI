---
name: experiment-launcher
description: Launches a DBZ training experiment end-to-end — starts PCSX2 via the MCP server, navigates to the fight screen using CV template matching, and kicks off train_agent.py. Use this skill whenever the user says "begin experiment", "start training", or "run experiment".
---

# experiment-launcher

Use this skill to start a training run for the DBZ agent. It covers the two-phase setup process: game launch (via MCP) and menu navigation (via the environment's template matching).

---

## Phase 1: Launch the Game via MCP

Call the MCP `launch_game` tool to start PCSX2 if it isn't already running:

```tool
mcp__pcsx2-dbz__launch_game
```

This calls `launch_pcsx2()` from `utils.py` — it checks for an existing window titled `Slot: 0` before spawning a new process, so it is safe to call even if the game is already running.

Then run `launch_game.py` to wait for the game to reach a playable state (health pointer readable):

```powershell
& "D:\VideoGame_AI\DBZ\venv\Scripts\python.exe" "D:\VideoGame_AI\DBZ\launch_game.py"
```

`launch_game.py` polls the pointer chain `pcsx2.exe base + 0x01243984 → read_int → +0xA4` every 2 seconds for up to 120 seconds. It exits 0 when health is in `(0, 40000]`, or exits 1 on timeout.

---

## Phase 2: Menu Navigation (handled by DBZ_Env)

When `train_agent.py` starts, `DBZ_Env.__init__` automatically handles navigation:

1. **`hook_memory_codes()`** — attaches pymem to the PCSX2 process and resolves all memory addresses.
2. **Template pre-check** — captures one frame and runs `match_menu_template()` against the 7 reference images in `menu_screenshots/`.
3. **Navigate if needed** — if a menu template matches (score ≥ 0.90), calls `navigate_to_fight()` which drives the menus to the character select screen using button sequences.
4. **Skip if already fighting** — if no template matches, the game is already in a fight and navigation is skipped.

### Menu Template Steps

Templates are loaded in order from `MENU_TEMPLATE_STEPS` in `utils.py`:

| Step | File | Action | Presses |
|------|------|--------|---------|
| 1 | `Start Screen_1.png` | Start button | 1 |
| 2 | `Start Screen_2.png` | Cross (A) | 1 |
| 3 | `Menu_3.png` | D-Pad Down | 3 |
| 4 | `Menu_4.png` | Cross (A) | 1 |
| 5 | `Versus_5.png` | Cross (A) | 1 |
| 6 | `Versus_6.png` | Cross (A) | 1 |
| 7 | `Character Select_7.png` | Cross (A) | 6 |

Navigation returns as soon as template 7 (`Character Select_7.png`) is matched. Each matched template fires its action, then waits up to 3 seconds before retrying if the screen hasn't advanced.

---

## Phase 3: Start Training

```powershell
& "D:\VideoGame_AI\DBZ\venv\Scripts\python.exe" "D:\VideoGame_AI\DBZ\train_agent.py" --model dueling_cnn --episodes 150
```

Or use the full run script (which includes the launch step):

```powershell
.\run_train_agent.ps1 -Model dueling_cnn -Episodes 150
```

Monitor progress by reading the log file:

```powershell
Get-Content "D:\VideoGame_AI\DBZ\models\dueling_cnn\dueling_cnn.logs" -Tail 20
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `TimeoutError: navigate_to_fight: no menu template matched` | Game is in a state not covered by templates (e.g., cutscene, loading) | Wait for game to reach title screen, then rerun |
| `launch_game.py` exits with code 1 | Game didn't load within 120s | Check PCSX2 is launching the correct ISO; increase `POLL_TIMEOUT` in `launch_game.py` |
| `hook_memory_codes` fails silently | PCSX2 window title changed from `Slot: 0` | Verify window title with `win32gui.GetWindowText`; update `WINDOW_TITLE` constants |
| Training hangs after Episode 1 | EasyOCR GPU model loading | Normal — can take 2–3 minutes on first run |
