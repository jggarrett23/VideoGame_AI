# Project Goal

Build reinforcement learning agents capable of playing popular video games, starting with Dragon Ball Z: Budokai Tenkaichi 3 via the PCSX2 emulator.

# Infrastructure & Tooling

- **CLAUDE.md** created — documents project overview, run/train commands, architecture, code conventions (Pydantic typing, single `train_agent.py` entry point), Git/GitHub conventions (2025-06-14)
- **Git branching convention** established — model branches merge only into `dev`, never `main`; `dev` branch created (2025-06-14)
- **`train_agent.py` refactored** — replaced loose constants with Pydantic configs (`TrainConfig`, `ExperimentConfig`, `CheckpointMeta`, `ExperimentResults`, `Transition`); added `MODEL_REGISTRY` for plug-and-play model swapping; eliminated globals in `select_action` and `optimize_model`; added TensorBoard logging (reward, loss, epsilon, wall time, VRAM); outputs now written to `models/<model_name>/` per convention (2025-06-14)
- **`.claude/skills/machine-learning-researcher.md`** created — full researcher workflow: experiment proposal, model file creation, TensorBoard monitoring, hyperparameter tuning, visualization (`./figures/`), model comparison, and commit conventions; VRAM reference table for RTX 5070 Ti included (2025-06-14)
- **`.claude/skills/progress-tracker/SKILLS.md`** created — skill for maintaining this file; enforces 200-line limit and structured sections (2025-06-14)
- **`mcp_pcsx2_server.py`** created — FastMCP server giving Claude direct access to live game state; tools: `connect_to_game`, `capture_frame`, `read_game_state`, `send_action`, `list_actions`, `navigate_to_fight_screen`, `reset_episode`, `launch_game`; registered in `.claude/settings.json` as the `pcsx2` MCP server (2026-06-19)
- **`capture_screen` background-window fix** — replaced desktop-DC `BitBlt` (captured whichever window was on top) with a `HWND_TOPMOST` pin + `BitBlt` + `HWND_NOTOPMOST` restore in `utils.py`; PCSX2 can now be behind other windows during capture without returning a black or wrong-window frame (2026-06-19)
- **`dueling_cnn` model added** — Nature DQN-style CNN (8×4s4, 4×4s2, 3×3s1) with separate value and advantage streams (Dueling DQN, Wang et al. 2016); shared 512-unit FC + duration head; replaces `cnn_fc` as the active training model; registered in `MODEL_REGISTRY` in `train_agent.py`; branch `dueling_cnn` created (2026-06-19)
- **`navigate_to_fight` crash fix** — `DBZ_Env.__init__` now does a CV template pre-check before calling `navigate_to_fight`; if no menu template matches the current frame (game is already in a fight), navigation is skipped entirely, preventing the 120s `TimeoutError` that occurred when training was launched with the game mid-fight (2026-06-19)
- **Game launch refactored** — removed `auto_launch` parameter from `DBZ_Env`; replaced with `navigate=True/False`; env always calls `hook_memory_codes()` then navigates if a menu template matches; standalone `launch_game.py` created — calls `mcp_pcsx2_server.launch_game()` to start PCSX2 then polls memory until health is readable; `run_train_agent.ps1` now runs `launch_game.py` before training; `mcp_pcsx2_server.connect_to_game()` uses `navigate=False` (2026-06-19)
- **`.claude/skills/experiment-launcher/SKILL.md`** created — skill documenting how to start experiments and how menu template navigation works (2026-06-19)

- **Reward function overhaul** — diagnosed flat 0.18–0.25 reward signal as caused by `special_attack_reward` `else: reward += 0.2` firing on nearly every step; removed that term; health differential rescaled to `5.0 * (opp_damage - player_damage) / full_health` so a 2000 HP hit yields ~0.25; `block_reward` and `attack_dist_reward` disabled (unreliable memory addresses); added `-0.005` step penalty; `eps_decay` raised from 1000 → 5000 (agent was stopping exploration after ~20 episodes); `duration_loss` removed from Bellman update (duration is continuous control, not a Q-value) (2026-06-20)
- **Log file naming** — changed from `model_name.logs` (overwritten each run) to `model_name_YYYYMMDD_HHMM.logs` per-run in `train_agent.py` (2026-06-20)
- **Vectorized environment training** — `vec_env.py` created with `VectorizedDBZEnv` class; uses `ThreadPoolExecutor` to step N `DBZ_Env` instances in parallel (IO-bound steps release GIL); `Custom_DBZ_Game.py` gets `env_idx: int = 0` param + `EnumWindows`-based window disambiguation in `hook_memory_codes`; `easyocr.Reader` moved from module scope into `DBZ_Env.__init__` (per-instance, thread-safe); `ReplayMemory` made thread-safe with `threading.Lock`; training loop in `train_agent.py` branches on `ExperimentConfig.num_envs` (single-env path unchanged) (2026-06-20)
- **Vec env active-mask fix** — `VectorizedDBZEnv.step()` accepts `active: list[bool]` mask; done envs are skipped entirely (no phantom button presses in freshly-reset fights while other envs finish) (2026-06-20)
- **Early stopping** — patience-based early stopping added to `train_agent.py`; `TrainConfig` gains `early_stopping_patience` (default 20) and `early_stopping_min_delta` (default 0.001); monitors 10-episode moving average; configurable via `--patience` CLI arg (default 5) (2026-06-20)
- **CLI / launch chain updated** — `train_agent.py` argparse: `--num-envs`, `--patience` added; `run_train_agent.ps1`: `$NumEnvs` param added, passed to both `launch_game.py` and `train_agent.py`; `utils.launch_pcsx2` refactored to support `num_instances` (spawns until N windows exist, returns list of handles); `mcp_pcsx2_server.launch_game` gains `num_envs` param; `launch_game.py` gains `--num-envs` and waits for all N instances to report readable health before exiting (2026-06-20)

# Experiments

| Date | Branch | Model | Episodes | Envs | Notes |
|------|--------|-------|----------|------|-------|
| 2026-06-19 | `dueling_cnn` | `dueling_cnn` | 150 | 1 | Pre-reward-fix; rewards flat 0.18–0.25; 1 win; agent not dealing meaningful damage |
| 2026-06-20 | `dueling_cnn` | `dueling_cnn` | 15 | 2 | Post-reward-fix + vec env; rewards trended −0.045 → +0.030; 1 win; opp health reaching 21k–26k (agent landing hits) |

# Known Issues

- Frame stacking may only save the last frame rather than all 4 — see `Custom_DBZ_Game.py`
- "Fight Again" episode-reset detection is partially implemented; relies on both OCR and health thresholds
- Opponent attack memory address reads are unreliable (noted in `Custom_DBZ_Game.py`); `block_reward` disabled as a result
- Distance threshold (`player_dist_threshold = 1131883873`) is an int/float mismatch — `attack_dist_reward` disabled until verified
