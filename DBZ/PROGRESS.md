# Project Goal

Build reinforcement learning agents capable of playing popular video games, starting with Dragon Ball Z: Budokai Tenkaichi 3 via the PCSX2 emulator.

# Infrastructure & Tooling

- **CLAUDE.md** created — documents project overview, run/train commands, architecture, code conventions (Pydantic typing, single `train_agent.py` entry point), Git/GitHub conventions (2025-06-14)
- **Git branching convention** established — model branches merge only into `dev`, never `main`; `dev` branch created (2025-06-14)
- **`train_agent.py` refactored** — replaced loose constants with Pydantic configs (`TrainConfig`, `ExperimentConfig`, `CheckpointMeta`, `ExperimentResults`, `Transition`); added `MODEL_REGISTRY` for plug-and-play model swapping; eliminated globals in `select_action` and `optimize_model`; added TensorBoard logging (reward, loss, epsilon, wall time, VRAM); outputs now written to `models/<model_name>/` per convention (2025-06-14)
- **`.claude/skills/machine-learning-researcher.md`** created — full researcher workflow: experiment proposal, model file creation, TensorBoard monitoring, hyperparameter tuning, visualization (`./figures/`), model comparison, and commit conventions; VRAM reference table for RTX 5070 Ti included (2025-06-14)
- **`.claude/skills/progress-tracker/SKILLS.md`** created — skill for maintaining this file; enforces 200-line limit and structured sections (2025-06-14)

# Experiments

No experiments run yet.

# Known Issues

- Frame stacking may only save the last frame rather than all 4 — see `gym_DBZ/envs/Custom_DBZ_Game.py`
- "Fight Again" episode-reset detection is partially implemented; relies on both OCR and health thresholds
- Opponent attack memory address reads are unreliable (noted in `Custom_DBZ_Game.py`)
