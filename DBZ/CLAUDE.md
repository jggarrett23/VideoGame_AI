# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is part of a broader effort to develop reinforcement learning agents capable of playing popular video games. The current implementation is a Deep Q-Network (DQN) agent that learns to play Dragon Ball Z: Budokai Tenkaichi 3 via the PCSX2 PS2 emulator on Windows. The agent reads game state directly from PCSX2 process memory (via pymem/Cheat Engine-discovered addresses), captures screen frames for visual input, and controls the game through a virtual Xbox controller (vgamepad).

## Running and Training

**Prerequisites before any training run:**
- PCSX2 emulator must be running with DBZ: Budokai Tenkaichi 3 loaded
- Game window must be titled `"Slot: 0"` (the environment captures this window by title)
- A virtual Xbox controller must be configured (vgamepad)
- CUDA GPU expected; CPU training is untested

**To train:**
```powershell
# Activate venv and train (also sleeps PC on completion)
.\run_train_agent.ps1

# Or run directly
.\venv\Scripts\activate
python train_agent.py
```

**To run offline autoencoder training on collected gameplay data:**
```powershell
python offline_training.py
```

## Architecture

## Launch game
`DBZ_Env` launches PCSX2, waits for the game to load, and navigates to the fight screen automatically when `auto_launch=True` (the default). No manual steps are needed — just run `.\run_train_agent.ps1`.

For manual/debug launches, the emulator can be started with:
```
'D:\PCSX2 1.6.0\pcsx2.exe' --nogui "D:\PCSX2 1.6.0\Dragon Ball Z - Budokai Tenkaichi 3 (USA) (En,Ja).iso"
```
Then pass `auto_launch=False` to `DBZ_Env` and navigate manually (Space → v → i×3 → v×3 → v×7, which maps to Start → Cross → D-Pad Down×3 → Cross×3 → Cross×7 on the gamepad).

### Environment (`gym_DBZ/envs/Custom_DBZ_Game.py`)
Wraps the live game as an OpenAI Gymnasium environment:
- **Observation**: 4-frame stacked grayscale images `(4, 128, 128)`, resized from PCSX2 window capture, normalized to `[0,1]`. Frame skip: every 4th frame is added to the buffer.
- **Action space**: 22 discrete actions — 12 base controller inputs (D-pad, face buttons, triggers) + 10 combo/special-move sequences. The network also predicts key-press duration (0.1–6 seconds).
- **Game state**: Read from PCSX2 process memory via pymem using hardcoded pointer offsets (documented in `dbz_cheat_engine.txt` and the Scans/ directory). Key addresses: player/opponent health, Ki, attack damage, distance.
- **Reward**: Multi-component dense reward — health differential, Ki generation, block rewards/penalties, attack-distance penalties, special-attack Ki penalties. Terminal: +1 win / -1 loss.
- **Episode reset**: Detected by health dropping below 20000; uses EasyOCR and pixel checks to identify "Fight Again" screen.

### Neural Networks (`models.py`)
Three model variants:
- **`cnn_fc`** — Current active model. 3 conv layers (32→64→128 channels) + dropout + two FC heads: action logits and duration prediction. Input noise augmentation during training.
- **`ViT`** — Custom Vision Transformer (patch embedding + multi-head attention + FC layers).
- **`PreTrained_DeiTModel`** — Facebook DeiT backbone (frozen) with fine-tuning head.

The active model in `train_agent.py` is `cnn_fc`.

### Training Loop (`train_agent.py`)
Standard DQN with:
- Experience replay buffer (capacity 10,000)
- Epsilon-greedy exploration: 0.9 → 0.05 over 1000 decay steps
- Soft target network updates (τ = 0.005)
- Loss: Huber (SmoothL1), optimizer: NAdam with weight decay 1e-5, gradient clipping at 100
- Q-values divided by 10 before loss computation for scale stability
- Checkpointing to `./Models/checkpoints/` when: opponent health decreases, player wins, or reward improves

### Data Collection & Offline Learning (`offline_training.py`)
Gameplay transitions (state, action, reward, next_state) are saved to HDF5 files in `./Gameplay_Data/`. `offline_training.py` trains a convolutional autoencoder on these frames for state representation pre-training.

## Key Hyperparameters (in `train_agent.py`)

```python
LR = 1e-3
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 0.9, 0.05, 1000
TAU = 0.005
NUM_EPISODES = 150
OBS_BUFFER = 4  # frame stack depth
```

## Code Conventions

- **Language & framework**: Python with PyTorch. All neural networks are `torch.nn.Module` subclasses defined in `models.py`.
- **Single training entry point**: `train_agent.py` is the one script used to train any model in this repo. Models are plug-and-play — swapping the active model means instantiating a different class from `models.py` and passing it to the training loop; no other changes should be needed. Do not create per-model training scripts.
- **Typing**: Use [Pydantic](https://docs.pydantic.dev/) for all structured data — hyperparameter configs, transition tuples, environment configs, checkpoint metadata, etc. Define types as `pydantic.BaseModel` subclasses with explicit field types, analogous to Rust struct declarations. Plain `dict` or loosely-typed dataclasses are not acceptable for structured data boundaries.

- **Hyperparameters**: Defined as a Pydantic model at the top of `train_agent.py` and passed explicitly — no module-level magic constants threaded through function signatures.

## Git & GitHub Conventions

- **Branching**: When testing a new model, create a dedicated branch named after the model (e.g., `cnn_fc`, `vit_base`). These branches merge only into `dev` — never into `main`.
- **Merging**: Only merge branches with explicit user permission.
- **Committing**: Only commit files with explicit user permission.
- **Ignore `N_game/`**: The `../N_game/` directory is a separate project effort. Never stage, commit, or otherwise touch any files under that path.

## Memory Address Notes

All game state memory addresses are PCSX2-process-relative pointer chains discovered via Cheat Engine. They are hardcoded in `Custom_DBZ_Game.py` and documented in `dbz_cheat_engine.txt`. If the emulator version or game region changes, these addresses will need to be re-scanned. The opponent attack detection address is noted in the code as **not reliable**.

## Known Issues

- Frame stacking may only be saving the last frame rather than all 4 — see comment in `Custom_DBZ_Game.py`.
- "Fight Again" detection is partially implemented; episode resets rely on both OCR and health thresholds.
- Opponent attack memory address reads are marked unreliable in comments.
