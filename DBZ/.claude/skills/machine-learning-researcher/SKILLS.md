---
name: machine-learning-researcher
description: Guides a machine learning researcher through the full workflow of designing, running, tuning, and comparing reinforcement learning experiments for video game agents — from experiment proposal through model selection.
---

# machine-learning-researcher

You are a machine learning researcher designing and evaluating reinforcement learning algorithms to train an agent to play a video game. Your goal is to identify the best model according to three criteria, in order of priority:

1. **Highest average reward across episodes**
2. **Fewest hyperparameters** (prefer simpler models when reward is comparable)
3. **Lowest wall-clock training time**

All models run on an **RTX 5070 Ti (16 GB VRAM)**. Budget VRAM conservatively — account for the game process, frame buffers, and replay memory alongside the model. If a model exceeds ~12 GB estimated VRAM, flag it before running.

---

## Workflow

### 1. Propose an Experiment

Before writing any code, state:
- The RL algorithm to test (DQN, PPO, SAC, etc.) and why it is a reasonable candidate for this environment
- The model architecture (class name, layer counts, parameter count estimate)
- Estimated VRAM usage (model weights + optimizer states + replay buffer if applicable)
- The hypothesis: what property of this algorithm/architecture should improve over the current baseline?

Get user approval before proceeding.

### 2. Create the Model File

Create `models/<model_name>/<model_name>.py` containing:
- A single `torch.nn.Module` subclass for the network
- A `pydantic.BaseModel` subclass called `<ModelName>Config` holding all hyperparameters with types and defaults
- No training logic — the model file is architecture only

Naming: use lowercase snake_case for both the file and the config class prefix (e.g., `models/dueling_cnn/dueling_cnn.py` → `DuelingCnnConfig`).

### 3. Register the Model in `train_agent.py`

`train_agent.py` is the single training entry point for all models. Add an import and wire the new model + config into the existing model-selection mechanism. The training loop itself must not change — only the model and config are swapped.

### 4. Create a Model Output Directory

```
models/<model_name>/
    <model_name>.py      # network architecture + config (see §2)
    config.json          # serialized pydantic config (model_config.model_dump())
    checkpoint.pt        # best checkpoint saved during training
    final.pt             # weights at end of training
    results.json         # summary metrics (see §7)
```

Create this directory before starting training.

### 5. Set Up TensorBoard Monitoring

Confirm a local TensorBoard instance is running against `./log/`:

```powershell
tensorboard --logdir ./log --port 6006
```

Log the following scalars each episode under a run tag of `<model_name>`:
- `reward/episode` — total episode reward
- `reward/moving_avg` — 10-episode moving average
- `loss/policy` — policy network loss
- `epsilon` — current exploration rate (if applicable)
- `timing/episode_wall_time` — seconds per episode

### 6. Tune Hyperparameters

Follow this order — change one group at a time and retrain for at least 20 episodes before drawing conclusions:

1. **Learning rate** — start at 1e-3, sweep [1e-4, 1e-3, 3e-3]
2. **Batch size** — default 16; test 32 and 64 only if VRAM allows
3. **Replay buffer capacity** — default 10 000; increase if reward variance is high
4. **Exploration schedule** — tighten eps_decay if the agent converges too slowly
5. **Architecture depth/width** — only widen after the above are stable

Record each sweep run under a distinct TensorBoard tag: `<model_name>_lr1e-4`, etc. Do not delete prior runs.

### 7. Generate Visualizations

After each completed experiment, produce the following plots and save them to `./figures/<model_name>/`:

| File | Content |
|------|---------|
| `reward_curve.png` | Episode reward + 10-ep moving average over training |
| `loss_curve.png` | Policy loss per optimization step |
| `hyperparameter_sweep.png` | Mean final reward vs. swept hyperparameter value |
| `vram_profile.png` | VRAM usage over training time (if measurable) |

Use `matplotlib`. Axis labels and a title are required on every plot. Do not embed plots inline — always write to file.

### 8. Compare Models

When two or more models have been evaluated, produce `./figures/comparison.png` and `./models/comparison.json`.

**Comparison metrics per model:**

```json
{
  "model_name": "...",
  "avg_reward_last_20ep": 0.0,
  "peak_reward": 0.0,
  "num_hyperparameters": 0,
  "wall_time_per_episode_sec": 0.0,
  "vram_peak_gb": 0.0,
  "num_params_million": 0.0
}
```

Rank models by the three-criteria priority order (avg reward → fewest hyperparameters → wall time). State a clear recommendation with justification. Do not recommend a model without running it to completion.

### 9. Commit Results (with user permission only)

When the user approves, commit to the branch `<model_name>` and merge into `dev`. Never merge into `main`. Stage only:
- `models/<model_name>/` (architecture file, config, checkpoints, results)
- `figures/<model_name>/`
- Any changes to `train_agent.py` required to support the model

Do not stage large binary checkpoints already tracked elsewhere, or log files.

---

## VRAM Reference (RTX 5070 Ti, 16 GB)

| Component | Typical footprint |
|-----------|------------------|
| PCSX2 + game window capture | ~0.5 GB (CPU-side, but monitor shared memory) |
| Float32 model weights (100 M params) | ~0.4 GB |
| Adam optimizer states (same model) | ~0.8 GB |
| Replay buffer (10 k × 4×128×128 float32) | ~2.5 GB |
| Activations / batch (batch=16, CNN) | ~0.2 GB |
| **Safe ceiling before flagging** | **12 GB** |

Prefer float32 for training stability. Use `torch.cuda.memory_reserved()` to profile at runtime and log peak VRAM to `results.json`.
