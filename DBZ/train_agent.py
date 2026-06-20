# TODO: Check Custom_DBZ_Game.py to make sure frames are being stacked properly, seems as though only the last frame is being saved

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

from Custom_DBZ_Game import DBZ_Env
from models import ViT, PreTrained_DeiTModel, cnn_fc, dueling_cnn
from vec_env import VectorizedDBZEnv


# ---------------------------------------------------------------------------
# Pydantic configs
# ---------------------------------------------------------------------------

class TrainConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 16
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 5000
    tau: float = 0.005
    num_episodes: int = 150
    obs_buffer: int = 4
    replay_capacity: int = 10_000
    img_size: int = 128
    weight_decay: float = 1e-5
    grad_clip: float = 100.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001


class ExperimentConfig(BaseModel):
    model_name: str = "cnn_fc"
    load_checkpoint: bool = False
    save_data: bool = False
    num_envs: int = 1


class CheckpointMeta(BaseModel):
    episode: int
    all_episode_rewards: list[float]


class ExperimentResults(BaseModel):
    model_name: str
    avg_reward_last_20ep: float
    peak_reward: float
    num_train_hyperparameters: int
    wall_time_per_episode_sec: float
    vram_peak_gb: float
    num_params_million: float
    total_episodes: int


# ---------------------------------------------------------------------------
# Transition & replay memory
# ---------------------------------------------------------------------------

class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: Any
    action: Any
    next_state: Any
    reward: Any


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, transition: Transition) -> None:
        with self._lock:
            self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        with self._lock:
            return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        with self._lock:
            return len(self.memory)


# ---------------------------------------------------------------------------
# Model registry — add new models here
# To register a new model:
#   1. Create models/<name>/<name>.py with the nn.Module subclass
#   2. Import the class and add it to MODEL_REGISTRY below
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn_fc": cnn_fc,
    "dueling_cnn": dueling_cnn,
    "vit": ViT,
    "pretrained_deit": PreTrained_DeiTModel,
}


def build_model(exp: ExperimentConfig, cfg: TrainConfig, n_actions: int) -> nn.Module:
    cls = MODEL_REGISTRY[exp.model_name]
    if exp.model_name == "vit":
        return cls(
            num_layers=4, in_channels=cfg.obs_buffer, img_size=cfg.img_size,
            emb_size=1024, patch_size=6, num_head=4, num_class=n_actions,
        )
    if exp.model_name == "pretrained_deit":
        return cls(in_channels=cfg.obs_buffer, num_classes=n_actions)
    return cls(in_channels=cfg.obs_buffer, num_classes=n_actions,
               img_shape=(cfg.img_size, cfg.img_size))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def select_action(
    state: Tensor,
    policy_net: nn.Module,
    env: DBZ_Env,
    cfg: TrainConfig,
    steps_done: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-steps_done / cfg.eps_decay)
    if random.random() > eps:
        with torch.no_grad():
            actions, duration = policy_net(state)
            return actions.max(1).indices.view(1, 1), duration.item()
    return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=device), 0.1


def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    memory: ReplayMemory,
    criterion: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
) -> Optional[float]:
    if len(memory) < cfg.batch_size:
        return None

    transitions = memory.sample(cfg.batch_size)

    non_final_mask = torch.tensor(
        [t.next_state is not None for t in transitions], device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [t.next_state for t in transitions if t.next_state is not None]
    ).to(device)

    state_batch  = torch.cat([t.state  for t in transitions]).to(device)
    action_batch = torch.cat([t.action for t in transitions]).to(device)
    reward_batch = torch.cat([t.reward for t in transitions]).to(device)

    optimizer.zero_grad()

    state_action_preds, state_duration_preds = policy_net(state_batch)
    state_action_values = state_action_preds.gather(1, action_batch)

    next_state_values        = torch.zeros(cfg.batch_size, device=device)
    next_state_duration_vals = torch.zeros(cfg.batch_size, device=device)
    with torch.no_grad():
        next_preds, next_duration_preds = target_net(non_final_next_states)
        next_state_values[non_final_mask]        = next_preds.max(1).values.to(next_state_values.dtype)
        next_state_duration_vals[non_final_mask] = next_duration_preds.squeeze().to(next_state_duration_vals.dtype)

    expected_action_values   = (next_state_values        * cfg.gamma) + reward_batch
    expected_duration_values = (next_state_duration_vals * cfg.gamma) + reward_batch

    # scale to prevent large Q-values destabilising loss
    action_loss   = criterion(state_action_values.squeeze()  / 10.0, expected_action_values   / 10.0)
    # duration_loss disabled: duration is a continuous control output, not a Q-value;
    # applying the Bellman equation to it is undefined and adds gradient noise to shared CNN layers.
    # duration_loss = criterion(state_duration_preds.squeeze() / 10.0, expected_duration_values / 10.0)
    total_loss = action_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), cfg.grad_clip)
    optimizer.step()

    return total_loss.item()


def soft_update(policy_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    target_sd = target_net.state_dict()
    policy_sd = policy_net.state_dict()
    for key in policy_sd:
        target_sd[key] = policy_sd[key] * tau + target_sd[key] * (1 - tau)
    target_net.load_state_dict(target_sd)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def initialize_h5_file(path: Path, img_size: int, obs_buffer: int, n_classes: int) -> None:
    with h5py.File(path, "w") as f:
        shape = (0, obs_buffer, img_size, img_size)
        max_shape = (None, obs_buffer, img_size, img_size)
        f.create_dataset("state",      shape=shape, maxshape=max_shape, dtype="float32")
        f.create_dataset("next_state", shape=shape, maxshape=max_shape, dtype="float32")
        f.create_dataset("action", shape=(0,), maxshape=(None,), dtype="float32")
        f.create_dataset("reward", shape=(0,), maxshape=(None,), dtype="float32")


def append_to_h5_file(path: Path, t: Transition) -> None:
    with h5py.File(path, "a") as f:
        for key, val in [("state", t.state), ("next_state", t.next_state),
                         ("action", t.action), ("reward", t.reward)]:
            f[key].resize(f[key].shape[0] + 1, axis=0)
            f[key][-1] = val.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors writes to two file-like objects (e.g. stdout + a log file)."""
    def __init__(self, *files):
        self._files = files

    def write(self, data: str) -> None:
        for f in self._files:
            f.write(data)
            f.flush()

    def flush(self) -> None:
        for f in self._files:
            f.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train DBZ DQN agent")
    parser.add_argument("--model",           type=str,   default="dueling_cnn",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train")
    parser.add_argument("--episodes",        type=int,   default=5,
                        help="Number of training episodes")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch-size",      type=int,   default=16,
                        help="Replay buffer batch size")
    parser.add_argument("--load-checkpoint", action="store_true",
                        help="Resume from saved checkpoint")
    parser.add_argument("--save-data",       action="store_true",
                        help="Save gameplay transitions to HDF5")
    parser.add_argument("--num-envs",  type=int, default=1, help="Number of parallel DBZ windows")
    parser.add_argument("--patience",  type=int, default=5, help="Early stopping patience (episodes)")

    args = parser.parse_args()


    cfg = TrainConfig(
        num_episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
    )
    exp = ExperimentConfig(
        model_name=args.model,
        load_checkpoint=args.load_checkpoint,
        save_data=args.save_data,
        num_envs=args.num_envs,
    )

    # --- paths ---------------------------------------------------------------
    model_dir      = Path("models") / exp.model_name
    checkpoint_path = model_dir / "checkpoint.pt"
    final_path      = model_dir / "final.pt"
    results_path    = model_dir / "results.json"
    log_dir         = Path("log") / exp.model_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = model_dir / f"{exp.model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.logs"
    _log_fh = open(log_file_path, "a")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)

    (model_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    # --- environment ---------------------------------------------------------
    device = torch.device("cuda")
    _env_kwargs = dict(
        game_window_title="Slot: 0",
        observation_size=cfg.img_size,
        observation_buffer_size=cfg.obs_buffer,
        health_threshold=20000,
        full_health=40000,
    )
    if exp.num_envs > 1:
        vec_env = VectorizedDBZEnv(num_envs=exp.num_envs, env_kwargs=_env_kwargs)
        env = vec_env.envs[0]  # reference env for action space / health props
    else:
        vec_env = None
        env = DBZ_Env(**_env_kwargs)
    n_actions = env.action_space.n.item()

    # --- models --------------------------------------------------------------
    policy_net = build_model(exp, cfg, n_actions).to(device)
    target_net = build_model(exp, cfg, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.NAdam(policy_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.SmoothL1Loss()
    memory    = ReplayMemory(cfg.replay_capacity)
    writer    = SummaryWriter(log_dir=str(log_dir))

    # --- state ---------------------------------------------------------------
    steps_done          = 0
    all_episode_rewards: list[float] = []
    episode_start       = 0
    max_reward          = 0.0
    opp_health_tracker  = env.full_health

    if exp.load_checkpoint and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        meta = CheckpointMeta(**ckpt["meta"])
        policy_net.load_state_dict(ckpt["model_state_dict"])
        target_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode_start       = meta.episode + 1
        all_episode_rewards = meta.all_episode_rewards
        max_reward          = max(all_episode_rewards) if all_episode_rewards else 0.0

    data_path: Optional[Path] = None
    if exp.save_data:
        data_path = Path("Gameplay_Data") / (exp.model_name + ".h5")
        data_path.parent.mkdir(exist_ok=True)
        initialize_h5_file(data_path, cfg.img_size, cfg.obs_buffer, n_actions)

    num_params_M      = sum(p.numel() for p in policy_net.parameters()) / 1e6
    player_wins       = 0
    ep_cnt            = 0
    vram_peak_gb      = 0.0
    total_wall_time   = 0.0
    training_start    = time.time()
    best_moving_avg   = -float("inf")
    patience_counter  = 0

    # --- training loop -------------------------------------------------------
    for i_episode in range(episode_start, episode_start + cfg.num_episodes):
        ep_start = time.time()
        print(f"Episode {i_episode + 1} / {episode_start + cfg.num_episodes}")

        episode_reward = 0.0
        step_cnt       = 0
        episode_loss   = 0.0
        loss_steps     = 0

        if exp.num_envs == 1:
            # ---- single-env path (unchanged) --------------------------------
            obs, _ = env.reset()
            obs  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            done = False

            while not done:
                actions, duration = select_action(obs.to(device), policy_net, env, cfg, steps_done, device)
                steps_done += 1

                next_obs, reward, terminated, truncated, _ = env.step(actions.item(), duration)
                episode_reward += reward
                step_cnt       += 1
                done = terminated or truncated

                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                reward_t   = torch.tensor([reward])

                t = Transition(state=obs, action=actions, next_state=next_obs_t, reward=reward_t)
                memory.push(t)
                obs = next_obs_t

                loss = optimize_model(policy_net, target_net, optimizer, memory, criterion, cfg, device)
                if loss is not None:
                    episode_loss += loss
                    loss_steps   += 1

                soft_update(policy_net, target_net, cfg.tau)

                if data_path is not None:
                    append_to_h5_file(data_path, t)

                vram_used = torch.cuda.memory_reserved(device) / 1e9
                if vram_used > vram_peak_gb:
                    vram_peak_gb = vram_used

            wins_this_ep  = 1 if env.player_health > env.opp_health else 0
            min_opp_health = env.opp_health

        else:
            # ---- vectorized path --------------------------------------------
            obs_list   = [torch.tensor(o, dtype=torch.float32).unsqueeze(0) for o in vec_env.reset()]
            done_flags = [False] * exp.num_envs

            while not all(done_flags):
                actions_list   = []
                durations_list = []
                for i in range(exp.num_envs):
                    if done_flags[i]:
                        actions_list.append(0)
                        durations_list.append(0.1)
                    else:
                        a, d = select_action(obs_list[i].to(device), policy_net, env, cfg, steps_done, device)
                        actions_list.append(a.item())
                        durations_list.append(d)
                steps_done += 1

                active = [not f for f in done_flags]
                results = vec_env.step(actions_list, durations_list, active=active)

                for i, (next_obs, reward, terminated, truncated, _) in enumerate(results):
                    if done_flags[i]:
                        continue
                    episode_reward += reward
                    step_cnt       += 1
                    done_i = terminated or truncated

                    next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                    reward_t   = torch.tensor([reward])
                    action_t   = torch.tensor([[actions_list[i]]], dtype=torch.long)

                    t = Transition(state=obs_list[i], action=action_t,
                                   next_state=None if done_i else next_obs_t,
                                   reward=reward_t)
                    memory.push(t)
                    obs_list[i]   = next_obs_t
                    done_flags[i] = done_flags[i] or done_i

                loss = optimize_model(policy_net, target_net, optimizer, memory, criterion, cfg, device)
                if loss is not None:
                    episode_loss += loss
                    loss_steps   += 1

                soft_update(policy_net, target_net, cfg.tau)

                vram_used = torch.cuda.memory_reserved(device) / 1e9
                if vram_used > vram_peak_gb:
                    vram_peak_gb = vram_used

            ph = vec_env.player_health
            oh = vec_env.opp_health
            wins_this_ep   = sum(1 for p, o in zip(ph, oh) if p > o)
            min_opp_health = min(oh)

        ep_wall_time     = time.time() - ep_start
        total_wall_time += ep_wall_time
        avg_reward       = episode_reward / max(step_cnt, 1)
        all_episode_rewards.append(avg_reward)

        player_wins += wins_this_ep
        ep_cnt      += 1

        eps        = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-steps_done / cfg.eps_decay)
        avg_loss   = episode_loss / loss_steps if loss_steps > 0 else 0.0
        moving_avg = sum(all_episode_rewards[-10:]) / min(len(all_episode_rewards), 10)

        if moving_avg > best_moving_avg + cfg.early_stopping_min_delta:
            best_moving_avg  = moving_avg
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.early_stopping_patience:
            print(f"  Early stopping: moving avg hasn't improved for {cfg.early_stopping_patience} episodes.")
            break

        writer.add_scalar("reward/episode",           avg_reward,   i_episode)
        writer.add_scalar("reward/moving_avg",        moving_avg,   i_episode)
        writer.add_scalar("loss/policy",              avg_loss,     i_episode)
        writer.add_scalar("epsilon",                  eps,          i_episode)
        writer.add_scalar("timing/episode_wall_time", ep_wall_time, i_episode)
        writer.add_scalar("vram/peak_gb",             vram_peak_gb, i_episode)

        print(f"  Reward: {avg_reward:.4f} | Moving avg: {moving_avg:.4f} | "
              f"Wins: {player_wins} | Losses: {ep_cnt - player_wins} | "
              f"Opp health: {min_opp_health} | Wall time: {ep_wall_time:.1f}s | "
              f"VRAM: {vram_peak_gb:.2f} GB")

        should_save = (
            min_opp_health < opp_health_tracker
            or wins_this_ep > 0
            or avg_reward >= max_reward
        )
        if should_save:
            print("  Saving checkpoint...")
            meta = CheckpointMeta(episode=i_episode, all_episode_rewards=all_episode_rewards)
            torch.save({
                "model_state_dict":     target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "meta":                 meta.model_dump(),
            }, checkpoint_path)
            torch.save(target_net.state_dict(), final_path)
            opp_health_tracker = env.full_health if wins_this_ep > 0 else min_opp_health

        max_reward = max(all_episode_rewards)

    if vec_env is not None:
        vec_env.close()
    writer.close()

    results = ExperimentResults(
        model_name=exp.model_name,
        avg_reward_last_20ep=sum(all_episode_rewards[-20:]) / min(len(all_episode_rewards), 20),
        peak_reward=max(all_episode_rewards),
        num_train_hyperparameters=len(cfg.model_fields),
        wall_time_per_episode_sec=total_wall_time / max(ep_cnt, 1),
        vram_peak_gb=vram_peak_gb,
        num_params_million=num_params_M,
        total_episodes=ep_cnt,
    )
    results_path.write_text(results.model_dump_json(indent=2))

    print(f"\nTraining complete. Duration: {(time.time() - training_start) / 60:.1f} minutes")
    print(results.model_dump_json(indent=2))

    sys.stdout = sys.__stdout__
    _log_fh.close()
    print(f"Log written to {log_file_path}")
