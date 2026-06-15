# TODO: Check Custom_DBZ_Game.py to make sure frames are being stacked properly, seems as though only the last frame is being saved

from __future__ import annotations

import json
import math
import random
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


# ---------------------------------------------------------------------------
# Pydantic configs
# ---------------------------------------------------------------------------

class TrainConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 16
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 1000
    tau: float = 0.005
    num_episodes: int = 150
    obs_buffer: int = 4
    replay_capacity: int = 10_000
    img_size: int = 128
    weight_decay: float = 1e-5
    grad_clip: float = 100.0


class ExperimentConfig(BaseModel):
    model_name: str = "cnn_fc"
    load_checkpoint: bool = False
    save_data: bool = False


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

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
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
    duration_loss = criterion(state_duration_preds.squeeze() / 10.0, expected_duration_values / 10.0)
    total_loss = action_loss + duration_loss

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

if __name__ == "__main__":

    cfg = TrainConfig()
    exp = ExperimentConfig(
        model_name="dueling_cnn",
        load_checkpoint=False,
        save_data=False,
    )

    # --- paths ---------------------------------------------------------------
    model_dir      = Path("models") / exp.model_name
    checkpoint_path = model_dir / "checkpoint.pt"
    final_path      = model_dir / "final.pt"
    results_path    = model_dir / "results.json"
    log_dir         = Path("log") / exp.model_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    # --- environment ---------------------------------------------------------
    device = torch.device("cuda")
    env = DBZ_Env(
        game_window_title="Slot: 0",
        observation_size=cfg.img_size,
        observation_buffer_size=cfg.obs_buffer,
        health_threshold=20000,
        full_health=40000,
    )
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

    # --- training loop -------------------------------------------------------
    for i_episode in range(episode_start, episode_start + cfg.num_episodes):
        ep_start = time.time()
        print(f"Episode {i_episode + 1} / {episode_start + cfg.num_episodes}")

        obs, _ = env.reset()
        obs  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0.0
        step_cnt       = 0
        episode_loss   = 0.0
        loss_steps     = 0

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

        ep_wall_time     = time.time() - ep_start
        total_wall_time += ep_wall_time
        avg_reward       = episode_reward / max(step_cnt, 1)
        all_episode_rewards.append(avg_reward)

        if env.player_health > env.opp_health:
            player_wins += 1
        ep_cnt += 1

        eps      = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-steps_done / cfg.eps_decay)
        avg_loss = episode_loss / loss_steps if loss_steps > 0 else 0.0
        moving_avg = sum(all_episode_rewards[-10:]) / min(len(all_episode_rewards), 10)

        writer.add_scalar("reward/episode",           avg_reward,   i_episode)
        writer.add_scalar("reward/moving_avg",        moving_avg,   i_episode)
        writer.add_scalar("loss/policy",              avg_loss,     i_episode)
        writer.add_scalar("epsilon",                  eps,          i_episode)
        writer.add_scalar("timing/episode_wall_time", ep_wall_time, i_episode)
        writer.add_scalar("vram/peak_gb",             vram_peak_gb, i_episode)

        print(f"  Reward: {avg_reward:.4f} | Moving avg: {moving_avg:.4f} | "
              f"Wins: {player_wins} | Losses: {ep_cnt - player_wins} | "
              f"Opp health: {env.opp_health} | Wall time: {ep_wall_time:.1f}s | "
              f"VRAM: {vram_peak_gb:.2f} GB")

        should_save = (
            env.opp_health < opp_health_tracker
            or env.player_health > env.opp_health
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
            opp_health_tracker = env.full_health if env.player_health > env.opp_health else env.opp_health

        max_reward = max(all_episode_rewards)

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
