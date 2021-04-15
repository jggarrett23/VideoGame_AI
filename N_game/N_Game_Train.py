import N_Game_wrapper as wrappers
import PyTorch_DeepQ as dqn_model

import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import gym_NGame
import argparse
import sys

game_name = 'NGame-v0'
game_path = 'D:\\Nv2-PC.exe'
model_dir = 'D:\\VideoGame_AI\\N_game\\models\\'
model_name = 'NGame_32Cnn_512Linear_Torch'
log_dir = 'D:\\VideoGame_AI\\N_game\\log\\' + model_name
checkpoint_dir = 'D:\\VideoGame_AI\\N_game\\chck_points\\'

# Settings for the Deep Q learning
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
NUM_EPISODES = 2

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action',
                               'reward', 'done', 'new_state']
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones,
                                                                                                  dtype=np.uint8), np.array(
            next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment

        new_state, reward, is_done, _ = self.env.step(action)

        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)

        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def save_checkpoint(state, checkpoint_dir, reward, model_dir):
    chck_path = checkpoint_dir + 'checkpoint.pt'
    torch.save(state, chck_path)
    full_model_name = model_name + '_muReward%.2f' % reward
    model_savePath = model_dir + full_model_name
    torch.save(state['state_dict'], model_savePath + "-best.dat")


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['mean_reward']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=game_name,
                        help="Name of the environment, default=" + game_name, type=str)
    parser.add_argument("--nEpisodes", type=int, default=NUM_EPISODES,
                        help='Number of episodes to run training, default=%d' % NUM_EPISODES)

    parser.add_argument("--load_best", default=False, action="store_true",
                        help="Load Best Model")
    parser.add_argument("--checkpoint_path", required="--load_best" in sys.argv, type=str,
                        help='Path to best Model')

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using: {device}")

    NUM_EPISODES = args.nEpisodes

    env = wrappers.make_env(args.env)

    if args.env == game_name:
        env.start_game(game_path)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    load_episode = 0

    best_mean_reward = None

    if args.load_best:
        net, optimizer, load_episode, best_mean_reward = load_checkpoint(args.checkpoint_path,
                                                                         net, optimizer)
        print('Model Checkpoint Loaded')

    writer = SummaryWriter(logdir=log_dir, comment='-' + args.env)
    print(f"Network Structure:\n{net}\n")

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()


    for iEpsiode in range(NUM_EPISODES):
        running_current_ep = True
        time.sleep(.2)
        while running_current_ep:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward = agent.play_step(net, epsilon, device=device)

            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon, speed
                ))

                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("mean_reward", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)

                if best_mean_reward is None or best_mean_reward < mean_reward:

                    checkpoint = {
                        'epoch': iEpsiode + 1 + load_episode,
                        'mean_reward': mean_reward,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    save_checkpoint(checkpoint, checkpoint_dir, mean_reward, model_dir)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (
                            best_mean_reward, mean_reward
                        ))
                    best_mean_reward = mean_reward
                reward = None
                running_current_ep = False

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            optimizer.step()
    env.close()
    writer.close()
