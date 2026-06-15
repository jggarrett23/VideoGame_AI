import gymnasium as gym
import numpy as np
from scipy.linalg import svd, solve, eig
from utils import optimal_SVHT_coef
from pydmd import DMDc
import torch
import torch.nn as nn
from torch.optim import Adam
import collections
import matplotlib.pyplot as plt
class custom_DMDc:
    def __init__(self, svd_thresh='optimal'):
        self.svd_thresh = svd_thresh
        self.modes = None
        self.amplitudes = None
        self.time = None
        self.dynamics = None
        self.eigenvalues = None

    def fit(self, X, Z):

        self.time = np.arange(X.shape[1])

        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # stack state matrix with input matrix
        Omega = np.concatenate((X1, Z), axis=0)

        # compute svd of input space
        U, Sdiag, Vh = svd(Omega, full_matrices=False)
        V = Vh.T.conj()

        # compute svd of output space
        U2, Sdiag2, _ = svd(X2, full_matrices=False)

        # threshold
        if self.thresh=='optimal':
            beta_ratio = X.shape[0] / X.shape[1]
            if beta_ratio > 1:
                beta_ratio = 1/beta_ratio

            self.svd_thresh = optimal_SVHT_coef(beta_ratio, 0) * np.median(Sdiag)

        if self.thresh:
            r1 = sum(Sdiag > self.thresh)
            r2 = sum(Sdiag2 > self.thresh)

            Util = U[:, :r1]
            Stil = np.diag(Sdiag[:r1])
            Vtil = V[:, :r1]
            Uhat = U2[:, :r2]

        else:
            Util = U
            Stil = np.diag(Sdiag)
            Vtil = V
            Uhat = U2

        Util_1 = Util[:X.shape[0], :]
        Util_2 = Util[X.shape[0]:, :]

        # compute redundant term between Atilde and Btilde
        # remember in matlab B/A = (A'\B')'. So to replicate in python do solve(a.T, b.T).T
        z = solve(Stil.T, (Uhat.T.conj() @ X2 @ Vtil).T).T

        # compute approximation of operators
        Atilde = z * Util_1.T.conj() @ Uhat
        Btilde = z * Util_2.T.conj()

        # decompose Atilde
        D, W = eig(Atilde)

        self.eigenvalues = D

        # compute dynamic modes of A
        Phi = solve(Stil.T, (X2 @ Vtil).T).T * Util_1 @ Uhat @ W

        # initial condition
        b = solve(Phi, X1[:, 0])

        self.modes = Phi
        self.amplitudes = b

        t_pow = np.repeat(np.arange(1, len(self.time) + 1), self.amplitudes.shape[0], 1)
        C = self.amplitudes ** t_pow

        # time dynamics
        self.dynamics = np.diag(self.amplitudes) @ C @ Z
        self.frequencies = abs(np.imag(np.log(np.diag(D)) / np.diff(self.time)[0]) / (2*np.pi))

    def transform(self, time, Z):
        # reconstruct data

        # construct vandermode matrix
        t_pow = np.repeat(np.arange(1, len(time)+1), self.amplitudes.shape[0], 1)
        C = self.amplitudes**t_pow

        # time dynamics
        self.dynamics = np.diag(self.amplitudes) @ C

        return self.modes @ self.dynamics


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = nn.Sequential(
        nn.Linear(in_features=obs_size, out_features=20),
        nn.ReLU(),
        nn.Linear(in_features=20, out_features=n_actions),
        nn.Sigmoid()
    )
    critic = DMDc()
    optimizer = Adam(params=actor.parameters(), lr=1e-3)
    nEpisodes = 1000
    epsilon = 1
    epsilon_delta = 1e-10
    gamma = 0.99
    min_frames = 10
    experience_queue = collections.deque(maxlen=50)

    all_reward = []
    for iEpisode in range(nEpisodes):
        state, _ = env.reset()
        episode_reward = 0.0
        frame_count = 0
        done = 0
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action_probs = actor(torch.Tensor(state).to(torch.float32))
                action = np.random.choice(n_actions, p=np.squeeze(action_probs))

            next_state, reward, done, info, _ = env.step(action)
            epsilon -= epsilon_delta

            # point of cartpole is to keep going as long as possible
            # so make reward reflect duration
            episode_reward += reward

            experience_queue.append((state[:, None], action, episode_reward, done, next_state[:, None]))

            if frame_count >= min_frames:

                states, actions, rewards, dones, next_states = zip(*[experience_queue[i] for i in range(len(experience_queue))])
                states = np.hstack(states)
                next_states = np.hstack(next_states)
                rewards = np.asarray(rewards)[None, :]
                actions = np.asarray(actions)
                u = np.vstack((states, actions))
                critic.fit(rewards, u[:, :-1])

                # critic loss
                critic_eigs = critic.eigs ** (frame_count+1)
                A = np.linalg.multi_dot(
                    [critic.modes, np.diag(critic_eigs), np.linalg.pinv(critic.modes)]
                )
                next_states_values = A.dot(rewards[:, 1:]) + critic._B.dot(np.vstack((next_states[:, 1:], actions[1:])))
                target_values = rewards[:, 1:] + gamma * (1 - done) * next_states_values
                predicted_values = A.dot(rewards) + critic._B.dot(np.vstack((states, actions)))
                advantages = target_values - predicted_values[:, 1:]
                critic_loss = np.square(predicted_values[:, 1:] - target_values).mean()

                # actor loss
                action_probs = actor(torch.Tensor(states).to(torch.float32).T)
                chosen_action_probs = (action_probs * torch.Tensor(actions).unsqueeze(1)).sum(axis=1)
                log_probs = torch.log(chosen_action_probs)
                actor_loss = -torch.mean(log_probs[1:] * torch.Tensor(advantages[0, :]).to(torch.float32))

                optimizer.zero_grad()
                actor_loss.backward()
                optimizer.step()

            frame_count += 1
            state = next_state

        print(f'Episode: {iEpisode + 1}, Reward: {episode_reward}')
        all_reward.append(episode_reward)
    env.close()

    all_reward = np.asarray(all_reward)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, nEpisodes+1), all_reward)
    plt.show()
