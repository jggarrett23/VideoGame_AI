from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from Custom_DBZ_Game import DBZ_Env


class VectorizedDBZEnv:
    """Runs N DBZ_Env instances in parallel using threads.

    Each env must be attached to a separate PCSX2 process. The envs are
    distinguished by env_idx (0..N-1), which selects the nth window matching
    game_window_title via EnumWindows.

    Usage:
        vec = VectorizedDBZEnv(num_envs=2, env_kwargs={
            "game_window_title": "Slot: 0",
            "observation_size": 128,
            ...
        })
        obs_list = vec.reset()                        # list of N observations
        results  = vec.step(actions, durations)       # list of N step tuples
        vec.close()
    """

    def __init__(self, num_envs: int, env_kwargs: dict) -> None:
        self.num_envs = num_envs
        self.envs = [DBZ_Env(**env_kwargs, env_idx=i) for i in range(num_envs)]
        self.executor = ThreadPoolExecutor(max_workers=num_envs)

    def reset(self) -> list:
        futures = {self.executor.submit(e.reset): i for i, e in enumerate(self.envs)}
        results = [None] * self.num_envs
        for f in as_completed(futures):
            results[futures[f]] = f.result()[0]
        return results

    def step(self, actions: list, durations: list,
             active: list[bool] | None = None) -> list[tuple]:
        if active is None:
            active = [True] * self.num_envs
        e0 = self.envs[0]
        _obs_shape = (e0.observation_buffer_size, e0.observation_height, e0.observation_width)
        _placeholder = (np.zeros(_obs_shape, dtype=np.float32), 0.0, True, False, {})

        futures = {
            self.executor.submit(e.step, a, d): i
            for i, (e, a, d, act) in enumerate(zip(self.envs, actions, durations, active))
            if act
        }
        results = [_placeholder] * self.num_envs
        for f in as_completed(futures):
            results[futures[f]] = f.result()
        return results

    def close(self) -> None:
        self.executor.shutdown(wait=False)

    @property
    def opp_health(self) -> list[int]:
        return [e.opp_health for e in self.envs]

    @property
    def player_health(self) -> list[int]:
        return [e.player_health for e in self.envs]

    @property
    def full_health(self) -> int:
        return self.envs[0].full_health
