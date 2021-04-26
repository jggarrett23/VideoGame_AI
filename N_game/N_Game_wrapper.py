import cv2
import gym
import gym.spaces
import numpy as np
import collections

"""
wrapper combines the repetitions of actions during K frames and pixels
from two consecutive frames
"""


class MaxAndSkipEnv(gym.Wrapper):

    def __init__(self, env=None, skip=4):

        """Return only ever 'skip'-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


"""
wrapper converts input observations to a grayscale 84x84 image
"""


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(84, 84, 1),
                                                dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        # shape of image from custom n game env
        img = np.reshape(frame, [84, 84, 3]).astype(np.uint8)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_resized_frame = cv2.resize(img_gray, (84, 84), interpolation=cv2.INTER_AREA)

        processed_frame = np.reshape(gray_resized_frame, [84, 84, 1])
        return processed_frame


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[-1],
                                      old_shape[0], old_shape[1]),
            dtype=np.float32)

    def observation(self, observation):

        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
                                    dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self,observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env,4)
    return ScaledFloatFrame(env)
