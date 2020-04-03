import gym
import retro
import numpy as np

env = retro.make('DoubleDunk-Atari2600')

observation = env.reset()
done = False
while not done:
	new_state, reward, done, info = env.step(env.action_space.sample())

	print(f"{info}:{reward}")


env.close()

	