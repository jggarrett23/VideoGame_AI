from gym.envs.registration import register

register(
    id = 'NGame-v0',
    entry_point='gym_NGame.envs:N_Game_Env'
)