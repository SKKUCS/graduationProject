from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from src.actions import RIGHT_ONLY
env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)

done = True
for step in range(2000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()