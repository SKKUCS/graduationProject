from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import RIGHT_ONLY
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)

done = True
lastx = 1
lastlife = 3
lastreward = 0
for step in range(500):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    x = info.get('x_pos')
    life = info.get('life')
    print(lastreward, end = '   ')
    print(lastx, end = '   ')
    print(step)
    lastreward = reward
    lastx = x
    lastlife = life
    env.render()
"""
    if lastlife != life:
        print(lastreward, end = '   ')
        print(lastx, end = '   ')
        print(step)
"""
env.close()
