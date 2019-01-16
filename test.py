from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import REALLY_RIGHT_ONLY
import math
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, REALLY_RIGHT_ONLY)

done = True
lastx = 1
cnt = 1
lastreward = 0
life = 2
    for step in range(1000):
	if done:
	    state = env.reset()
	    life = 2
    state, reward, done, info = env.step(env.action_space.sample())
    x = info.get('x_pos')
    if life != info.get('life'):
        cnt += 1
    life = info.get('life')
    print(cnt, end = '    ')
    print(lastreward, end = '   ')
    print(lastx, end = '   ')
    print(step)
    lastreward = reward
    lastx = x
    
    #lastlife = life
    env.render()
"""
    if lastlife != life:
        print(lastreward, end = '   ')
        print(lastx, end = '   ')
        print(step)
"""
env.close()
