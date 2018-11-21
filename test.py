from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import RIGHT_ONLY
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)

done = True
xpos = 0
for step in range(1000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    temp = info.get('x_pos')
    if temp > xpos:
        xpos = temp
        print(xpos, end = '   ')
        print(step)
    env.render()

env.close()
