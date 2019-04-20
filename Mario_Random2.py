from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import REALLY_RIGHT_ONLY
import math
import random
import numpy as np
from PIL import Image

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, REALLY_RIGHT_ONLY)


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):
    return img[47:223:2, 0:256:2]
def preprocess(img):
    return to_grayscale(downsample(img))
def to_grayscale2(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    #grayscale with ITU-R 601-2 luma transformation
def preprocess2(img):
    return downsample(to_grayscale2(img))
done = True
max_x = 0
cnt = 1
lastreward = 0
life = 2
for step in range(1):
    if done:
        state = env.reset()
        life = 2
    state, reward, done, info = env.step(random.randrange(0,1))
    '''
    img = state
    img = Image.fromarray(state, 'RGB')
    img.save('my.png')
    img.show()
    '''
    x = info.get('x_pos')

    #print(state)
    print(np.shape(state))
    state = preprocess(state)
    #print(state)
    print(np.shape(state))
    img = state
    img = Image.fromarray(state)
    img.save('my.png')
    img.show()

    if life != info.get('life'):
        cnt += 1
    life = info.get('life')
    if x > max_x:
        max_x = x
        print(cnt, end = '    ')
        print(lastreward, end = '   ')
        print(max_x, end = '   ')
        print(step)
    lastreward = reward
    #lastlife = life
    env.render()
"""
    if lastlife != life:
        print(lastreward, end = '   ')
        print(lastx, end = '   ')
        print(step)
"""
env.close()
