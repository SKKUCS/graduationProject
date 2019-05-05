from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import REALLY_RIGHT_ONLY
import math
import random
import numpy as np
from PIL import Image

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
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
env.reset()
max_x = 0
cnt = 1
lastreward = 0
life = 2
for step in range(300):
    state, reward, done, info = env.step(1)
    env.render()
    print(np.random.rand())
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = np.reshape(a, (1, 3, 3, 1))
    b = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
    b = np.reshape(b, (1, 3, 3, 1))
    c = np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]])
    c = np.reshape(c, (1, 3, 3, 1))
    d = np.array([[31, 32, 33], [34, 35, 36], [37, 38, 39]])
    d = np.reshape(d, (1, 3, 3, 1))
    e = np.array([[41, 42, 43], [44, 45, 46], [47, 48, 49]])
    e = np.reshape(e, (1, 3, 3, 1))
    f = np.array([[51, 52, 53], [54, 55, 56], [57, 58, 59]])
    f = np.reshape(f, (1, 3, 3, 1))
    history = np.stack((a, b, c, d), axis=2)
    history = np.reshape([history], (1, 3, 3, 4))

    history2 = np.append(e, history[:, :, :, :3], axis=3)
    history3 = np.append(f, history2[:, :, :, :3],  axis=3)


    if done:
        print(a)
        print(history)
        print(history2)
        print(history3)
        print(reward)
        img = state
        img = Image.fromarray(state, 'RGB')
        img.save('ori.png')
        img1 = preprocess(state)
        img1 = Image.fromarray(img1)
        img1.save('grayscale1.png')
        img2 = preprocess2(state)
        img2 = Image.fromarray(img2)
        img2.save('grayscale2.png')
        break
"""
    img = state
    img = Image.fromarray(state, 'RGB')
    img.save('my.png')
    img.show()

    x = info.get('x_pos')

    #print(state)
    #print(np.shape(state))
    #state = preprocess(state)
    #print(state)
    #print(np.shape(state))

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
    #env.render()

    if lastlife != life:
        print(lastreward, end = '   ')
        print(lastx, end = '   ')
        print(step)
"""
env.close()
