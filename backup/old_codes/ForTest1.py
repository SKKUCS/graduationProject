from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import REALLY_RIGHT_ONLY
import math
import random
import numpy as np
from PIL import Image
from collections import deque


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
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
env.reset()
max_x = 0
cnt = 1
lastreward = 0
life = 2
memory = deque(maxlen=3)
for step in range(300):
    state, reward, done, info = env.step(1)
    env.render()
    print(np.random.rand())
    a = np.array([[1, 2, 3], [4, 5, 6]])
    a = np.reshape(a, (1, 2, 3, 1))
    b = np.array([[11, 12, 13], [14, 15, 16]])
    b = np.reshape(b, (1, 2, 3, 1))
    c = np.array([[21, 22, 23], [24, 25, 26]])
    c = np.reshape(c, (1, 2, 3, 1))
    d = np.array([[31, 32, 33], [34, 35, 36]])
    d = np.reshape(d, (1, 2, 3, 1))
    e = np.array([[41, 42, 43], [44, 45, 46]])
    e = np.reshape(e, (1, 2, 3, 1))
    f = np.array([[51, 52, 53], [54, 55, 56]])
    f = np.reshape(f, (1, 2, 3, 1))
    history = np.stack((a, b, c, d), axis = 3)
    history = np.reshape([history], (1, 2, 3, 4))
    history2 = np.append(history[:, :, :, 1:4], e, axis=3)
    history3 = np.append(history2[:, :, :, 1:4], f, axis=3)

    if done:
        memory.append((history, reward, '2', history2))
        memory.append((history2, reward, '2', history3))
        replay_history = np.zeros((2, 2, 3, 4))
        for i in range(2):
            replay_history[i] = memory[i][0]
        print(a)
        print(history)
        print(history2)
        print(history3)
        print(reward)
        print(replay_history[0])
        print(replay_history[1])
        img = state
        img = Image.fromarray(state, 'RGB')
        img.save('ori.png')
        img1 = preprocess(state)
        img1 = Image.fromarray(img1)
        img1.save('grayscale1.png')
        img2 = preprocess(state)
        img2 = Image.fromarray(img2)
        img2.save('grayscale2.png')
        test1 = expand_dimension(img1)
        print(test1)
        print(np.shape(test1))
        test2=np.reshape(test1, (1,88,128,1))
        print(test2)
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

