import gym
import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from src.actions import FOR_DEBUG
from src.actions import RIGHT_ONLY

env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = BinarySpaceToDiscreteSpaceEnv(env, FOR_DEBUG)
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
ACTION_SIZE = 5
env.reset()
goal_steps = 50
score_requirement = 5
initial_games = 20



def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):
    return img[48:216:2, 44:212:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))


def build_model(input__shape, output_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu',input_shape=input__shape))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), len(training_data[1][0]), 1)
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input__shape=np.shape(X[0]), output_size=len(y[0]))
    model.fit(X, y, epochs=10)
    return model


def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, ACTION_SIZE)
            observation, reward, done, info = env.step(action)
            observation = preprocess(observation)
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            previous_observation = observation
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = [0 for _ in range(ACTION_SIZE)]
                output[data[1]] = 1
                training_data.append([data[0], output])
        env.reset()
    print(accepted_scores)
    return training_data


training_data = model_data_preparation()
trained_model = train_model(training_data)


scores = []
choices = []
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, ACTION_SIZE)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, 84, 84, 1))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        new_observation = preprocess(new_observation)
        prev_obs = new_observation
        score += reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores) / len(scores))
for i in range(ACTION_SIZE):
    print('choice {}:{}'.format(i, choices.count(i) / len(choices)),end="  ")

'''
for step_index in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("Step {}:".format(step_index))
    print("action: {}".format(action))
    print("observation: {}".format(observation))
    print("reward: {}".format(reward))
    print("done: {}".format(done))
    print("info: {}".format(info))
    if done:
        break
env.close()
'''