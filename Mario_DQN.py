import gym
import random
import numpy as np
from collections import deque
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from src.actions import FOR_DEBUG
from src.actions import COMPLEX_MOVEMENT


state_size = [84, 84, 1]
action_size = 9
EPISODES = 100


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):
    return img[48:216:2, 44:212:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(-1, 84, 84, 1))
        return np.argmax(act_values[0])  # returns action

    def train(self, training_data):
        x = np.array([i[0] for i in training_data]).reshape(-1, 84, 84, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, self.action_size)
        self.model.fit(x, y, epochs=2)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(-1,84,84,1))[0]))
            target_f = self.model.predict(state.reshape(-1, 84,84,1))
            target_f[0][action] = target
            self.model.fit(state.reshape(-1, 84,84,1), target_f, epochs=1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/mario_DQN.h5")
    done = False
    batch_size = 1000

    for e in range(EPISODES):
        print('new epi')
        state = env.reset()
        state = preprocess(state)
        total_reward = 0
        #training_data = []
        #state = np.reshape(state, [1, state_size])
        for time in range(24000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            reward = reward if not done else -10
            total_reward += reward
            #next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, time : {}, total = {}, e: {:.2}"
                      .format(e, EPISODES, time, total_reward, agent.epsilon))
                '''
                for data in agent.memory:
                    output = [0 for _ in range(action_size)]
                    output[data[1]] = 1
                    training_data.append([data[0], output])
                agent.train(training_data)
                '''
                break

        agent.replay(batch_size)

        if e % 1 == 0:
             agent.save("./save/mario_DQN.h5")
