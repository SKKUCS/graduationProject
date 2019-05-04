import gym
import math, random
import numpy as np
import tensorflow as tf
from collections import deque
from keras import backend as K
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam, RMSprop
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from src.actions import REALLY_COMPLEX_MOVEMENT


action_size = 12
EPISODES = 50000
memory_len = 200000
replay_start = 20000
global_step = 0
max_decay_ep = 10000

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def to_grayscale2(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    #grayscale with ITU-R 601-2 luma transformation
def downsample(img):
    return img[47:223:2, 0:256:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))
def preprocess2(img):
    return expand_dimension(to_grayscale2(downsample(img)))



class DQNAgent:
    def __init__(self, action_size):
        self.render = True
        self.load_model = False
        self.state_size = (88, 128, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_len)
        self.gamma = 0.9    # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_now = lambda episode: self.epsilon_min + \
                                     (self.epsilon_max - self.epsilon_min) * math.exp(-1. * episode / max_decay_ep)
        self.learning_rate = 0.00025
        self.batch_size = 32
        self.update_freq = 1000
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.optimizer = self.optimizer()

        # Applying tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/SMB_DDQN', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save/SMB_DDQN.h5")
            print("Weight Loaded!")

        self.update_target_model()

    # Customized optimizer for Huber loss calculation
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=self.learning_rate, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(action_size))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    def act(self, history, epsilon):
        _history = np.float32(history / 255.0)
        _epsilon = epsilon
        if np.random.rand() <= _epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(history)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        replay_history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        replay_next_history = np.zeros((self.batch_size, 1, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        replay_target = np.zeros((self.batch_size,))
        replay_action, replay_reward, replay_done= [], [], []
        target_action = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            replay_history[i] = np.float32(mini_batch[i][0] / 255.)
            replay_next_history[i] = np.float32(mini_batch[i][3] / 255.)
            replay_action.append(mini_batch[i][1])
            replay_reward.append(mini_batch[i][2])
            replay_done.append(mini_batch[i][4])
            target_act_values = self.model.predict(replay_next_history[i])
            target_action[i] = np.argmax(target_act_values[0])

        for i in range(self.batch_size):
            target_value_list = self.target_model.predict(replay_next_history[i])
            target_value = target_value_list[0][int(target_action[i])]
            if replay_done[i]:
                replay_target[i] = replay_reward[i]
            else:
                replay_target[i] = replay_reward[i] + self.gamma * \
                            target_value

        loss = self.optimizer([replay_history, replay_action, replay_target])
        self.avg_loss += loss[0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    # Applying tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


if __name__ == "__main__":
   
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, REALLY_COMPLEX_MOVEMENT)
    agent = DQNAgent(action_size)

    total_rewards, episodes = [], []

    for e in range(EPISODES):
        state = env.reset()
        step, total_reward = 0, 0
        done = False
        for _ in range(4):
            start, _, _, _ = env.step(0)
        start = preprocess2(start)
        start = np.reshape(start, (1, 88, 128, 1))
        history = np.stack((start, start, start, start), axis=3)
        history = np.reshape([history], (1, 88, 128, 4))
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
            step_reward = 0
            epsilon = agent.epsilon_now(e)
            action = agent.act(history, epsilon)
            for _ in range(4):
                next_state, reward, done, _ = env.step(action)
                step_reward += reward
                total_reward += reward
                if done:
                    break

            next_state = np.reshape(preprocess2(next_state), (1, 88, 128, 1))
            next_history = np.append(history[:, :, :, 1:4], next_state, axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            agent.remember(history, action, step_reward, next_history, done)

            history = next_history

            if len(agent.memory) > replay_start:
                agent.replay()

            if done:
                if global_step >= 2000:
                    stats = [total_reward, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("Episode #", e, "  total reward :", format(total_reward, '.2f'), "  memory length:",
                      len(agent.memory), "  epsilon:", format(epsilon, '.4f'),
                      "  global_step:", global_step, "  average_q:",
                      format(agent.avg_q_max / float(step), '.4f'), "  average loss:",
                      format(agent.avg_loss / float(step), '.4f'))

                agent.avg_q_max, agent.avg_loss = 0, 0
                # agent.update_target_model()

            if global_step % agent.update_freq == 1:
                agent.update_target_model()
                # print("target updated")

        if e % 100 == 1:
            agent.save("./save/SMB_DDQN.h5")
            print('Weight Saved!')
