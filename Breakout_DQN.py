from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym
import math

from PIL import Image


EPISODES = 50000
def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def to_grayscale2(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    #grayscale with ITU-R 601-2 luma transformation
def downsample(img):
    return img[0:210:2, 0:160:2]
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
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.max_decay_ep = 10000
        self.epsilon_now = lambda episode: self.epsilon_min + \
                                           (self.epsilon_max - self.epsilon_min) * math.exp(
            -1. * episode / self.max_decay_ep)
        self.batch_size = 32
        self.train_start = 20000
        self.update_target_rate = 1000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=200000)
        self.no_op_steps = 30
        self.learning_rate = 0.00025
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/Breakout_DQN', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save/Breakout_DQN.h5")

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
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history, epsilon):
        history = np.float32(history / 255.0)
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))


    def train_model(self):

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, 1, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []
        target_action = np.zeros((self.batch_size))

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
            target_act_values = self.model.predict(next_history[i])
            target_action[i] = np.argmax(target_act_values[0])

        for i in range(self.batch_size):
            target_value_list = self.target_model.predict(next_history[i])
            target_value = target_value_list[0][int(target_action[i])]
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        target_value

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

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


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe[32:200, 8:152]), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)



        state = pre_processing(observe)

        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            epsilon = agent.epsilon_now(e)
            action = agent.get_action(history, epsilon)
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(history[:, :, :, 1:4], next_state, axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            agent.append_sample(history, action, reward, next_history, dead)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                history = next_history

            if done:
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("# ", e, "  Score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", format(epsilon, '.4f'),
                      "  global_step:", global_step, "  average_q:",
                      format(agent.avg_q_max / float(step), '.5f'), "  average loss:",
                      format(agent.avg_loss / float(step), '.5f'))

                agent.avg_q_max, agent.avg_loss = 0, 0

        if e % 100 == 1:
            agent.model.save_weights("./save/Breakout_DQN.h5")
            print('Weight Saved!')