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
#from src.actions import FOR_DEBUG
from src.actions import REALLY_COMPLEX_MOVEMENT



action_size = 12
EPISODES = 50000
memory_len = 200000
replay_start = 10000
global_step = 0
max_decay_step = 1000000
def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):
    return img[47:223:2, 0:256:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))



class DQNAgent:
    def __init__(self, action_size):
        self.render = True
        self.load_model = True
        self.state_size = (88, 128, 4)
        self.action_size = action_size
        self.memory = deque(maxlen=memory_len)
        self.gamma = 0.99    # discount rate
        self.epsilon_max = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_now = lambda global_step: self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1. * global_step / max_decay_step)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # Applying tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/mario_DQN', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save/mario_DQN.h5")

    #customized optimizer to use huber loss
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

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        #model.add(Dense(action_size, activation='linear'))
        model.add(Dense(action_size))
        #model.compile(loss='mse', optimizer=Adam())
        return model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, history, action, reward, next_history, done):
        self.memory.append((history, action, reward, next_history, done))

    def act(self, history, epsilon):
        history = np.float32(history / 255.0)
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(history)
        return np.argmax(act_values[0])  # returns action
    '''
    def train(self, training_data):
        x = np.array([i[0] for i in training_data]).reshape(-1, 84, 84, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, self.action_size)
        self.model.fit(x, y, epochs=2)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    '''
    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            done.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if done[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.gamma * \
                            np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
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
        print('Episode #', e)
        state = env.reset()
        state = preprocess(state)
        step, total_reward = 0, 0
        done = False
        #training_data = []
        #state = np.reshape(state, [1, state_size])
        for _ in range(5):
            start, _, _, _ = env.step(0)
        start = preprocess(start)
        history = np.stack((start, start, start, start), axis = 2)
        history = np.reshape([history], (1, 88, 128,4))
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
            step_reward = 0
            epsilon = agent.epsilon_now(global_step)
            action = agent.act(history, epsilon)
            for _ in range(4):
            	next_state, reward, done, _ = env.step(action)
            	step_reward += reward
            	total_reward += reward
            	if done:
            		break
            next_state = np.reshape(preprocess(next_state), (1,88,128,1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            
            total_reward += reward
            #next_state = np.reshape(next_state, [1, state_size])
            agent.remember(history, action, step_reward, next_history, done)
            if len(agent.memory) > replay_start:
                agent.replay()
                #print('now replaying, mem ', len(agent.memory))
            history = next_history
            if done:
                if global_step >=2000:
                    stats = [total_reward, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  total reward :", total_reward, "  memory length:",
                      len(agent.memory), "  epsilon:", epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))
                agent.avg_q_max, agent.avg_loss = 0, 0
                '''
                print("episode: {}/{}, time : {}, total = {}, e: {:.2}"
                      .format(e, EPISODES, time, total_reward, agent.epsilon))
                for data in agent.memory:
                    output = [0 for _ in range(action_size)]
                    output[data[1]] = 1
                    training_data.append([data[0], output])
                agent.train(training_data)
                break
                '''


        if e % 100 == 1:
             agent.save("./save/mario_DQN.h5")
             print('weight saved')
