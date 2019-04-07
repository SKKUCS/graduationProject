import gym_super_mario_bros
from src.actions import REALLY_COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, LSTM, Reshape
#from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading
import random
import time
import gym



def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def crop_img(img):
    return img[47:223:2, 0:256:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(crop_img(img)))


action_size = 13
# Initialize Global Variables
global episode
episode = 0
global thread_count
thread_count = 1
EPISODES = 8000000




# Global Network
class A3CAgent:
    def __init__(self, action_size):
        self.state_size = (88, 128, 3)
        self.action_size = action_size
        # A3C Parameters
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        # Number of Threads
        self.threads = 2
        # Loading model weights
        self.if_load_model = True
        # Building Global Network
        self.actor, self.critic = self.build_model()

        if self.if_load_model:
            try:
                self.load_model('./save/Mario_LSTM')
                print('Weight Loaded')
            except FileNotFoundError:
                print('Weight not found')

        # Customized optimizer for entropy calculation
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # Applying Tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/Mario_LSTM', self.sess.graph)

    # Activating thread for training
    def train(self):
        # Initializing agents
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # Activating each agents
        for agent in agents:
            time.sleep(1)
            agent.start()

        # Save model every 10 min(60 sec)
        while True:
            time.sleep(60 * 10)
            self.save_model("./save/Mario_LSTM")
            print('weight saved:', time.localtime())

    # Building policy and value network
    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        # Simplifying network from original DQN
        #conv = Conv2D(128, (2, 2), strides=(1, 1), activation='relu')(conv)

        conv = Flatten()(conv)
        # Simplifying network from original DQN
        #fc = Dense(512, activation='relu')(conv)
        fc = Dense(256)(conv)
        rs = Reshape((1, 256))(fc)

        lstm = LSTM(256)(rs)
        # Using softmax function to make probability
        policy = Dense(self.action_size, activation='softmax')(lstm)
        value = Dense(1)(lstm)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # Making predicting function
        # According to textbook, this is not for training itself
        # For preventing error in multi-threading
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # Customized optimizer for updating policy network
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # Policy cross-entropy loss function
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # Entropy loss for continuous exploration
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # Final loss function using both entropy
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01, clipnorm=40)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # Customized optimizer for updating value network
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # Square of [Return - Value] for loss function
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # Recording training information each episode
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op




# Actor-Runner class(Thread)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        # Inherit from A3CAgent
        global thread_count
        self.thread_count = thread_count
        thread_count += 1
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor

        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # Memory for samples, emptied every t_max time steps
        self.states, self.actions, self.rewards = [], [], []

        # Building local network
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # Model update rate
        self.t_max = 30
        self.t = 0

    def run(self):
        global episode
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = BinarySpaceToDiscreteSpaceEnv(env, REALLY_COMPLEX_MOVEMENT)
        step = 0

        while episode < EPISODES:
            done = False

            max_x = 40
            no_progress = 0
            score = 0
            state = env.reset()

            '''
            # Making initial history with random actions
            # Seems to be not need in LSTM
            for _ in range(5):
                next_state = state
                state, _, _, _ = env.step(random.randint(0, 12))
            '''
            state = crop_img(state)
            state = np.reshape([state], (1, 88, 128, 3))

            while not done:
                # Rendering code
                # Seems to be causing error in Mac OS
                if self.thread_count==1:
                    env.render()
                step += 1
                self.t += 1

                action, policy = self.get_action(state)

                # Taking 3 steps with selected action
                # Mimicking frame skip
                for _ in range(3):
                    next_state, reward, done, info = env.step(action)
                    if done:
                        break

                # Kill Mario if Mario is making no progress for 10 seconds
                x_now = info.get('x_pos')
                # Handling exception x_pos = 65535
                if x_now == 65535:
                    x_now = max_x
                if max_x < x_now:
                    max_x = x_now
                    no_progress = 0
                else:
                    no_progress += 1
                if no_progress == 300:
                    done = True
                    reward -= 1
                    print("#",self.thread_count, " STUCK")
                # Preprocessing each states
                #next_state = crop_img(next_state)
                next_state = np.reshape([crop_img(next_state)], (1, 88, 128, 3))


                # Average policy max value
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(state / 255.)))

                score += reward

                # Appending sample
                state = next_state
                self.append_sample(state, action, reward)
                if self.t >= self.t_max or done:
                #if done:
                    self.train_model(done)
                    self.update_local_model()
                    #self.reset_lstm_state()
                    self.t = 0

                if done:
                    # Recording training information

                    episode += 1
                    print("#", self.thread_count, "  episode:", episode, "  score:", format(score, '.2f'), "  step:",
                          step, "max_x :", max_x)

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # Calculating discounted prediction for future reward
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1] / 255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # Update networks
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), 88, 128, 3))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.local_critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []


    # Building local networks
    def build_local_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        # Simplifying network from original DQN
        #conv = Conv2D(128, (2, 2), strides=(1, 1), activation='relu')(conv)
        conv = Flatten()(conv)
        # Simplifying network from original DQN
        #fc = Dense(512, activation='relu')(conv)
        fc = Dense(256)(conv)
        rs = Reshape((1, 256))(fc)
        lstm = LSTM(256)(rs)
        policy = Dense(self.action_size, activation='softmax')(lstm)
        value = Dense(1)(lstm)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        # Synchronizing with global network
        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

    # Synchronizing with global network
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # Selecting action from output of policy network
    def get_action(self, state):
        state = np.float32(state / 255.)
        policy = self.local_actor.predict(state)[0]
        np.random.seed(random.randint(0, 100))
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # Appending sample
    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


if __name__ == "__main__":
    global_agent = A3CAgent(action_size)
    global_agent.train()