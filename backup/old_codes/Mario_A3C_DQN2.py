import gym_super_mario_bros
from src.actions import REALLY_COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, concatenate
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
def downsample(img):
    return img[47:223:2, 0:256:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))

action_size = 13
# Initialize Global Variables
global episode
episode = 0
global thread_num
thread_num = 1
EPISODES = 8000000




# Global Network
class A3CAgent:
    def __init__(self, action_size):
        self.state_size = (88, 128, 4)
        self.action_size = action_size
        # A3C Parameters
        self.discount_factor = 0.99
        self.actor_lr = 0.0017
        self.critic_lr = 0.0017
        # Number of Threads
        self.threads = 8
        # Loading model weights
        self.if_load_model = False
        # Building Global Network
        self.actor, self.critic = self.build_model()



        # Customized optimizer for entropy calculation
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # Applying Tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        if self.if_load_model:
            self.load_model('./save/Mario_A3C_DQN_2')
            print('Weight Loaded')
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/Mario_A3C_DQN_2', self.sess.graph)

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
            self.save_model("./save/Mario_A3C_DQN_2")
            print('weight saved:', time.localtime())

    # Building policy and value network
    def build_model(self):
        input = Input(shape=self.state_size)
        conv1 = Conv2D(64, (1, 1), padding = 'same', activation='relu')(input)
        conv1 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(conv1)
        # Simplifying network from original DQN
        #conv = Conv2D(128, (2, 2), strides=(1, 1), activation='relu')(conv)
        conv2 = Conv2D(32, (1, 1), padding = 'same', activation='relu')(input)
        conv2 = Conv2D(32, (5, 5), padding = 'same', activation='relu')(conv2)

        conv3 = MaxPooling2D((3, 3), strides=(1, 1), padding = 'same')(input)
        conv3 = Conv2D(64, (1, 1), padding = 'same', activation='relu')(conv3)

        concat = concatenate([conv1, conv2, conv3], axis = 3)

        conv = Flatten()(concat)
        # Simplifying network from original DQN
        #fc = Dense(512, activation='relu')(conv)
        fc = Dense(256)(conv)

        # Using softmax function to make probability
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1)(fc)

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
        global thread_num
        self.thread_num = thread_num
        thread_num += 1
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
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
        env = BinarySpaceToDiscreteSpaceEnv(env, REALLY_COMPLEX_MOVEMENT)
        step = 0

        while episode < EPISODES:
            done = False

            max_x = 40
            no_progress = 0
            score = 0
            state = env.reset()

            # Making initial history with random actions
            for _ in range(5):
                next_state = state
                state, _, _, _ = env.step(0)

            state = preprocess(state)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 88, 128, 4))

            while not done:
                # Rendering code
                # Seems to be causing error in Mac OS
                if self.thread_num==1:
                    env.render()
                step += 1
                self.t += 1

                step_reward = 0

                action, policy = self.get_action(history)

                # Taking 3 steps with selected action
                # Mimicking frame skip
                for _ in range(6):
                    next_state, reward, done, info = env.step(action)
                    score += reward
                    step_reward += reward
                    if done:
                        break

                # Kill Mario if Mario is making no progress for 10 seconds
                x_now = info.get('x_pos')
                # Handling exception x_pos = 65535
                if x_now == 65535:
                    x_now = max_x
                if max_x <= x_now:
                    max_x = x_now
                    no_progress = 0
                else:
                    no_progress += 1
                if no_progress == 150:
                    done = True
                    reward -= 1
                    step_reward -= 1
                    score -= 1
                    print("#",self.thread_num, " STUCK")

                # Preprocessing each states
                next_state = preprocess(next_state)
                next_state = np.reshape([next_state], (1, 88, 128, 1))
                next_history = np.append(next_state, history[:, :, :, :3],
                                         axis=3)

                # Average policy max value
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(history / 255.)))


                # Appending sample
                self.append_sample(history, action, step_reward)
                history = next_history
                if self.t >= self.t_max or done:
                #if done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    # Recording training information

                    episode += 1
                    print("#", self.thread_num, "  episode:", episode, "  score:", format(score, '.2f'), "  step:",
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

        states = np.zeros((len(self.states), 88, 128, 4))
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
        conv1 = Conv2D(64, (1, 1), padding = 'same', activation='relu')(input)
        conv1 = Conv2D(64, (3, 3), padding = 'same', activation='relu')(conv1)
        # Simplifying network from original DQN
        #conv = Conv2D(128, (2, 2), strides=(1, 1), activation='relu')(conv)
        conv2 = Conv2D(32, (1, 1), padding = 'same', activation='relu')(input)
        conv2 = Conv2D(32, (5, 5), padding = 'same', activation='relu')(conv2)

        conv3 = MaxPooling2D((3, 3), strides=(1, 1), padding = 'same')(input)
        conv3 = Conv2D(64, (1, 1), padding = 'same', activation='relu')(conv3)

        concat = concatenate([conv1, conv2, conv3], axis = 3)

        conv = Flatten()(concat)
        # Simplifying network from original DQN
        #fc = Dense(512, activation='relu')(conv)
        fc = Dense(256)(conv)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1)(fc)

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
    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_actor.predict(history)[0]
        np.random.seed(random.randint(0, 100))
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # Using Epsilon-greedy method, but seems to be not needed
    def get_action2(self, history):
        history = np.float32(history / 255.0)
        policy = self.local_actor.predict(history)[0]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), policy
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy  # returns action

    # Appending sample
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


if __name__ == "__main__":
    global_agent = A3CAgent(action_size)
    global_agent.train()