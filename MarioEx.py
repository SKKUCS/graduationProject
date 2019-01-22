import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym_super_mario_bros
import matplotlib.pyplot as plt
# %matplotlib inline

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import RIGHT_ONLY
import math
#import random

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
def downsample(img):
    return img[48:216:2, 44:212:2]
def expand_dimension(img):
    return np.expand_dims(img, axis=2)
def preprocess(img):
    return expand_dimension(to_grayscale(downsample(img)))

try:
    xrange = xrange
except:
    xrange = range

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, s_size2, a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        #WA : changed number of parameter from the originial code,
        #self.state_in= tf.placeholder(shape=[None,s_size, s_size2],dtype=tf.float32, name = 'agent_state_in')
        self.state_in = tf.placeholder(tf.float32, [None, 84, 84, 1], name = 'state_in')
        W1 = tf.Variable(tf.random_normal([8, 8, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.state_in, W1, strides=[1, 4, 4, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        W2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        W4 = tf.Variable(tf.random_normal([11 * 11 * 128, 512], stddev=0.01))
        L4 = tf.reshape(L3, [-1, 11 * 11 * 128])
        L4 = tf.matmul(L4, W4)
        L4 = tf.nn.relu(L4)
        W5 = tf.Variable(tf.random_normal([512, 5], stddev=0.01))
        self.output = tf.nn.softmax(tf.nn.relu(tf.matmul(L4, W5)))
        #L4 = tf.nn.dropout(L4, keep_prob)
        #hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        #self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name = 'reward')
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name = 'action')

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

tf.reset_default_graph() #Clear the Tensorflow graph.

myAgent = agent(lr=1e-2,s_size=84,s_size2=84, a_size=5,h_size=8) #Load the agent.
#WA : changed the parameter from original code

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        s = preprocess(s)
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            #Probabilistically pick an action given our network outputs.
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            #print(np.shape(a_dist)) #print shape for check
            a = np.random.choice(a_dist[0], p = a_dist[0])
            a = np.argmax(a_dist == a)

            s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
            s1 = preprocess(s1)
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            if d == True:
                #Update the network.
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_lenght.append(j)
                break
            env.render()


            #Update our running tally of scores.
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1
