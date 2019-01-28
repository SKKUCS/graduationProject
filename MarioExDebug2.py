import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym_super_mario_bros
import matplotlib.pyplot as plt
# %matplotlib inline

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
#from gym_super_mario_bros.actions import RIGHT_ONLY
from src.actions import FOR_DEBUG
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
env = BinarySpaceToDiscreteSpaceEnv(env, FOR_DEBUG)

gamma = 0.99
global_step = tf.Variable(0, trainable=False, name = 'global_step')

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        #WA : changed number of parameter from the originial code,
        #self.state_in= tf.placeholder(shape=[None,s_size, s_size2],dtype=tf.float32, name = 'agent_state_in')
lr = 0.01
state_in = tf.placeholder(tf.float32, [None, 84, 84, 1], name = 'state_in')
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([8, 8, 1, 32], stddev=0.01, name = 'W1'))
    L1 = tf.nn.conv2d(state_in, W1, strides=[1, 4, 4, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([4, 4, 32, 64], stddev=0.01, name = 'W2'))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01, name = 'W3'))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_normal([11 * 11 * 128, 512], stddev=0.01, name = 'W4'))
    L4 = tf.reshape(L3, [-1, 11 * 11 * 128])
    L4 = tf.matmul(L4, W4)
    L4 = tf.nn.relu(L4)
with tf.name_scope('output'):
    W5 = tf.Variable(tf.random_normal([512, 2], stddev=0.01, name = 'W5'))
    output = tf.nn.softmax(tf.nn.relu(tf.matmul(L4, W5)))
#self.W1_show = W1
    #L4 = tf.nn.dropout(L4, keep_prob)
    #hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
    #self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
chosen_action = tf.argmax(output,1)
    #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
    #to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[None],dtype=tf.float32, name = 'reward')
action_holder = tf.placeholder(shape=[None],dtype=tf.int32, name = 'action')
indexes = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + action_holder
responsible_outputs = tf.gather(tf.reshape(output, [-1]), indexes)
tvars = tf.trainable_variables()
gradient_holders = []
for idx,var in enumerate(tvars):
    placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
    gradient_holders.append(placeholder)
with tf.name_scope('optimizer'):
    loss = -tf.reduce_mean(tf.log(responsible_outputs)*reward_holder)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    update_batch = optimizer.apply_gradients(zip(gradient_holders,tvars))
    gradients = tf.gradients(loss,tvars)
    tf.summary.scalar('loss', loss)

#tf.reset_default_graph() #Clear the Tensorflow graph.

#myAgent = agent(lr=1e-2,s_size=84,s_size2=84, a_size=2,h_size=8) #Load the agent.
#WA : changed the parameter from original code

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()
# Launch the tensorflow graph
sess = tf.Session()
#saver = tf.train.Saver(tf.global_variables())
"""
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(init)
"""
sess.run(init)
writer = tf.summary.FileWriter('./log/20190127-1', sess.graph)
i = 0
total_reward = []
total_lenght = []
gradBuffer = sess.run(tf.trainable_variables())
for ix,grad in enumerate(gradBuffer):
    gradBuffer[ix] = grad * 0
while i < total_episodes:
    s = env.reset()
    s = preprocess(s)
    d = False
    running_reward = 0
    ep_history = []
    for j in range(max_ep):
        if d == True:
            print('handled exception')
            break
        #Probabilistically pick an action given our network outputs.
        merged = tf.summary.merge_all()
        a_dist = sess.run(output, feed_dict={state_in:[s]})
        summary = sess.run(merged, feed_dict={state_in:[s]})
        #print(np.shape(a_dist)) #print shape for check
        writer.add_summary(summary, global_step = sess.run(global_step))
        a = np.random.choice(a_dist[0], p = a_dist[0])
        a = np.argmax(a_dist == a)
        s1,r,d,_ = env.step(a) #Get our reward for taking an action given a bandit.
        s1 = preprocess(s1)
        ep_history.append([s,a,r,s1])
        s = s1
        running_reward += r
        if d == True:
            #Update the network.
            print(W1, ', Updated')
            ep_history = np.array(ep_history)
            ep_history[:,2] = discount_rewards(ep_history[:,2])
            feed_dict={reward_holder:ep_history[:,2],
                    action_holder:ep_history[:,1],state_in:np.vstack(ep_history[:,0])}
            try:
                grads = sess.run(gradients, feed_dict=feed_dict)
            except:
                print('Hmm')
                continue
            for idx,grad in enumerate(grads):
                gradBuffer[idx] += grad
            if i % update_frequency == 0 and i != 0:
                feed_dict= dictionary = dict(zip(gradient_holders, gradBuffer))
                _ = sess.run(update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
            total_reward.append(running_reward)
            total_lenght.append(j)
            break
        env.render()
        #Update our running tally of scores.
    if (i % 10 == 0) and i != 0:
        print(np.mean(total_reward[-100:]))
    i += 1
    #saver = tf.train.Saver(tf.global_variables())
    #saver.save(sess, './model/MarioExDebug.ckpt', global_step=global_step)
