import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from distemper import Distemper
#ENV_NAME = 'CartPole-v0'

import tensorflow as tf
from keras import backend as K

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Get the environment and extract the number of actions available in the Cartpole problem
#env = gym.make(ENV_NAME)
env = Distemper()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, env.observation_space.n)))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)

dqn.test(env, nb_episodes=1, visualize=False)