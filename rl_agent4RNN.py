import numpy as np
import pandas as pd
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import GRU 
from keras.optimizers import Adam
from rl.agents import *
from rl.policy import *
from rl.memory import *

#from distemper import Distemper as Distemper
from distemper import Distemper2 as Distemper

import tensorflow as tf
from keras import backend as K

"""
	Actionable Items:
	
	Action Space
	State Space
	RL method
	Network Architecture
	Simulation method
"""

def main(try_load_model=True):      
    num_cores = 4
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, 
                            allow_soft_placement=True,
                            device_count = {'GPU' : 1}
                           )

    session = tf.Session(config=config)
    K.set_session(session)

    # Get the environment and extract the number of actions available in the Cartpole problem
    env = Distemper(turn_around_rate=20)
    np.random.seed(1234)
    env.seed(1234)
    nb_actions = env.action_space.n
    batch = 1

    # Build Model
    def agent(states, actions):
        model = Sequential()
        model.add(GRU(64,input_shape=(1,states),recurrent_dropout=.5,return_sequences=True))
        model.add(GRU(128,recurrent_dropout=.5,return_sequences=True))
        model.add(Flatten(input_shape=(1,128)))
        model.add(Dense(actions, activation='linear'))
        return model
      
    model = agent(env.observation_space.n, env.action_space.n)
    print(model.summary())

    policy = MaxBoltzmannQPolicy(eps=.5)
    test_policy = GreedyQPolicy()
    memory = SequentialMemory(limit=10000, window_length=batch)
    rl_agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1024,target_model_update=1e-2, policy=policy, test_policy=test_policy)
    rl_agent.compile(Adam(lr=1e-4), metrics = ['mse'])

    def _get_nice_display_results(rl_agent, env, runs=4):

        results = []
        action_stats = []
        for _ in range(runs):
            env.reset()
            rl_agent.test(env, nb_episodes=1, visualize=False)
            results.append(env.simulation._get_disease_stats())
            action_stats.append(env._get_action_stats())
            print(env._get_action_stats())
            
        results_dataframe = pd.DataFrame.from_records(results)
        results_dataframe = results_dataframe.drop(['S', 'IS', 'SY', 'D'], axis=1)
        results_dataframe = results_dataframe.rename(index=str,
                                                     columns={"E": "Total Intake",
                                                              "I": "Total Infected"})
        results_dataframe['Infection Rate'] = \
            results_dataframe['Total Infected'] / results_dataframe['Total Intake']
        means = results_dataframe.mean()
        stes = results_dataframe.std() / np.sqrt(len(results_dataframe))
        cols = results_dataframe.columns

        return means, stes, cols
  
    # Train
    if try_load_model: 
        rl_agent.load_weights('CNNAgent_weights.h5f')
    else:
        rl_agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
    
    # Test
    m, s, c = _get_nice_display_results(rl_agent, env, runs=4)
    print(m), print(s), print(c)

    rl_agent.save_weights('CNNAgent_weights.h5f', overwrite=True)
    
if __name__ == "__main__":
    main(try_load_model=False)