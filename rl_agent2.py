import numpy as np
import pandas as pd
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents import *
from rl.policy import *
from rl.memory import *

from distemper import Distemper

import tensorflow as tf
from keras import backend as K


"""
03/01/2019 Test stats over 30 episodes

Mean
Total Intake      971.666667
Total Infected    143.866667
Infection Rate      0.148088
dtype: float64

Std
Total Intake      2.702206
Total Infected    2.158508
Infection Rate    0.002245
dtype: float64
Index(['Total Intake', 'Total Infected', 'Infection Rate'], dtype='object')
"""

def main(try_load_model=True):      
    num_cores = 4
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, 
                            allow_soft_placement=True,
                            device_count = {'GPU' : 0}
                           )

    session = tf.Session(config=config)
    K.set_session(session)

    # Get the environment and extract the number of actions available in the Cartpole problem
    env = Distemper()
    #env = gym.make('CartPole-v1')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    def agent(states, actions):
        model = Sequential()
        model.add(Flatten(input_shape = (1, states)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model
      
    model = agent(env.observation_space.n, env.action_space.n)
    print(model.summary())

    policy = MaxBoltzmannQPolicy()
    test_policy = GreedyQPolicy()
    rl_agent = SARSAAgent(model = model, policy = policy, test_policy = test_policy, nb_actions = env.action_space.n)
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
        rl_agent.load_weights('sarsa_weights.h5f')
    else:
        rl_agent.fit(env, nb_steps=10000, visualize=False, verbose=1)
    
    # Test
    m, s, c = _get_nice_display_results(rl_agent, env, runs=4)
    print(m), print(s), print(c)

    rl_agent.save_weights('sarsa_weights.h5f', overwrite=True)
    
if __name__ == "__main__":
    main(try_load_model=False)