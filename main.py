"""This module is the top-level simulation.
"""
import os

from copy import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import json

from copy import deepcopy

import tqdm
import simulation

import pickle as pkl
import argparse

from multiprocessing.pool import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--mode", dest="mode", default="visual", help="can be 'batch' 'visual' or 'stats'")
args = parser.parse_args()

def stat_sim(params):
    sim = simulation.Simulation(params,
                                spatial_visualization=False,
                                aggregate_visualization=False,
                                return_on_equillibrium=True,)
    results = []
    # Manually reimplement the run loop so we can intercept the state over time
    sim.running = True
    while sim.running:
        sim.update()
        results.append((sim.disease.time, sim._get_disease_stats()))
    return results

def main(mode='visual'):
    '''This main function allows quick testing of the batch and non-batch versions
    of the simulation.

    Keyword Arguments:
        batch {bool} -- if True, the simulation will run a batch experiment (default: {False})
    '''
    np.random.seed(1234)
    
    if os.path.exists('./realistic_sim_params.json'):
        with open('./realistic_sim_params.json') as f:
            params = json.load(f)
            print("Loaded ./realistic_sim_params.json"+"-"*30)

    else:
        # Note: all probabilities are in units p(event) per hour
        params = {
            # Intake Probabilities (Note, 1-sum(these) is probability of no intake)
            'pSusceptibleIntake': 0.125,
            'pInfectIntake': 0.02,
            'pSymptomaticIntake': 0.01,
            'pInsusceptibleIntake': 0.05,

            # Survival of Illness
            'pSurviveInfected': 0.025,
            'pSurviveSymptomatic': 0.025,

            # Alternate Death Rate
            'pDieAlternate': 0.001,

            # Discharge and Cleaning
            'pDischarge': 0.05,
            'pCleaning': 0.9,

            # Disease Refractory Period
            'refractoryPeriod': 3.0*24.0,

            # Death and Symptoms of Illness
            'pSymptomatic': 0.04,
            'pDie': 0.05,

            # Infection Logic
            'infection_kernel': [0.05, 0.01],
            'infection_kernel_function': 'lambda node, k: k*(1-node[\'occupant\'][\'immunity\'])',

            # Immunity Growth (a0*immunity+a1)
            # (1.03, 0.001 represents full immunity in 5 days)
            #'immunity_growth_factors': [1.03, 0.001],
            'immunity_growth_factors': [0.0114, 0.0129, 0.0146, 0.0166, 0.0187, 0.0212, 0.0240,
                                        0.0271, 0.0306, 0.0346, 0.0390, 0.0440, 0.0496, 0.0559,
                                        0.0629, 0.0707, 0.0794, 0.0891, 0.0998, 0.1117, 0.1248,
                                        0.1392, 0.1549, 0.1721, 0.1908, 0.2109, 0.2326, 0.2558,
                                        0.2804, 0.3065, 0.3338, 0.3623, 0.3918, 0.4221, 0.4530,
                                        0.4843, 0.5157, 0.5470, 0.5779, 0.6082, 0.6377, 0.6662,
                                        0.6935, 0.7196, 0.7442, 0.7674, 0.7891, 0.8092, 0.8279,
                                        0.8451, 0.8608, 0.8752, 0.8883, 0.9002, 0.9109, 0.9206,
                                        0.9293, 0.9371, 0.9441, 0.9504, 0.9560, 0.9610, 0.9654,
                                        0.9694, 0.9729, 0.9760, 0.9788, 0.9813, 0.9834, 0.9854,
                                        0.9871, 0.9886],
            'immunity_lut': True,

            # End Conditions
            'max_time': 31*24, # One month
            'max_intakes': None,

            # Intervention
            'intervention': 'TimedRemovalIntervention()' # Different interventions can go here
            }
        with open('./sim_params.json', 'w+') as out:
            json.dump(params, out)
            
    if mode == 'visual':
        print(params['intervention'])
        sim = simulation.Simulation(params,
                                    spatial_visualization=True,
                                    aggregate_visualization=True,
<<<<<<< Updated upstream
                                    return_on_equillibrium=True,)
=======
									return_on_equillibrium=False,)
>>>>>>> Stashed changes
        print(sim.run())
    elif mode == 'stats':
        runs = 30
        run_results = []
        with Pool(8) as thread_pool:
            tasks = tqdm.tqdm(thread_pool.imap_unordered(stat_sim,
                                                         [deepcopy(params) for _ in
                                                          range(0, runs)]),
                              total=runs)
            for i in tasks:
                run_results.append(i)
            thread_pool.close()
            thread_pool.join()
        with open('stats.pkl', 'wb') as fp:
            pkl.dump(run_results, fp)
    elif mode == 'batch':
        # Run batch simulation comparing interventions
        runs = 30
        bar_width = 0.25
        colors = [cm.jet(0), cm.jet(0.33), cm.jet(0.66)] #pylint: disable=E1101
        alphas = [0.75, 0.5, 0.25]
        labels = ['Room Lock Intervention', 'Snake Intervention', 'No Intervention']
        print(params)
        params['intervention'] = 'RoomLockIntervention()'

        params1 = copy(params)
        params1['intervention'] = 'SnakeIntervention()'

        params2 = copy(params)
        params2['intervention'] = 'TimedRemovalIntervention()'

        def _get_nice_display_results(_p):
            print(_p['intervention'])
            results = simulation.BatchSimulation(_p, runs).run()
            results_dataframe = pd.DataFrame.from_records(results)
            results_dataframe = results_dataframe.drop([col for col in results_dataframe.columns if col != "E" and col != "I"], axis=1)
            results_dataframe = results_dataframe.rename(index=str,
                                                         columns={"E": "Total Intake",
                                                                  "I": "Total Infected"})
            results_dataframe['Infection Rate'] = \
                results_dataframe['Total Infected'] / results_dataframe['Total Intake']
            means = results_dataframe.mean()
            stes = results_dataframe.std() / np.sqrt(len(results_dataframe))
            cols = results_dataframe.columns

            return means, stes, cols

        plt.rcdefaults()

        m_0, s_0, c_0 = _get_nice_display_results(params)
        m_1, s_1, c_1 = _get_nice_display_results(params1)
        m_2, s_2, c_2 = _get_nice_display_results(params2)
        assert all(c_0 == c_1) and all(c_1 == c_2), "columns mismatch"

        means = np.transpose([m_0, m_1, m_2])
        stes = np.transpose([s_0, s_1, s_2])

        objects = labels
        y_pos = np.arange(len(objects))

        _, axs = plt.subplots(1, 3, figsize=(9, 4), sharey=False)

        axs[0].bar(y_pos-bar_width*1.1,
                   means[0],
                   bar_width,
                   align='center',
                   alpha=alphas[0],
                   yerr=stes[0],
                   color=colors[0],
                   label=c_0[0])

        axs[1].bar(y_pos,
                   means[1],
                   bar_width,
                   align='center',
                   alpha=alphas[1],
                   yerr=stes[1],
                   color=colors[1],
                   label=c_0[1])

        axs[2].bar(y_pos+bar_width*1.1,
                   means[2],
                   bar_width,
                   align='center',
                   alpha=alphas[2],
                   yerr=stes[2],
                   color=colors[2],
                   label=c_0[2])

        plt.sca(axs[0])
        plt.xticks(y_pos, objects, rotation=30, ha='right')
        plt.ylabel('Mean Animal Count')
        plt.title('Total Intakes')
        plt.sca(axs[1])
        plt.xticks(y_pos, objects, rotation=30, ha='right')
        plt.ylabel('Mean Animal Count')
        plt.title('Total Infected')
        plt.sca(axs[2])
        plt.xticks(y_pos, objects, rotation=30, ha='right')
        plt.ylabel('Mean Infection Rate')
        plt.title('Infection Rate')

        plt.suptitle(f'Average Simulation Performance (n={runs})')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


if __name__ == '__main__':
<<<<<<< Updated upstream
    assert args.mode in ['visual', 'batch', 'stats']
    #main(mode=args.mode)
    main(mode='stats')
=======
    main(batch=False)#args.use_batch)
>>>>>>> Stashed changes
