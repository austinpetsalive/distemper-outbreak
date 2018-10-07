"""This module is the top-level simulation.
"""

from copy import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import simulation


def main(batch=False):
    '''This main function allows quick testing of the batch and non-batch versions
    of the simulation.

    Keyword Arguments:
        batch {bool} -- if True, the simulation will run a batch experiment (default: {False})
    '''

    # Note: all probabilities are in units p(event) per hour
    params = {
        # Intake Probabilities (Note, 1-sum(these) is probability of no intake)
        'pSusceptibleIntake': 0.125,
        'pInfectIntake': 0.02,
        'pSymptomaticIntake': 0.01,
        'pInsusceptibleIntake': 0.75,

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
        'immunity_growth_factors': [1.03, 0.001], # (1.03, 0.001 represents full immunity in 5 days)

        # End Conditions
        'max_time': 31*24, # One month
        'max_intakes': None,

        # Intervention
        'intervention': None # 'SortIntervention()' # Different interventions can go here
        }
    if not batch:
        sim = simulation.Simulation(params,
                                    spatial_visualization=True,
                                    return_on_equillibrium=False,
                                    aggregate_visualization=True)
        print(sim.run())
    else:
        # Run batch simulation comparing interventions
        runs = 32
        bar_width = 0.35
        proportion = True
        colors = [cm.jet(0), cm.jet(0.5)] #pylint: disable=E1101
        alphas = [0.5, 0.25]
        labels = ['Sort Intervention', 'No Intervention']

        params1 = copy(params)
        params1['intervention'] = None

        results = simulation.BatchSimulation(params, runs).run()

        total = sum(list(results[0].values()))
        results_dataframe = pd.DataFrame.from_records(results)
        if proportion:
            results_dataframe /= total

        plt.rcdefaults()

        objects = results_dataframe.columns
        y_pos = np.arange(len(objects))

        plt.bar(y_pos-bar_width/2,
                results_dataframe.mean(),
                bar_width,
                align='center',
                alpha=alphas[0],
                yerr=results_dataframe.std()/np.sqrt(len(results_dataframe)),
                color=colors[0],
                label=labels[0])

        results = simulation.BatchSimulation(params1, runs).run()
        results_dataframe = pd.DataFrame.from_records(results)
        if proportion:
            results_dataframe /= total
        plt.bar(y_pos+bar_width/2,
                results_dataframe.mean(),
                bar_width,
                align='center',
                alpha=alphas[1],
                yerr=results_dataframe.std()/np.sqrt(len(results_dataframe)),
                color=colors[1],
                label=labels[1])

        plt.xticks(y_pos, objects)
        plt.ylabel('Mean Animal Count')
        plt.ylim(0, 1)
        plt.title('Average Simulation Performance')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    main(batch=True)
