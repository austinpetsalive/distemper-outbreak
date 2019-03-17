"""This module is the top-level simulation.
"""
import os

from copy import copy

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import json

import simulation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_batch", action='store_true', help="if True, the simulation will run a batch experiment")
args = parser.parse_args()

    
"""
Final Results
Maximum value: -0.238032
Best parameters:  {'infection_kernel_0': 0.05, 'infection_kernel_1': 0.001, 'pDieAlternate': 0.0005, 'pInfectIntake': 0.01, 'pInsusceptibleIntake': 0.05534796523444544, 'pSusceptibleIntake': 0.1}
Total Intake 847 970.0
E2I 68 55.5
sum_S2D_IS2D 68 15.5
E2S 432 584.0
E2IS 347 330.5
S2I 111 131.5

Final Results
Maximum value: -0.177720
Best parameters:  {'infection_kernel_0': 0.048877811938572956, 'infection_kernel_1': 0.01, 'pDieAlternate': 0.0025, 'pInfectIntake': 0.005, 'pInsusceptibleIntake': 0.025, 'pSusceptibleIntake': 0.05}
Total Intake 847 934.5
E2I 68 59.0
sum_S2D_IS2D 68 81.5
E2S 432 578.0
E2IS 347 297.5
S2I 111 131.5


Final Results
Maximum value: -0.133748
Best parameters:  {'infection_kernel_0': 0.045234156049009, 'infection_kernel_1': 0.01, 'pDieAlternate': 0.0025, 'pInfectIntake': 0.005, 'pInsusceptibleIntake': 0.029339218966702094, 'pSusceptibleIntake': 0.05}
Total Intake 847 941.5
E2I 68 50.5
sum_S2D_IS2D 68 82.0
E2S 432 581.0
E2IS 347 310.0
S2I 111 123.5
"""

best_params = {'infection_kernel_0': 0.045234156049009, 'infection_kernel_1': 0.01, 'pDieAlternate': 0.0025, 'pInfectIntake': 0.005, 'pInsusceptibleIntake': 0.029339218966702094, 'pSusceptibleIntake': 0.05}

def main(batch=False):
    '''This main function allows quick testing of the batch and non-batch versions
    of the simulation.

    Keyword Arguments:
        batch {bool} -- if True, the simulation will run a batch experiment (default: {False})
    '''
    np.random.seed(1234)

    # Note: all probabilities are in units p(event) per hour
    params = {
        # Intake Probabilities (Note, 1-sum(these) is probability of no intake)
        'pSusceptibleIntake': 0.125,
        'pInfectIntake': 0.02,
        'pSymptomaticIntake': 0.0,
        'pInsusceptibleIntake': 0.05,

        # Survival of Illness
        'pSurviveInfected': 0.0,
        'pSurviveSymptomatic': 0.0,

        # Alternate Death Rate
        'pDieAlternate': 0.001,

        # Discharge and Cleaning
        'pDischarge': 0.0,
        'pCleaning': 1.0,

        # Disease Refractory Period
        'refractoryPeriod': 7.0*24.0,

        # Death and Symptoms of Illness
        'pSymptomatic': 0.0,
        'pDie': 0.0,

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
            
    if not batch:
        print(params['intervention'])

        params['pSusceptibleIntake'] = best_params['pSusceptibleIntake']
        params['pInfectIntake'] = best_params['pInfectIntake']
        params['pInsusceptibleIntake'] = best_params['pInsusceptibleIntake']
        params['pDieAlternate'] = best_params['pDieAlternate']
        params['infection_kernel'] = [best_params['infection_kernel_0'], best_params['infection_kernel_1']]
        
        sim = simulation.Simulation(params,
                                    spatial_visualization=True,
                                    aggregate_visualization=True,
                                    return_on_equillibrium=True,)
        print(sim.run())
    else:
        # Run batch simulation comparing interventions
        
        """
        Grid Search Method with Baysian Optimization
        `pSusceptibleIntake`, `pInfectIntake`, `pInsusceptibleIntake`, `pDieAlternate`, and `infection_kernel`
        """
        
        from bayes_opt import BayesianOptimization
        from bayes_opt.observer import JSONLogger
        from bayes_opt.event import Events
        import warnings
        
        log_name = 'APA-XGB_BO-Distemper-03-16-2019-v1'
        logger = JSONLogger(path='./'+log_name+'.json')
        orig_params = params.copy()
        Test = False            
        Target = {
                'Total Intake': 847,
                'E2I':68,
                'sum_S2D_IS2D':68,
                'E2S':432,
                'E2IS':347,
                'S2I':111
                }
        
        
        def _get_results(_p):
            runs = 2
            results = simulation.BatchSimulation(_p, runs).run()
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
            
        def _heuristic(
                        pSusceptibleIntake,
                        pInfectIntake,
                        pInsusceptibleIntake,
                        pDieAlternate,
                        infection_kernel_0,
                        infection_kernel_1
                        ):
            params = orig_params.copy()
            params['pSusceptibleIntake'] = pSusceptibleIntake
            params['pInfectIntake'] = pInfectIntake
            params['pInsusceptibleIntake'] = pInsusceptibleIntake
            params['pDieAlternate'] = pDieAlternate
            params['infection_kernel'] = [infection_kernel_0,infection_kernel_1]
            
            m_0, s_0, c_0 = _get_results(params)
            
            if Test:
                return m_0
            else:
                loss = 0
                for key, value in Target.items():
                    # category-wise normalized L2 loss
                    loss += abs((m_0[key]-value)/value)
                loss /= len(Target)
                    
                return -1.*loss
        
        """
        Desired ouput
        Total Intake = 847
        Empty->Infected = 68, 
        Susceptible->Dead + Insusceptible->Dead = 68, 
        Empty->Susceptible=432, 
        Empty->Insusceptible=347, 
        Susceptible->Infected=111
        
        When
            'pSusceptibleIntake': 0.125,
            'pInfectIntake': 0.02,
            'pSymptomaticIntake': 0.0,
            'pInsusceptibleIntake': 0.05,

            # Survival of Illness
            'pSurviveInfected': 0.0,
            'pSurviveSymptomatic': 0.0,

            # Alternate Death Rate
            'pDieAlternate': 0.001,

            # Discharge and Cleaning
            'pDischarge': 0.0,
            'pCleaning': 1.0,

            # Disease Refractory Period
            'refractoryPeriod': 7.0*24.0,

            # Death and Symptoms of Illness
            'pSymptomatic': 0.0,
            'pDie': 0.0,

            # Infection Logic
            'infection_kernel': [0.05, 0.01],
            'infection_kernel_function': 'lambda node, k: k*(1-node[\'occupant\'][\'immunity\'])',
        We have
        {'E': 987, 'S': 0, 'IS': 511, 'I': 369, 'SY': 0, 'D': 46, 'E2I': 98, 'sum_S2D_IS2D': 26, 'E2S': 623, 'E2IS': 266, 'S2I': 271}
        
        We need to 
            Decrease E2I (↓pInfectIntake)
            Increase sum_S2D_IS2D (↑pDieAlternate)
            Decrese E2S (↓pSusceptibleIntake)
            Increase E2IS (↑pInsusceptibleIntake)
            Decrease S2I (↓infection_kernel)
        """
        BO_wrapper = BayesianOptimization(
                        _heuristic,
                        {
                        'pSusceptibleIntake':(0.05,0.2),
                        'pInfectIntake':(0.005,0.03),
                        'pInsusceptibleIntake':(0.025,0.2),
                        'pDieAlternate':(0.0025,0.01),
                        'infection_kernel_0':(0.01,0.1),
                        'infection_kernel_1':(0.001,0.01)
                        }
        )

        BO_wrapper.subscribe(Events.OPTMIZATION_STEP, logger)

        print('-'*130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            BO_wrapper.maximize(init_points=20, n_iter=50, acq='ei', xi=0.01)
            
        print('-'*130)
        print('Final Results')
        print('Maximum value: %f' % BO_wrapper.max['target'])
        print('Best parameters: ', BO_wrapper.max['params'])
        
        Test = True
        m_0 = _heuristic(**BO_wrapper.max['params'])
        
        for key, value in Target.items():
            print(key, value, m_0[key])

if __name__ == '__main__':
    main(batch=args.use_batch)
