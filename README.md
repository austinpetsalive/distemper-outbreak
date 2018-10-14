# Distemper Outbreak Simulation
This repo contains models for optimizing animal placement in kennels in shelters at risk of a distemper outbreak.

The model uses a non-deterministic automata to update the local status for each kennel in a simulated shelter. Intervention objects can then operate on the kennel structure at any time point in the simulation to try out organizational strategies based on visible knowledge about the simulation state. The architecture of the simulation can be seen here:

![Architecture of Distemper Model](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/Distemper%20Model.png?raw=true)

## Parameters

| Parameter Name            | Description                                                                                                                                       | Type/Units                 | Hypothesized Value (or Range) | Source                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------: | ----------------------------: | ------------------------------------------------------------------- |
| pSusceptibleIntake        | The probability that an animal intakes as Susceptible to the virus                                                                                | probability/hour           | ???                           | N/A                                                                 |
| pInfectIntake             | The probability that an animal intakes as Infected with the virus (but not Symptomatic)                                                           | probability/hour           | ???                           | N/A                                                                 |
| pSymptomaticIntake        | The probability that an animal intakes as Infected and Symptomatic with the virus                                                                 | probability/hour           | ???                           | N/A                                                                 |
| pInsusceptibleIntake      | The probability that an animal intakes as Infected and Symptomatic with the virus                                                                 | probability/hour           | ???                           | N/A                                                                 |
| pSurviveInfected          | The probability that an animal will survive (become Insusceptible) given they are Infected (but not Symptomatic)                                  | probability/hour           | ???                           | N/A                                                                 |
| pSurviveSymptomatic       | The probability that an animal will survive (become Insusceptible) given they are Symptomatic                                                     | probability/hour           | ???                           | N/A                                                                 |
| pDieAlternate             | The probability that an animal will die of alternate causes                                                                                       | probability/hour           | ???                           | N/A                                                                 |
| pDischarge                | The probability that an animal will be discharged given it is Insusceptible                                                                       | probability/hour           | ???                           | N/A                                                                 |
| pCleaning                 | The probability that a kennel will be cleaned given the presence of a Deceased animal                                                             | probability/hour           | ???                           | N/A                                                                 |
| pSymptomatic              | The probability that an animal will become symptomatic given they are Infected                                                                    | probability/hour           | ???                           | N/A                                                                 |
| pDie                      | The probability that an animal will die given they are Symptomatic                                                                                | probability/hour           | ???                           | N/A                                                                 |
| refractoryPeriod          | The minimum time required for particular state transitions such as Symptomatic->Deceased and Infected->Symptomatic to have a non-zero probability | number of hours            | ???                           | N/A                                                                 |
| infection_kernel          | The probability of infection given a distance in kennel connections                                                                               | list of probabilities      | ???                           | N/A                                                                 |
| infection_kernel_function | A function that determines how immunity impacts infection rate given kernel probabilities from adjacent kennels                                   | string lambda function     | k*(1-immunity)                | N/A                                                                 |
| immunity_growth_factors   | Either a pair of floating point values specifying an iterative exponential growth function's parameters or a lookup table of values for each hour | list of 0-1 bounded values | ???                           | N/A                                                                 |
| immunity_lut              | A flag specifying if immunity growth will be exponential or follow a lookup table of values for each hour                                         | boolean                    | True                          | True for lookup table function, False for exponential growth factor |
| max_time                  | The maximum time the simulation will run before ending automatically                                                                              | number of hours            | 744                           | 1 Month Estimate                                                    |
| max_intakes               | The maximum number of intakes before the simulation ends automatically                                                                            | number of animals          | None                          | N/A                                                                 |

# Demonstration

Using some probabilities which have not been verified, we can see how the simulation performs. 

## Single Simulation
When run in single simulation mode, visualizations of the temporal progress of the disease can be seen. Aggregate variable graphs are shown as follows:

![Video of Graphs](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/graphs.gif?raw=true)

Additionally, a simulation of the kennel network states can be seen here:

![Video of Simulation](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/sim.gif?raw=true)

## Batch Simulation
Finally, when run in batch mode, different strategies can be compared which intervene in position. Here, animals are sorted by immunity to avoid infection of new dogs. This intervention is compared to no intervention:
![Comparison of Methods](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/Figure_1.png?raw=true)
