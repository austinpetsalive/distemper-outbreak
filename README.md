# Distemper Outbreak Simulation
This repo contains models for optimizing animal placement in kennels in shelters at risk of a distemper outbreak.

The model uses a non-deterministic automata to update the local status for each kennel in a simulated shelter. Intervention objects can then operate on the kennel structure at any time point in the simulation to try out organizational strategies based on visible knowledge about the simulation state. The architecture of the simulation can be seen here:

![Architecture of Distemper Model](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/Distemper%20Model.png?raw=true)

## Parameters

| Parameter Name            | Description                                                                                                                                       | Type/Units                 | Hypothesized Value (or Range) | Source                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------: | ----------------------------: | ------------------------------------------------------------------- |
| pSusceptibleIntake        | The probability that an animal intakes as Susceptible to the virus                                                                                | probability/hour           | ???                           | APA Vaccination Statistics                                          |
| pInfectIntake             | The probability that an animal intakes as Infected with the virus (but not Symptomatic)                                                           | probability/hour           | ???                           | Latent Variable Analysis of Vaccination Papers                      |
| pSymptomaticIntake        | The probability that an animal intakes as Infected and Symptomatic with the virus                                                                 | probability/hour           | ???                           | APA Intake Statistics/Hypothetical                                  |
| pInsusceptibleIntake      | The probability that an animal intakes as Fully Vaccinated/Insusceptible                                                                          | probability/hour           | ???                           | APA Intake Statistics                                               |
| pSurviveInfected          | The probability that an animal will survive (become Insusceptible) given they are Infected (but not Symptomatic)                                  | probability/hour           | ???                           | Latent Variable Analysis of Vaccination Papers                      |
| pSurviveSymptomatic       | The probability that an animal will survive (become Insusceptible) given they are Symptomatic                                                     | probability/hour           | ???                           | APA Survival Statistics                                             |
| pDieAlternate             | The probability that an animal will die of alternate causes                                                                                       | probability/hour           | ???                           | APA Survival Statistics                                             |
| pDischarge                | The probability that an animal will be discharged given it is Insusceptible                                                                       | probability/hour           | ???                           | Shelter-Specific/Hypothetical                                       |
| pCleaning                 | The probability that a kennel will be cleaned given the presence of a Deceased animal                                                             | probability/hour           | ???                           | Shelter-Specific/Hypothetical                                       |
| pSymptomatic              | The probability that an animal will become symptomatic given they are Infected                                                                    | probability/hour           | ???                           | Latent Variable Analysis of Vaccination Papers                      |
| pDie                      | The probability that an animal will die given they are Symptomatic                                                                                | probability/hour           | ???                           | Literature and APA Statistics                                       |
| refractoryPeriod          | The minimum time required for particular state transitions such as Symptomatic->Deceased and Infected->Symptomatic to have a non-zero probability | number of hours            | ???                           | Literature Review                                                   |
| infection_kernel          | The probability of infection given a distance in kennel connections                                                                               | list of probabilities      | ???                           | Literature Review/APA Kennel-Specific Statistics                    |
| infection_kernel_function | A function that determines how immunity impacts infection rate given kernel probabilities from adjacent kennels                                   | string lambda function     | k*(1-immunity)                | Purely Hypothetical                                                 |
| immunity_growth_factors   | Either a pair of floating point values specifying an iterative exponential growth function's parameters or a lookup table of values for each hour | list of 0-1 bounded values | ???                           | Estimated From Vaccination Papers/Hypothetical                      |
| immunity_lut              | A flag specifying if immunity growth will be exponential or follow a lookup table of values for each hour                                         | boolean                    | True                          | True for lookup table function, False for exponential growth factor |
| max_time                  | The maximum time the simulation will run before ending automatically                                                                              | number of hours            | 744                           | 1 Month Estimate                                                    |
| max_intakes               | The maximum number of intakes before the simulation ends automatically                                                                            | number of animals          | None                          | N/A                                                                 |

## Parameter Estimation

| Meta-Parameter Name            | Description                                                                                                                                       | Type/Units                 | Hypothesized Value (or Range) |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------: | ----------------------------: |
| ![equation](https://latex.codecogs.com/gif.latex?T) | A time period in which to compute the meta-parameters (and, therefore, the regular parameters) | hours | 744 (i.e. 1 month) |
| ![equation](https://latex.codecogs.com/gif.latex?N_{T}) | The number of animals in a given time period, T | count | ??? |
| ![equation](https://latex.codecogs.com/gif.latex?S_{T}) | The number of animals in a given time period, T, who came in with distemper symptoms | count | ??? |
| ![equation](https://latex.codecogs.com/gif.latex?V_{T}) | The number of animals in a given time period, T, who came in confirmed fully vaccinated | count | ??? |
| ![equation](https://latex.codecogs.com/gif.latex?d_{T}) | The number of animals in a given time period, T, who died due to reasons other than distemper | count | ??? |
| ![equation](https://latex.codecogs.com/gif.latex?D_{T}) | The number of animals in a given time period, T, who died due to distemper | count | ??? |
| ![equation](https://latex.codecogs.com/gif.latex?G_{T}) | The number of animals in a given time period, T, who were symptomatic of distemper but cleared the illness | count | ??? |

| Parameter Name            | Estimation Method                                                                    | Type/Units                 | Hypothesized Value (or Range) |
| ------------------------- | ------------------------------------------------------------------------------------ | :------------------------: | ----------------------------: |
| pSusceptibleIntake        | ???                                                                                  | probability/hour           | ???                           |
| pInfectIntake             | ???                                                                                  | probability/hour           | ???                           |
| pSymptomaticIntake        | ![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{S_{T}}{N_{T}})^{T^{-1}}) | probability/hour           | ???                           |
| pInsusceptibleIntake      | ![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{V_{T}}{N_{T}})^{T^{-1}}) | probability/hour           | ???                           |
| pSurviveInfected          | ???                                                                                  | probability/hour           | ???                           |
| pSurviveSymptomatic       | ![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{G_{T}}{N_{T}})^{T^{-1}}) | probability/hour           | ???                           |
| pDieAlternate             | ![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{d_{T}}{N_{T}})^{T^{-1}}) | probability/hour           | ???                           |
| pDischarge                | N/A                                                                                  | probability/hour           | 0.0 (unused)                  |
| pCleaning                 | N/A                                                                                  | probability/hour           | 0.75                          |
| pSymptomatic              | ???                                                                                  | probability/hour           | ???                           |
| pDie                      | ![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{D_{T}}{N_{T}})^{T^{-1}}) | probability/hour           | ???                           |
| refractoryPeriod          | ???                                                                                  | number of hours            | ???                           |
| infection_kernel          | ???                                                                                  | list of probabilities      | ???                           |
| infection_kernel_function | N/A                                                                                  | string lambda function     | k*(1-immunity)                |
| immunity_growth_factors   | ![equation](https://latex.codecogs.com/gif.latex?x=[0...T]\mapsto\frac{1}{1+e^{-a*(x-b)}}) for a,b fit from immunity data                                                                                  | list of 0-1 bounded values | ???                           |
| immunity_lut              | N/A                                                                                  | boolean                    | True                          |
| max_time                  | N/A                                                                                  | number of hours            | 744                           |
| max_intakes               | N/A                                                                                  | number of animals          | None                          |

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
