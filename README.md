# Distemper Outbreak Simulation
This repo contains models for optimizing animal placement in kennels in shelters at risk of a distemper outbreak.

The model uses a non-deterministic automata to update the local status for each kennel in a simulated shelter. Intervention objects can then operate on the kennel structure at any time point in the simulation to try out organizational strategies based on visible knowledge about the simulation state. The architecture of the simulation can be seen here:

![Architecture of Distemper Model](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/Distemper%20Model.png?raw=true)

## Parameters

| Parameter Name            | Description                                                                                                                                       | Type/Units                 | Hypothesized Value (or Range) | Source                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------: | ----------------------------: | ------------------------------------------------------------------- |
| pSusceptibleIntake        | The probability that an animal intakes as Susceptible to the virus                                                                                | probability/hour           | ???                           | APA Vaccination Statistics                                          |
| pInfectIntake             | The probability that an animal intakes as Infected with the virus (but not Symptomatic)                                                           | probability/hour           | ???                           | Latent Variable Analysis of Vaccination Papers                      |
| pSymptomaticIntake        | The probability that an animal intakes as Infected and Symptomatic with the virus                                                                 | probability/hour           | 0                             | Always 0 Due to Simulation Rules                                    |
| pInsusceptibleIntake      | The probability that an animal intakes as Fully Vaccinated/Insusceptible                                                                          | probability/hour           | ???                           | APA Intake Statistics                                               |
| pSurviveInfected          | The probability that an animal will survive (become Insusceptible) given they are Infected (but not Symptomatic)                                  | probability/hour           | ???                           | Latent Variable Analysis of Vaccination Papers                      |
| pSurviveSymptomatic       | The probability that an animal will survive (become Insusceptible) given they are Symptomatic                                                     | probability/hour           | ???                           | APA Survival Statistics                                             |
| pDieAlternate             | The probability that an animal will die of alternate causes                                                                                       | probability/hour           | ???                           | APA Survival Statistics                                             |
| pDischarge                | The probability that an animal will be discharged given it is Insusceptible                                                                       | probability/hour           | 1                             | Always 1 Due to Simulation Rules                                    |
| pCleaning                 | The probability that a kennel will be cleaned given the presence of a Deceased animal                                                             | probability/hour           | 1                             | Always 1 Due to Simulation Rules                                    |
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
| ![equation](https://latex.codecogs.com/gif.latex?N_{T}) | The number of animals in a given time period, T | count | [847](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Dog-Intakes-Chart/r5dg-6cvf) |
| ![equation](https://latex.codecogs.com/gif.latex?S_{T}) | The number of animals in a given time period, T, who came in with distemper symptoms | count | 0 (assumed) |
| ![equation](https://latex.codecogs.com/gif.latex?V_{T}) | The number of animals in a given time period, T, who came in confirmed fully vaccinated | count | 347 ([1](https://www.sciencedirect.com/science/article/pii/S0378113511006705?via%3Dihub), [2](https://www.tandfonline.com/doi/full/10.1080/10888705.2018.1435281?scroll=top&needAccess=true), [3](https://avmajournals.avma.org/doi/abs/10.2460/javma.236.12.1317)) |
| ![equation](https://latex.codecogs.com/gif.latex?d_{T}) | The number of animals in a given time period, T, who died due to reasons other than distemper | count | [68](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238) |
| ![equation](https://latex.codecogs.com/gif.latex?D_{T}) | The number of animals in a given time period, T, who died due to distemper | count | 111 (Watch+Confirmed Proportion) or 178 (Confirmed Proportion) (APA Internal Data) |
| ![equation](https://latex.codecogs.com/gif.latex?G_{T}) | The number of animals in a given time period, T, who were symptomatic of distemper but cleared the illness | count | 0 (assumed) |

**Rate Averaging Equation**
This equation can be used to take population numbers over a time interval and convert them to probabilities per unit time in that interval.

![equation](https://latex.codecogs.com/gif.latex?1-(1-\frac{S_{T}}{N_{T}})^{T^{-1}})

| Parameter Name            | Estimation Method                                                                    | Type/Units                 | Hypothesized Value (or Range) |
| ------------------------- | ------------------------------------------------------------------------------------ | :------------------------: | ----------------------------: |
| pSusceptibleIntake        | Literature Review of [1](https://www.sciencedirect.com/science/article/pii/S0378113511006705?via%3Dihub), [2](https://www.tandfonline.com/doi/full/10.1080/10888705.2018.1435281?scroll=top&needAccess=true), [3](https://avmajournals.avma.org/doi/abs/10.2460/javma.236.12.1317) to determine average rate of 1-0.41-pInfectIntake in population across 4 shelters, extrapolated to AAC data and applied to equation (1) | probability/hour           | ???                           |
| pInfectIntake             | 0.08 applied to equation (1) based on APA Internal Data on distemper watch or confirm intakes between August 2018 and January 2019 post-outbreak at AAC                                                                                  | probability/hour           | ???                           |
| pSymptomaticIntake        | Assumed 0 | probability/hour           | 0                           |
| pInsusceptibleIntake      | Literature Review of [1](https://www.sciencedirect.com/science/article/pii/S0378113511006705?via%3Dihub), [2](https://www.tandfonline.com/doi/full/10.1080/10888705.2018.1435281?scroll=top&needAccess=true), [3](https://avmajournals.avma.org/doi/abs/10.2460/javma.236.12.1317) to determine average rate of 0.41 in population across 4 shelters, extrapolated to AAC data and applied to equation (1) | probability/hour           | ???                           |
| pSurviveInfected          | Determined via examination of all distemper watch animals at APA for months of August 2018 to January 2019 post-outbreak at AAC and extrapolated to AAC population then applied to equation (1)                                                                                  | probability/hour           | ???                           |
| pSurviveSymptomatic       | Determined via examination of all distemper confirmed animals at APA for months of August 2018 to January 2019 post-outbreak at AAC and extrapolated to AAC population then applied to equation (1) or [0.15](https://www.tandfonline.com/doi/full/10.1080/23737867.2016.1148644) if no treatment is provided | probability/hour           | ???                           |
| pDieAlternate             | Determined via AAC average death rate extrapolated then applied to equation (1) | probability/hour           | ???                           |
| pDischarge                | Assumed 1                                                                                  | probability/hour           | 0.0 (unused)                  |
| pCleaning                 | Assumed 1                                                                                  | probability/hour           | 0.75                          |
| pSymptomatic              | Determined via comparison of distemper exposed evolution in APA animals fro August 2018 to January 2019 post-outbreak at AAC and extrapolated to AAC population then applied to equation 1                                                                                  | probability/hour           | ???                           |
| pDie                      | Determined via 1-pSurviveSymptomatic or [0.85](https://www.tandfonline.com/doi/full/10.1080/23737867.2016.1148644) applied to equation (1) depending on assumptions of treatment (i.e. the former with APA protocols and the latter without) | probability/hour           | ???                           |
| refractoryPeriod          | Assumed 0 days (hypothetically if infection occurs on-site it is [7-14 days](http://veterinarycalendar.dvm360.com/canine-distemper-proceedings), but infection could have occurred before arrival - the simulation does not currently differentiate these situations so no refractor period will be given)                                                                                  | number of hours            | ???                           |
| infection_kernel          | Computed from [0.0053](https://www.tandfonline.com/doi/full/10.1080/23737867.2016.1148644) per day global spread probabilitiy applied to equation (1) in neighbor then inverse square law (due to diffusion) for second neighbor                                                                                  | list of probabilities      | ???                           |
| infection_kernel_function | Inverse square law due to diffusion over [20 foot radius](http://veterinarycalendar.dvm360.com/canine-distemper-proceedings) - assumed to be 2 kennel max distance for convenience (though this should be adjusted for sufficiently tight layouts)                                                                                  | string lambda function     | k*(1-immunity)                |
| immunity_growth_factors   | Assumed to follow linear part of Michaelis-Menten Kinetics with fit to 0 and t=0 and 0.9 at t=72                                                                                  | list of 0-1 bounded values | y=0.0125\*t for t=0..72                           |
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
![Comparison of Methods](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/30_intervention_compairson.png?raw=true)
