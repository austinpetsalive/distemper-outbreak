# Distemper Outbreak Simulation
This repo contains models for optimizing animal placement in kennels in shelters at risk of a distemper outbreak.

The model uses a non-deterministic automata to update the local status for each kennel in a simulated shelter. Intervention objects can then operate on the kennel structure at any time point in the simulation to try out organizational strategies based on visible knowledge about the simulation state. The architecture of the simulation can be seen here:

![Architecture of Distemper Model](https://github.com/austinpetsalive/distemper-outbreak/blob/master/media/Distemper%20Model.png?raw=true)

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
