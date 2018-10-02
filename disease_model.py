import networkx

import epydemic

import epyc
import math
import numpy
import pickle
from copy import copy

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
import seaborn

class SynchronousDynamics(epydemic.Dynamics):
    '''A dynamics that runs synchronously in discrete time, applying local
    rules to each node in the network. These are simple to understand and
    simple to code for many cases, but can be statistically inexact and slow
    for large systems.'''

    # additional metadata
    TIMESTEPS_WITH_EVENTS = 'timesteps_with_events'  #: Metadata element holding the number timesteps that actually had events occur within them

    def __init__( self, g = None, tick_callback = None):
        '''Create a dynamics, optionally initialised to run on the given prototype
        network.
        
        :param g: prototype network to run over (optional)'''
        self.tick_callback = tick_callback
        super(SynchronousDynamics, self).__init__(g)

    def do( self, params):
        '''Synchronous dynamics.
        
        :param params: the parameters of the simulation
        :returns: a dict of experimental results'''
        model = self._model
        # run the dynamics
        g = self.network()
        t = 0
        events = 0
        timestepEvents = 0
        while not self.at_equilibrium(t):            
            # retrieve all the events, their loci, probabilities, and event functions
            dist = self.eventDistribution(t)

            # run through all the events in the distribution
            nev = 0
            for (l, p, ef) in dist:
                if p > 0.0:
                    # run through every possible element on which this event may occur
                    for e in copy(l.elements()):
                        # test for occurrance of the event on this element
                        if numpy.random.random() <= p:
                            # yes, perform the event
                            print('performing event' + str(ef))
                            ef(self, t, g, e)
                            
                            # update the event count
                            nev = nev + 1

            # add the events to the count
            events = events + nev
            if nev > 0:
                # we had events happen in this timestep
                timestepEvents = timestepEvents + 1
            try:
                self.tick_callback(self, t, events, timestepEvents, g, params)
            except:
                pass
            # advance to the next timestep
            t = t + 1

        # add some more metadata
        (self.metadata())[self.TIME] = t
        (self.metadata())[self.EVENTS] = events
        (self.metadata())[self.TIMESTEPS_WITH_EVENTS] = timestepEvents

        # report results
        rc = self.experimentalResults()
        return rc


class CompartmentedSynchronousDynamics(SynchronousDynamics):
    '''A :term:`synchronous dynamics` running a compartmented model. The
    behaviour of the simulation is completely described within the model
    rather than here.'''
        
    def __init__( self, m, g = None, tick_callback = None ):
        '''Create a dynamics over the given disease model, optionally
        initialised to run on the given prototype network.
        
        :param m: the model
        :param g: prototype network to run over (optional)'''
        super(CompartmentedSynchronousDynamics, self).__init__(g, tick_callback)
        self._model = m

    def setUp( self, params ):
        '''Set up the experiment for a run. This performs the default action
        of copying the prototype network and then builds the model and
        uses it to initialise the nodes into the various compartments
        according to the parameters.

        :params params: the experimental parameters'''
        
        # perform the default setup
        super(CompartmentedSynchronousDynamics, self).setUp(params)

        # build the model
        self._model.reset()
        self._model.build(params)

        # initialise the network from the model
        g = self.network()
        self._model.setUp(self, g, params)

    def eventDistribution( self, t ):
        '''Return the model's event distribution.

        :param t: current time
        :returns: the event distribution'''
        return self._model.eventDistribution(t)

    def experimentalResults( self ):
        '''Report the model's experimental results.

        :returns: the results as seen by the model'''
        return self._model.results(self.network())