# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T

#from optimisers import Adam
from utils import createSrng, np_log_mean_exp_stable
from Hybrid_variational_model import Hybrid_variational_model
from MLP_variational_model import MLP_variational_model
from MLP_likelihood_model import MLP_likelihood_model
from Hysteresis_variational_model import Hysteresis_variational_model
from jitterProtect import JitterProtect
from printable import Printable
import time as time
#import collections
#from theano.compile.nanguardmode import NanGuardMode
import lasagne


precision = th.config.floatX

class AutoEncoderModel(Printable):

    def __init__(self,
                 data,
                 params,
                 encoderParameters,
                 decoderParameters,
                 L_terms='Train'):

        theanoRandomSeed = params['theanoRandomSeed']
        numpyRandomSeed = params['numpyRandomSeed']
        self.srng = createSrng(seed=theanoRandomSeed)

        # set the data
        data = np.asarray(data, dtype=precision)                
        if params['BinaryFromContinuous']:
            self.binarise = True
            self.data = th.shared(data, name='data')   
            y_threshold = self.srng.uniform(size=self.data.shape, low=0.0, high=1.0, ndim=None)                     
            self.y = self.data > y_threshold
            self.sample_y_threshold = th.function([], y_threshold)
            if decoderParameters['continuous'] == True:
                raise RuntimeError('Incompatible optimstion BinaryFromContinuous & continuous')
        else:
            self.binarise = False
            self.y = th.shared(data)
            
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.B = params['miniBatchSize']
        self.Q = params['dimZ']
        
        self.numberofBatchesPerEpoch = int(np.ceil(np.float32(self.N) / self.B))
        numPad = self.numberofBatchesPerEpoch * self.B - self.N

        self.batchStream = self.srng.permutation(n=self.N)
        self.padStream   = self.srng.choice(size=(numPad,), a=self.N,
                                       replace=False, p=None, ndim=None, dtype='int32')

        self.batchStream.name = 'batchStream'
        self.padStream.name = 'padStream'

        self.iterator = th.shared(0)
        self.iterator.name = 'iterator'

        self.allBatches = T.reshape(T.concatenate((self.batchStream, self.padStream)), [self.numberofBatchesPerEpoch, self.B])
        self.allBatches.name = 'allBatches'

        self.currentBatch = T.flatten(self.allBatches[self.iterator, :])
        self.currentBatch.name = 'currentBatch'

        self.y_miniBatch = self.y[self.currentBatch, :]
        self.y_miniBatch.name = 'y_miniBatch'

        self.sample_batchStream = th.function([], self.batchStream)
        self.sample_padStream   = th.function([], self.padStream)

        self.lowerBound = -np.inf  # Lower bound
        self.lowerBounds = []

        self.getCurrentBatch = th.function([], self.currentBatch, no_default_updates=True)

        self.jitterProtector = JitterProtect()
        if encoderParameters['Type'] == 'Hybrid':
            self.encoder = Hybrid_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                encoderParameters,
                self.srng,
                self.jitterProtector)
        elif encoderParameters['Type'] == 'MLP':
            self.encoder = MLP_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                encoderParameters,
                self.srng)
        elif encoderParameters['Type'] == 'Hysteresis':
                self.encoder = Hysteresis_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                encoderParameters,
                self.srng)
        else:
            raise RuntimeError('Unrecognised encoder type')

        if decoderParameters['Type'] == 'MLP':
            self.decoder = MLP_likelihood_model(self.y_miniBatch, self.B,
                self.P, self.Q, self.encoder, decoderParameters, self.srng)
        else:
            raise RuntimeError('Unrecognised decoder type')

        self.encoder.construct_L_terms()
        self.decoder.construct_L_terms(self.encoder)

        self.gradientVariables = self.encoder.gradientVariables + self.decoder.gradientVariables

        if L_terms == 'Train':
            self.L = self.encoder.L_terms + self.decoder.L_terms
            self.dL = T.grad(self.L, self.gradientVariables)
            for i in range(len(self.dL)):
                self.dL[i].name = 'dL_d' + self.gradientVariables[i].name
        elif L_terms == 'Test':
            self.L = self.decoder.log_pyz
        else:
            raise RuntimeError('L_terms is in {''Train'',''Test''}')

        # Sample batch before randomisation
        self.sample_batchStream()
        self.sample_padStream()

        # Initialise the variables in the networks
        rnd = np.random.RandomState(seed=numpyRandomSeed)
        self.encoder.randomise(rnd)
        self.decoder.randomise(rnd)

        
    def sample(self):
        self.sample_batchStream()
        self.sample_padStream()
        if self.binarise:
            self.sample_y_threshold()

    def train(self,
        numberOfEpochs=1,
        maxIters=np.inf,
        constrain=False,
        printDiagnostics=0,
        testModel=None,
        numberOfTestSamples=5000
        ):

        if not type(self.encoder) == Hybrid_variational_model:
            constrain = False
            printDiagnostics=0

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...

        print "training for {} epochs".format(numberOfEpochs)

        # pbar = progressbar.ProgressBar(maxval=numberOfIterations*numberOfEpochs).start()

        for ep in range(numberOfEpochs):

            # Sample a new batch
            self.sample()

            for it in range(self.numberofBatchesPerEpoch):

                # Sample from the encoder
                self.encoder.sample()
                self.decoder.sample()
                self.iterator.set_value(it)
                lbTmp = self.jitterProtector.jitterProtect(self.updateFunction, reset=False)
                if constrain:
                    self.encoder.gp_encoder.constrainKernelParameters()

                lbTmp = lbTmp.flatten()
                self.lowerBound = lbTmp[0]

                currentTime  = time.time()
                wallClock    = currentTime - startTime
                stepTime     = wallClock - wallClockOld
                wallClockOld = wallClock

                print("\n Ep %d It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (ep, it, wallClock, stepTime, self.lowerBound))
                if printDiagnostics > 0 and (it % printDiagnostics) == 0:
                    self.encoder.gp_encoder.printDiagnostics()

                self.lowerBounds.append((self.lowerBound, ep, it, wallClock))

                if ep * self.numberofBatchesPerEpoch + it > maxIters:
                    break

            if testModel is not None:
                testModel.copyGradientVariables(self)
                testModel.lowerBounds.append((testModel.MCLogLikelihood(numberOfTestSamples),ep,it))
     
            if ep * self.numberofBatchesPerEpoch + it > maxIters:
                break
            # pbar.update(ep*numberOfIterations+it)
        # pbar.finish()

        return self.lowerBounds

    def constructUpdateFunction(self, learning_rate=0.0001, beta_1=0.99, beta_2=0.999, profile=False):

#        gradColl = collections.OrderedDict([(param, T.grad(self.L, param)) for param in self.gradientVariables])
#
#        self.optimiser = Adam(self.gradientVariables, learning_rate, beta_1, beta_2)
#
#        updates = self.optimiser.updatesIgrad_model(gradColl, self.gradientVariables)
#
#        # Get the update function to also return the bound!
#        self.updateFunction = th.function([],
#                                          self.L,
#                                          updates=updates,
#                                          no_default_updates=True,
#                                          profile=profile),
#                                          #mode=NanGuardMode(nan_is_error=True,
#                                          #                  inf_is_error=True,
#                                          #                  big_is_error=True))
##

        grads = T.grad(-self.L, self.gradientVariables)
        clip_grad = 1
        max_norm = 5
        mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
        cgrads = [T.clip(g,-clip_grad, clip_grad) for g in mgrads]

        updates = lasagne.updates.adam(cgrads, self.gradientVariables, learning_rate=learning_rate)

        self.updateFunction = th.function([],
                                      self.L,
                                      updates=updates,
                                      no_default_updates=True)


    def construct_L_function(self):
        self.L_func = th.function([], self.L, no_default_updates=True)
        
    def construct_dL_function(self):
        self.dL_func = th.function([], self.dL, no_default_updates=True)

    def MCLogLikelihood(self, numberOfSamples=5000):
        
        self.sample()
        ll = [0] * self.numberofBatchesPerEpoch * numberOfSamples
        c = 0
        for i in range(self.numberofBatchesPerEpoch):
            print '{} of {}, {} samples'.format(i,
                                                self.numberofBatchesPerEpoch,
                                                numberOfSamples)
            self.iterator.set_value(i)
            self.jitterProtector.reset()
            for k in range(numberOfSamples):
                self.decoder.sample()
                self.encoder.sample()
                ll[c] = self.jitterProtector.jitterProtect(self.L_func, reset=False)
                c += 1
        return np_log_mean_exp_stable(ll)


    def copyGradientVariables(self, other):
        for i in range(len(self.gradientVariables)):
            if self.gradientVariables[i].name == other.gradientVariables[i].name:
                self.gradientVariables[i].set_value(other.gradientVariables[i].get_value(), borrow=False)
            else:
                raise RuntimeError('Cannot copy gradient variables')


if __name__ == "__main__":

    params = {'miniBatchSize':2, 'dimZ':20}
    data = np.ones((10,2))
    encoderType = 'MLP'
    encoderParameters = {'numHiddenUnits_encoder' : 10, 'numHiddenLayers_encoder' : 1}
    decoderType = 'MLP'
    decoderParameters = {'numHiddenUnits_decoder' : 10, 'numHiddenLayers_decoder' : 1}
    ae = AutoEncoderModel(
                 data,
                 params,
                 encoderType,        #MLP, Hybrid
                 encoderParameters,
                 decoderType,        #MLP_decoder_moder, IBP_factor
                 decoderParameters)

    ae.sample()

    ae.construct_L_dL_functions()







