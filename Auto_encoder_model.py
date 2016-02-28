# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T

from optimisers import Adam
from utils import createSrng, np_log_mean_exp_stable
from Hybrid_variational_model import Hybrid_encoder
from MLP_variational_model import MLP_variational_model
from MLP_likelihood_model import MLP_likelihood_model
from Hysterisis_variational_model import Hysterisis_encoder
from jitterProtect import JitterProtect
from printable import Printable
import time as time
import collections

precision = th.config.floatX

class AutoEncoderModel(Printable):

    def __init__(self,
                 data,
                 params,
                 encoderType,        #MLP, Hybrid
                 encoderParameters,
                 decoderType,        #MLP_decoder_moder, IBP_factor
                 decoderParameters):

        # set the data
        data = np.asarray(data, dtype=precision)
        self.y = th.shared(data)
        self.N = data.shape[0]
        self.P = data.shape[1]
        self.B = params['miniBatchSize']
        self.Q = params['dimZ']


        self.srng = createSrng(seed=123)

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
        if encoderType == 'Hybrid':
            self.encoder = Hybrid_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                self.jitterProtector,
                encoderParameters,
                self.srng)
        elif encoderType == 'MLP':
            self.encoder = MLP_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                encoderParameters,
                self.srng)
        elif encoderType == 'Hysterisis':
                self.encoder = Hysterisis_variational_model(
                self.y_miniBatch,
                self.B,
                self.P,
                self.Q,
                encoderParameters,
                self.srng)
        else:
            raise RuntimeError('Unrecognised encoder type')

        if decoderType == 'MLP':
            self.decoder = MLP_likelihood_model(self.y_miniBatch,
                decoderParameters, self.B, self.P, self.Q, self.encoder,
                params, self.srng)
        else:
            raise RuntimeError('Unrecognised decoder type')

        self.encoder.construct_L_terms()
        self.decoder.construct_L_terms(self.encoder)

        self.L = self.encoder.L_terms + self.decoder.L_terms

        self.dL = T.grad(self.L, self.gradientVariables)
        for i in range(len(self.dL)):
            self.dL[i].name = 'dL_d' + self.gradientVariables[i].name

    def sample(self):
        self.sample_batchStream()
        self.sample_padStream()

    def train(self,
        numberOfEpochs=1,
        learningRate=1e-3,
        fudgeFactor=1e-6,
        maxIters=np.inf,
        constrain=False,
        printDiagnostics=0
        ):

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...

        print "training for {} epochs with {} learning rate".format(numberOfEpochs, learningRate)

        # pbar = progressbar.ProgressBar(maxval=numberOfIterations*numberOfEpochs).start()

        for ep in range(numberOfEpochs):

            # Sample a new batch
            self.sample()

            for it in range(self.numberofBatchesPerEpoch):

                # Sample from the encoder
                self.encoder.sample()
                self.iterator.set_value(it)
                lbTmp = self.jitterProtector.jitterProtect(self.updateFunction, reset=False)
                if constrain:
                    self.constrainKernelParameters()

                lbTmp = lbTmp.flatten()
                self.lowerBound = lbTmp[0]

                currentTime  = time.time()
                wallClock    = currentTime - startTime
                stepTime     = wallClock - wallClockOld
                wallClockOld = wallClock

                print("\n Ep %d It %d\tt = %.2fs\tDelta_t = %.2fs\tlower bound = %.2f"
                      % (ep, it, wallClock, stepTime, self.lowerBound))
                if printDiagnostics > 0 and (it % printDiagnostics) == 0:
                    self.printDiagnostics()

                self.lowerBounds.append((self.lowerBound, wallClock))

                if ep * self.numberofBatchesPerEpoch + it > maxIters:
                    break

            if ep * self.numberofBatchesPerEpoch + it > maxIters:
                break
            # pbar.update(ep*numberOfIterations+it)
        # pbar.finish()

        return self.lowerBounds

    def constructUpdateFunction(self, learning_rate=0.001, beta_1=0.99, beta_2=0.999, profile=False):

        gradColl = collections.OrderedDict([(param, T.grad(self.L, param)) for param in self.gradientVariables])

        self.optimiser = Adam(self.gradientVariables, learning_rate, beta_1, beta_2)

        updates = self.optimiser.updatesIgrad_model(gradColl, self.gradientVariables)

        # Get the update function to also return the bound!
        self.updateFunction = th.function([], self.L, updates=updates, no_default_updates=True, profile=profile)

    def construct_L_dL_functions(self):
        self.L_func = th.function([], self.L, no_default_updates=True)
        self.dL_func = th.function([], self.dL, no_default_updates=True)


    def getMCLogLikelihood(self, numberOfTestSamples=100):

        self.sample()
        ll = [0] * self.numberofBatchesPerEpoch * numberOfTestSamples
        c = 0
        for i in range(self.numberofBatchesPerEpoch):
            print '{} of {}, {} samples'.format(i, self.numberofBatchesPerEpoch, numberOfTestSamples)
            self.iterator.set_value(i)
            self.jitter.set_value(self.jitterDefault)
            for k in range(numberOfTestSamples):
                self.sample()
                ll[c] = self.jitterProtect(self.L_func, reset=False)
                c += 1

        return np_log_mean_exp_stable(ll)


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







