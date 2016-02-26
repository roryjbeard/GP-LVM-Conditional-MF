# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:44:40 2016

@author: clloyd
"""

import numpy as np
import theano as th
import theano.tensor as T
from theano.tensor import nlinalg

from GP_LVM_CMF import SGPDV
from testTools import checkgrad
from utils import log_mean_exp_stable, dot, trace, softplus, sharedZeroVector, sharedZeroMatrix, plus

precision = th.config.floatX

class VA(SGPDV):

    def __init__(self,
                 data,
                 encoderType,    #MLP, Hybrid
                 encoderParameters,
                 decoderType,    #MLP_decoder_moder, IBP_factor):

        # set the data
        data = np.asarray(data, dtype=precision)
        
        self.numberofBatchesPerEpoch = int(np.ceil(np.float32(self.N) / self.B))
        numPad = self.numberofBatchesPerEpoch * self.B - self.N

        self.batchStream = srng.permutation(n=self.N)
        self.padStream   = srng.choice(size=(numPad,), a=self.N,
                                       replace=False, p=None, ndim=None, dtype='int32')

        self.batchStream.name = 'batchStream'
        self.padStream.name = 'padStream'

        self.iterator = th.shared(0)
        self.iterator.name = 'iterator'

        self.allBatches = T.reshape(T.concatenate((self.batchStream, self.padStream)), [self.numberofBatchesPerEpoch, self.B])
        self.currentBatch = T.flatten(self.allBatches[self.iterator, :])

        self.allBatches.name = 'allBatches'
        self.currentBatch.name = 'currentBatch'

        self.y_miniBatch = self.y[self.currentBatch, :]
        self.y_miniBatch.name = 'y_miniBatch'

        self.lowerBound = -np.inf  # Lower bound

        if encoderType == 'Hybrid'
            self.jitterProtector = jitterProtector()
            self.encoder = Hybrid_encoder(
                self.y_miniBatch,
                self.jitterProtector,
                encoderParams["numberOfInducingPoints"],
                encoderParams["minibatchSize"],
                encoderParams["dimY"],
                encoderParams["dimX"],
                encoderParams["dimZ"],
                encoderParams["use_r"],
                encoderParams["kernelType"],
                encoderParams["encoderType_qX"],
                encoderParams["encoderType_rX"],
                encoderParams["Xu_optimise"],
                encoderParams["numberOfEncoderHiddenUnits"] )
        elif encoderType:
            pass

        if decoder == 'MLP':
            self.decoder( self.encoder )



        self.L = self.encoder.L + self.decoder.L


        else:
            self.L += self.log_p_z() - self.log_q_z_uX()

        self.dL = T.grad(self.L, self.gradientVariables)
        for i in range(len(self.dL)):
            self.dL[i].name = 'dL_d' + self.gradientVariables[i].name

def epochSample(self):

        self.sample_batchStream()
        self.sample_padStream()
        self.iterator.set_value(0)


 def train(self, numberOfEpochs=1, learningRate=1e-3, fudgeFactor=1e-6, maxIters=np.inf, constrain=False, printDiagnostics=0):

        startTime    = time.time()
        wallClockOld = startTime
        # For each iteration...

        print "training for {} epochs with {} learning rate".format(numberOfEpochs, learningRate)

        # pbar = progressbar.ProgressBar(maxval=numberOfIterations*numberOfEpochs).start()

        for ep in range(numberOfEpochs):

            self.epochSample()

            for it in range(self.numberofBatchesPerEpoch):

                self.sample()
                self.iterator.set_value(it)
                lbTmp = self.jitterProtect(self.updateFunction, reset=False)
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