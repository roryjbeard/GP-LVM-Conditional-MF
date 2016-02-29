import numpy as np
import theano as th
import theano.tensor as T
from printable import Printable
import argparse
import gzip, cPickle
import os

from Auto_encoder_model import AutoEncoderModel

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()


numberOfEpochs = 10
learning_rate = 1e-4

[N,dimY] = x_train.shape

data = x_train
dimZ = 40
dimX = 30
params = {'miniBatchSize' : 200, 'dimZ':400,
         'theanoRandomSeed':123, 'numpyRandomSeed':123,
         'BinaryFromContinuous':True}
encoderParameters = {'Type':None, 'numHiddenUnits_encoder' : 400, 'numHiddenLayers_encoder' : 2, 'numStochasticLayers_encoder':None,
                     'dimX':dimX, 'numberOfInducingPoints':400, 'kernelType':'ARD', 'theta':np.ones((1,dimX+1)), 'theta_min':1e-3, 'theta_max':1e3}
decoderParameters = {'Type':'MLP', 'numHiddenUnits_decoder' : 400, 'numHiddenLayers_decoder' : 2, 'numStochasticLayers_encoder':None, 'continuous':False}

experimentNumber = raw_input('Enter experiment number (1-6): ')
if experimentNumber == '1':
    encoderParameters['Type'] = 'MLP'
    encoderParameters['numStochasticLayers_encoder'] = 1
    decoderParameters['numStochasticLayers_decoder'] = 1
elif experimentNumber == '2':
    encoderParameters['Type'] = 'Hysteresis'
    encoderParameters['numStochasticLayers_encoder'] = 1
    decoderParameters['numStochasticLayers_decoder'] = 1
elif experimentNumber == '3':
    encoderParameters['Type'] = 'Hybrid'
    encoderParameters['numStochasticLayers_encoder'] = 1
    decoderParameters['numStochasticLayers_decoder'] = 1
elif experimentNumber == '4':
    encoderParameters['Type'] = 'MLP'
    encoderParameters['numStochasticLayers_encoder'] = 2
    decoderParameters['numStochasticLayers_decoder'] = 2
elif experimentNumber == '5':
    encoderParameters['Type'] = 'Hysteresis'
    encoderParameters['numStochasticLayers_encoder'] = 2
    decoderParameters['numStochasticLayers_decoder'] = 2
elif experimentNumber == '6':
    encoderParameters['Type'] = 'Hybrid'
    encoderParameters['numStochasticLayers_encoder'] = 2
    decoderParameters['numStochasticLayers_decoder'] = 2

evalTestLLhood = False
print "Initialising"

vae = AutoEncoderModel(data, params, encoderParameters, decoderParameters)


vae.construct_L_dL_functions()
vae.constructUpdateFunction(learning_rate=learning_rate)
if evalTestLLhood:
    vae.construct_MCLogLikelihood()

vae.train(numberOfEpochs=numberOfEpochs,
        maxIters=np.inf,
        constrain=False,
        printDiagnostics=0,
        evalTestLLhood
        )



