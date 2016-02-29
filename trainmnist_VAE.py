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


numberOfEpochs = 1
learning_rate = 1e-3

[N,dimY] = x_train.shape

data = x_train
dimZ = 40
params = {'miniBatchSize' : 200, 'dimZ':400,
         'theanoRandomSeed':123, 'numpyRandomSeed':123}
encoderParameters = {'Type':'MLP', 'numHiddenUnits_encoder' : 400, 'numHiddenLayers_encoder' : 1}
decoderParameters = {'Type':'MLP', 'numHiddenUnits_decoder' : 400, 'numHiddenLayers_decoder' : 1, 'continuous':True}

print "Initialising"


vae = AutoEncoderModel(data, params, encoderParameters, decoderParameters)


vae.construct_L_dL_functions()
vae.constructUpdateFunction()

vae.train(numberOfEpochs=numberOfEpochs,
        learningRate=learning_rate,
        fudgeFactor=1e-6,
        maxIters=np.inf,
        constrain=False,
        printDiagnostics=0
        )
