import numpy as np
import cPickle as pkl
import gzip
import os
from copy import deepcopy

from Auto_encoder_model import AutoEncoderModel

continuous=False

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
if continuous:
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = pkl.load(f)
    f.close()
    datatype = 'continuous'
else:
    x_train = np.loadtxt('binarized_mnist_train.amat')
    x_test = np.loadtxt('binarized_mnist_test.amat')
    datatype = 'binary'

numberOfEpochs = 10
learning_rate = 1e-4

dimZ = 40
dimX = 300
params = {'miniBatchSize' : 200, 'dimZ':300,
         'theanoRandomSeed':123, 'numpyRandomSeed':123,
         'BinaryFromContinuous':False}
encoderParameters = {'Type':None,  'numHiddenUnits_encoder' : 500, 'numHiddenLayers_encoder' : 2, 'numStochasticLayers_encoder':None,
                     'dimX':dimX, 'numberOfInducingPoints':400, 'kernelType':'ARD', 'theta':np.ones((1,dimX+1)), 'theta_min':1e-3, 'theta_max':1e3}
decoderParameters = {'Type':'MLP', 'numHiddenUnits_decoder' : 500, 'numHiddenLayers_decoder' : 2, 'numStochasticLayers_encoder':None, 'continuous':continuous}

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
else:
    raise RuntimeError('Enter number between 1 and 6')

annotation = raw_input('Enter file annotation: ')


filename = 'mnist_' + datatype + '_exp_' + experimentNumber + '_' + annotation + '.pkl'

vae_train = AutoEncoderModel(x_train, params, encoderParameters, decoderParameters, L_terms='Train')
vae_test  = AutoEncoderModel(x_test,  params, encoderParameters, decoderParameters, L_terms='Test')

vae_train.constructUpdateFunction(learning_rate=learning_rate)
vae_test.construct_L_function()

vae_train.train(numberOfEpochs=numberOfEpochs,
        maxIters=np.inf,
        #maxIters=10,
        constrain=False,
        printDiagnostics=0,
        printFrequency=50,
        testModel=None,
        numberOfTestSamples=10)

trainingBounds = deepcopy(vae_train.lowerBounds)
testBounds = deepcopy(vae_test.lowerBounds)

filename_to_save = os.path.join('results', filename)
try:
    with open(filename_to_save, "wb") as f:
        pkl.dump([trainingBounds,
                  testBounds], f, protocol=pkl.HIGHEST_PROTOCOL)
except:
    print "Failed to write to file {}".format(filename_to_save)


#try:
#    with open(filename_to_save, "rb") as f:
#        encoderParameters_load,
#        decoderParameters_load,
#        vae_train_load,
#        vae_test_load = pkl.load(f)
#except:
#    print "Failed to load file {}".format(filename_to_save)