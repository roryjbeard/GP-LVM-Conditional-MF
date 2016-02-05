

from Auto_encoder_model import VA
import numpy as np
import argparse
import gzip, cPickle
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

[N,dimX] = data.shape

dimZ = 20
dimX = 5
HU_decoder = 400
batchSize = 100

# numTestSamples = 5000

numberOfInducingPoints = 5

# learning_rate = 1e-3


print "Initialising"

va = VA(
    numberOfInducingPoints, # Number of inducing ponts in sparse GP
    batchSize,              # Size of mini batch
    dimX,                   # Dimensionality of the latent co-ordinates
    dimZ,                   # Dimensionality of the latent variables
    x_train,                   # [NxP] matrix of observations
    kernelType='RBF',
    encoderType_qX='MLP',  # 'FreeForm', 'MLP', 'Kernel'.
    encoderType_rX='MLP',  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
    encoderType_ru='FreeForm',  # 'FreeForm', 'MLP', 'NoEncoding'
    z_optimise=False,
    numHiddenUnits_encoder=0,
    numHiddentUnits_decoder=10,
    continuous=True
)

va.construct_L_using_r()

va.setKernelParameters(0.01, 5*np.ones((2,)),
    1e-100, 0.5,
    [1e-10,1e-10], [10,10] )

va.randomise()

va.constructUpdateFunction()

print "Training"
learning_rate = 1e-3
numberOfEpochs = 1

va.train(numberOfEpochs=numberOfEpochs, maxIters=10)

#for i in range(1,8):
#
#    learning_rate = 1e-4*round(10.**(1-(i-1)/7.), 1)
#    va.train_adagrad(numberOfEpochs=3**(i-1), learningRate=learning_rate )


    # print va.L_func()
    # temp = va.dL_func()

    # f = open('workfile', 'w')
    # f.write(str(temp))
    # f.close()


#lowerBounds = va.train_adagrad( n_iter, learningRate=learning_rate )
#
#print "Testing"
#vatest = va = VA(
#    numberOfInducingPoints, # Number of inducing ponts in sparse GP
#    batchSize,              # Size of mini batch
#    dimX,                   # Dimensionality of the latent co-ordinates
#    dimZ,                   # Dimensionality of the latent variables
#    x_test,                   # [NxP] matrix of observations
#    kernelType='RBF',
#    encoderType_qX='MLP',  # 'FreeForm', 'MLP', 'Kernel'.
#    encoderType_rX='MLP',  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
#    encoderType_ru='FreeForm',  # 'FreeForm', 'MLP', 'NoEncoding'
#    z_optimise=False,
#    numHiddenUnits_encoder=0,
#    numHiddentUnits_decoder=10,
#    continuous=True
#)
#
#
#vatest.copyParameters(va)
#
#testLogLhood = vatest.getTestLowerBound()
#print testLogLhood
