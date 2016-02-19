

from Auto_encoder_model import VA
import numpy as np
import argparse
import gzip, cPickle
import os
import theano.tensor as T

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["LD_LIBRARY_PATH"]   = os.path.dirname(os.path.realpath(__file__)) + '/fastlin/'
os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(os.path.realpath(__file__)) + '/fastlin/'
parser = argparse.ArgumentParser()

args = parser.parse_args()

print "Loading MNIST data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

[N,dimX] = data.shape

dimZ = 40
dimX = 3
batchSize = 100
encoderType_qX='FreeForm2'
encoderType_rX='FreeForm2'
encoderType_qu='Kernel'
encoderType_ru='FreeForm2'
Xu_optimise=True
kernelType='RBF'
numHiddenUnits_encoder=200
numHiddentUnits_decoder=200
numberOfInducingPoints =50
learning_rate = 1e-3
numTestSamples = 100

print "Initialising"

va = VA(
    numberOfInducingPoints, # Number of inducing ponts in sparse GP
    batchSize,              # Size of mini batch
    dimX,                   # Dimensionality of the latent co-ordinates
    dimZ,                   # Dimensionality of the latent variables
    x_train,                   # [NxP] matrix of observations
    kernelType=kernelType,
    encoderType_qX=encoderType_qX,  # 'FreeForm1', 'FreeForm2', 'MLP', 'Kernel'.
    encoderType_qu=encoderType_qu,  # 'Kernel'
    encoderType_rX=encoderType_rX,  # 'FreeForm1', 'FreeForm2', 'MLP', 'Kernel', 'NoEncoding'.
    encoderType_ru=encoderType_ru,  # 'FreeForm', 'MLP', 'NoEncoding'
    Xu_optimise=Xu_optimise,
    numHiddenUnits_encoder=numHiddenUnits_encoder,
    numHiddentUnits_decoder=numHiddentUnits_decoder,
    continuous=True
)

va.construct_L_using_r()

va.setKernelParameters(0.01, 5*np.ones((2,)),
    1e-100, 0.5,
    [1e-10,1e-10], [10,10] )

va.randomise()

va.init_Xu_from_Xf()

va.constructUpdateFunction()

print "Training"
learning_rate = 1e-3
numberOfEpochs = 1

#va.printMemberTypes()

va.printMemberTypes(memberType=T.sharedvar.TensorSharedVariable)
va.printMemberTypes(T.sharedvar.ScalarSharedVariable)

va.train(numberOfEpochs=numberOfEpochs, maxIters=20)

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
#    kernelType=kernelType,
#    encoderType_qX=encoderType_qX,  # 'FreeForm', 'MLP', 'Kernel'.
#    encoderType_rX=encoderType_rX,  # 'FreeForm', 'MLP', 'Kernel', 'NoEncoding'.
#    encoderType_ru=encoderType_ru,  # 'FreeForm', 'MLP', 'NoEncoding'
#    Xu_optimise=Xu_optimise,
#    numHiddenUnits_encoder=numHiddenUnits_encoder,
#    numHiddentUnits_decoder=numHiddentUnits_decoder,
#    continuous=True
#)

#va.construct_L_using_r()
#vatest.copyParameters(va)

#testLogLhood = vatest.getMCLogLikelihood(numTestSamples)
#print testLogLhood
