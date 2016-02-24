
from Auto_encoder_model import VA
import numpy as np
import argparse
import gzip, cPickle
import os
import theano.tensor as T
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

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


[N,dimX] = x_train.shape

# print 'Loading PCA initialisation of Xf with dimX components if it exists'
# filename = 'PCA_init_Xf_{}_components.pkl'.format(dimX)
# try:
#     with open(filename, 'rb') as f:
#         Xf = cPickle.load(f)
# except:
#     'Does not exist so performing the PCA now and saving result'
#     pcaXf = PCA(n_components=dimX)
#     Xf = pcaXf.fit_transform(x_train)
#     filename = 'PCA_init_Xf_{}_components'.format(dimX)
#     with open(filename, 'wb') as ff:
#         cPickle.dump(Xf, ff)


data = x_train

dimZ = 40
dimX = 3
batchSize = 200
encoderType_qX='MLP'
encoderType_rX='MLP'
Xu_optimise=True
kernelType='RBF'
numHiddenUnits_encoder=400
numHiddentUnits_decoder=400
numberOfInducingPoints =500
learning_rate = 1e-3
numTestSamples = 100
numHiddenLayers_decoder = 1

print "Initialising"

va = VA(
    numberOfInducingPoints, # Number of inducing ponts in sparse GP
    batchSize,              # Size of mini batch
    dimX,                   # Dimensionality of the latent co-ordinates
    dimZ,                   # Dimensionality of the latent variables
    x_train,                   # [NxP] matrix of observations
    kernelType=kernelType,
    encoderType_qX=encoderType_qX,  # 'FreeForm1', 'FreeForm2', 'MLP', 'Kernel'.
    encoderType_rX=encoderType_rX,  # 'MLP', 'Kernel', 'NoEncoding'.
    Xu_optimise=Xu_optimise,
    numHiddenUnits_encoder=numHiddenUnits_encoder,
    numHiddenUnits_decoder=numHiddentUnits_decoder,
    numHiddenLayers_decoder=numHiddenLayers_decoder,
    continuous=True
)

va.construct_L_using_r()

va.setKernelParameters(5*np.ones((2,)),
    1e-100, 0.5,
    [1e-10,1e-10], [10,10] )

va.randomise()
# va.Xz.set_value()
va.init_Xu_from_Xz()


#va.printMemberTypes()
#va.printSharedVariables()
#va.printTheanoVariables()


va.constructUpdateFunction()

print "Training"
learning_rate = 1e-3
numberOfEpochs = 100




va.train(numberOfEpochs=numberOfEpochs)

makePlots=False
if makePlots:
    # Generate synthesised data from the learned decoder
    # at each point in a grid of a 2d latent space

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    va.create_new_data_function()
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            va.z_test.set_value(z_mu)
            x_mean = va.new_data_function()
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()

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
