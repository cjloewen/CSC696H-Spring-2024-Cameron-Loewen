'''
This file stands for "Gaussian Process Experiments"
This holds the code necessary to replicating the experiments I reported

# credit to https://peterroelants.github.io/posts/gaussian-process-tutorial
^ the above reference helped with finding resources for sampling from a gaussian process
'''

import numpy.linalg as linalg
from network import Network as nw
import tensorflow.keras as keras
import scipy
import utility
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div
from scipy.special import rel_entr
from tensorflow_probability import distributions


# This is the RBF kernel, this function is from peterreolants
def exponentiated_quadratic(xa, xb, sigma = 1):
    """Exponentiated quadratic  with sigma=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm / (sigma**2))

# this function is a modified version of the code provided by peterreolants
# Allows me to sample nFunc defined at nSample equidistant points with meanFunc
def gp(nSample: int = 100, nFunc: int = 100, sigma=1, meanFunc=(lambda x : 0)):
    # Independent variable samples
    x = np.expand_dims(np.linspace(-1, 1, nSample), 1)
    # Kernel of data points
    cov = exponentiated_quadratic(x, x, sigma)
    mean = [meanFunc(x[0]) for x in np.expand_dims(np.linspace(-1, 1, nSample), 1)]
    y = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=nFunc
    )
    return x, cov, y

'''
Domain restricted, distance to origin is no greater than 1
The function passed should also have range in [-1, 1]
'''
def sampleData(nSamples = 100, nFunc = 100, sigma=1, meanFunc=(lambda x: 0)):
    examples = []
    labels = []
    count: int = 0
    x, cov, y = gp(nSamples, nFunc, sigma, meanFunc)

    examples = x.tolist() * nFunc
    for output in y:
        labels += output.tolist()

    dataset = tf.data.Dataset.from_tensor_slices((examples, labels))
    return dataset

def sampleExample(nSample = 100, start = -1, stop = 1):
    x = np.expand_dims(np.linspace(start, stop, nSample), 1)
    return tf.data.Dataset.from_tensor_slices((x, x))

'''
Variables for the experiment:
# Define the Gaussian Process
MeanFunc of the gaussian process [default is 0]
Sigma is the hyperparameter of the kernel, higher is more controlled [default is 1, which is the RBF Kernel]
# Define the sampling process
The number of points to sample along the x-axis, currently this is constrained to [-1, 1], [default is 100]
numFunc is the number of functions that are sampled from the gaussian process, this is equivalent to the number of samples from a distribution [default is 100]
# Define the Network parameters
The activation of the network [default is relu]
The number of epochs to run for [default is 50]
The configuration of the networks width and height, can be lists to run for each possible combination (resources permitting) [default is width: 8, depth: 2]
'''
def experiment(meanFunc=(lambda x: 0), sigma = 1, points = 100, numFunc = 100, activation=keras.activations.relu, epochs=50, config=[[8], [2]]):
    # get data for dataset, and testInput and testOutput for comparing our trained model
    dataset = sampleData(points, numFunc, sigma, meanFunc)
    _, _, testOutput = gp(points, numFunc, 1, meanFunc)
    testInput = sampleExample(points)
    trueMean = [meanFunc(x[0]) for x in np.expand_dims(np.linspace(-1, 1, points), 1)]
    # set the model parameters for the experiment
    network: nw = nw(dataset)
    network.setDimensionality(1) # only dealing with d=1 for this project (currently)
    network.setEpochs(epochs)
    network.setActivationFunction(activation)

    # the testinput is the same x points
    testInput, _ = list(testInput.batch(points))[0]
    # calculate the empirical distribution of the examples, there should be 1 pdf for each sampled point x

    # run test
    stats = {}
    for width in config[0]:
        for depth in config[1]:
            print(f"Config: {(width, depth)}")
            if(np.log2(width ** depth) > 10):
                print("Config skipped due to size constraints")
                continue
            network.setLayerWidth(width)
            network.setDepth(depth)
            network.startExperiment()
            # compare (lots of computation here)
            pred_list = []
            for _ in range(numFunc):
                y_pred = network.makePredictions(testInput)
                pred_list.append(y_pred)
            stats[(width, depth)] = (utility.getTrueStats(pred_list, trueMean), utility.getVarStats(pred_list, testOutput), utility.getKLStats(pred_list, testOutput))
    return stats, network

def runAllExperiments():
    
    sigma = 1 # see if this improves anything
    meanFunc = lambda x: x
    numFunc = 10 # see if this improves anything
    numSamples = 10
    activation = tf.keras.activations.sigmoid


    for epochs in epoch_list:
        for meanFunc in funcs:
            config = [[2**x for x in [2,3,4,6,8,10]], [1, 2, 3, 4, 5]]
            stats, _ = experiment(meanFunc, sigma, numSamples, numFunc, activation, epochs, config)
            utility.tempWrite(stats, f"{epochs}_{dictNames[meanFunc]}.txt")




def formatToTable():
    for epochs in epoch_list:
        for meanFunc in funcs:
            results = utility.readIn(f"{epochs}_{dictNames[meanFunc]}.txt")
            print(epochs, dictNames[meanFunc])
            for x in results.keys():
                print(x[0], x[1], results[x][0][0], results[x][0][1], results[x][1][1], results[x][2][0], results[x][2][0], sep=" & ", end="\\\\\n")
            print()

def getMins(name='zero'):
    min_avg_kl = 1000
    min_avg_diff = 1000
    kl_exp = ()
    diff_exp = ()
    for epochs in epoch_list:
        for meanFunc in funcs:
            if(dictNames[meanFunc] != name):
                continue
            results = utility.readIn(f"{epochs}_{dictNames[meanFunc]}.txt")
            for x in results.keys():
                cur_kl = results[x][2][0]
                cur_diff = results[x][0][1]
                if(cur_diff < min_avg_diff):
                    min_avg_diff = cur_diff
                    diff_exp = (epochs, dictNames[meanFunc], x)
                if(cur_kl < min_avg_kl):
                    min_avg_kl = cur_kl
                    kl_exp = (epochs, dictNames[meanFunc], x)
    print(f"Lowest Avg kl: {min_avg_kl}, achieved by {kl_exp}")
    print(f"Lowest Avg diff: {min_avg_diff}, achieved by {diff_exp}")
                

def recordModel(config, epochs, func):
    testInput = sampleExample(200, -2, 2)
    testInput, _ = list(testInput.batch(200))[0]
    testOutputs = []
    for i in range(100):
        stats, network = experiment(func, 1, 100, 100, "sigmoid", epochs, config)
        testOutputs.append(network.makePredictions(testInput))
    utility.writeFunction(testInput, testOutputs, "test_1_func.txt")
            
def graphModel():
    x, y = readFunction("demo1_1_func.txt")
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.plot(x, y[i], linestyle='-', marker='o', markersize=3)
    plt.xlim([-2, 2])
    plt.show()
    #lt.show()
    mean_y = utility.meanY(x, y)
    plt.xlim([-2, 2])
    plt.plot(x, mean_y, linestyle='-', marker='o', markersize=3)
    plt.show()



fun1 = lambda x: 0
fun2 = lambda x: x
fun3 = lambda x: np.sin(np.pi*x)
fun4 = lambda x: x**3
fun5 =  lambda x: np.sin(4*np.pi*x) * np.log(x + 1.1)
dictNames = {fun1: "zero", fun2: "line", fun3: "sine", fun4: "cube", fun5: "good"}

epoch_list = [10, 50, 300]
funcs = [fun1, fun2, fun3, fun4, fun5]
formatToTable()