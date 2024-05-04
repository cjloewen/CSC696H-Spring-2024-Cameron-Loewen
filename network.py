'''
This file uses the tensorflow.keras library to train a bayesian neural network.
The class Network provided encapsulates some basic variables to create different models to experiment with
See more info on gp_exp.py for the way it is used

This model is intended only for univariate functions

Credit to the keras.io website for most of the tutorials necessary to create this file, especially the first part of:
https://keras.io/examples/keras_recipes/bayesian_neural_networks/
'''
from typing import Callable
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp



class Network:
    # parameters to vary per experiment
    dataset = None
    model = None
    dimensionality: int = 1
    numEpochs: int = 500
    width: int = 100
    depth: int = 3
    activation = "sigmoid"
    size: int = 1000
    loss = keras.losses.MeanSquaredError()

    def __init__(self, dataset):
        self.dataset = dataset

    def setDimensionality(self, d: int):
        self.dimensionality = d

    def setEpochs(self, e: int):
        self.numEpochs = e
    
    def setLayerWidth(self, w: int):
        self.width = w

    def setDepth(self, d: int):
        self.depth = d
    
    def setActivationFunction(self, a):
        self.activation = a

    def setLoss(self, l):
        self.loss = l

    def startExperiment(self):
        amount = len(self.dataset)
        self.size = amount * .9
        
        self.createBNNModel()
        
        train, test = getTrainAndTestSplits(self.dataset, int(amount * .9), int(amount / 8));
        return self.runExperiment(train, test)


    def createBNNModel(self):
        self.model = None
        inputs = createModelInputs(self.dimensionality)
        features = keras.layers.concatenate(list(inputs.values()))
        features = layers.BatchNormalization()(features)

        #Create hidden layers with weight uncertainty using the DenseVariational layer.
        for units in range(self.depth):
            features = tfp.layers.DenseVariational(
                units=self.width,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / self.size,
                activation=self.activation,
            )(features)

        outputs = layers.Dense(units=1)(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        self.model = model
        

    def runExperiment(self, train_dataset, test_dataset):

        # set basic settings of the neural network (we use RMSE)
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate = .01),
            loss=self.loss,
            metrics=[keras.metrics.RootMeanSquaredError()],
        )

        # set the learning rate, callback is passed when training begins
        def scheduler(epoch, lr):

            ratio = epoch/self.numEpochs
            if(ratio == 0):
                return lr
            return min(-np.log10(ratio) / 10, .005)
        callback = keras.callbacks.LearningRateScheduler(scheduler)

        #print("Start training the model...")
        self.model.fit(train_dataset, epochs=self.numEpochs, callbacks=[callback], validation_data=test_dataset, verbose=0)
        print("Model training finished.")
        _, rmse = self.model.evaluate(train_dataset, verbose=0)
        print(f"Train RMSE: {round(rmse, 3)}")

        print("Evaluating model performance...")
        _, rmse = self.model.evaluate(test_dataset, verbose=0)
        print(f"Test RMSE: {round(rmse, 3)}")

        return rmse


    def makePredictions(self, examples):
        return self.model(examples).numpy()


def getTrainAndTestSplits(dataset, train_size, batch_size=1):
    examples = []
    labels = []
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset

# All this does is say there is 1 feature of d dimensions, which corresponds to the input of the function
def createModelInputs(dim: int=1):
    inputs = {}

    inputs["input"] = layers.Input(
        name="input", shape=(dim, ), dtype=tf.float32
    )
    return inputs


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model



