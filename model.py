import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers
import os

dfmain = pd.read_excel(os.getcwd() + "\\MLdrivenDataAnalysisCensus2011\\DDWCT-0000B-01.xls", header=6)
df = dfmain.pivot_table(index = [5], columns = [7], values = [x for x in range(8, 23 + 1)])

df = df.drop(columns=[( 23,          '10-14'),
            ( 23,          '15-19'),
            ( 23,          '15-59'),
            ( 23,          '20-24'),
            ( 23,          '25-29'),
            ( 23,          '30-34'),
            ( 23,          '35-39'),
            ( 23,          '40-49'),
            ( 23,            '5-9'),
            ( 23,          '50-59'),
            ( 23,            '60+'),
            ( 23,          '60-69'),
            ( 23,          '70-79'),
            ( 23,            '80+'),
            ( 23, 'Age not stated')])

threshold = 5e6
dataset = df[df[23, 'Total'] < threshold]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

PopData = tf.convert_to_tensor(train_dataset, dtype=tf.float32)

x_train = PopData[:,:-1]
y_train = tf.reshape(PopData[:, -1], [402])

PopData = tf.convert_to_tensor(test_dataset, dtype=tf.float32)

x_test = PopData[:,:-1]
y_test = tf.reshape(PopData[:, -1], [101])

class KerasLinearModel(tf.keras.Model):
    def __init__(self, **kwargs): # The first element in layers is 28 * 28 and the last element is 10
        super().__init__(**kwargs)

        num_params = x_train.shape[1]

        self.Grads = tf.Variable(tf.ones([num_params]))
        self.Biases = tf.Variable(tf.ones([num_params]))

    # @tf.function # Graph execution is not supported to execute this loop
    def call(self, features):
        results = []
        for x in features:
            results.append(tf.math.reduce_mean(x * self.Grads + self.Biases))
        return tf.stack(results)

LinModel = KerasLinearModel()

LinModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                    loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                    metrics=['accuracy'],
                    run_eagerly = True)
history = LinModel.fit(x_train, y_train, batch_size=10, epochs=250)

plt.plot(history.history['loss'])

# Using the model

def GetDiagonal(TwoDTensor):
    # Assume that it is a square matrix
    Diagonal = []
    for i in range(TwoDTensor.shape[0]):
        Diagonal.append(TwoDTensor[i,i])
    return tf.stack(Diagonal)

# Get the names of the parameters
columns = []
for col in df.columns:
    columns.append(col)
columns.pop()

def GetImpFactors(i, model):
    x = tf.Variable(x_test[i:i+1])

    with tf.GradientTape(persistent = True) as g:
        g.watch(x)
        y_pred = model(x)
    dydx = tf.squeeze(g.jacobian(y_pred, x)).numpy()

    sortedGrad = np.sort(dydx)
    sortedGrad = sortedGrad[::-1]
    positions = []
    for i in sortedGrad:
        positions.append(int(np.where(dydx == i)[0]))
    # Positions and sortedGrad are parallel arrays (Positions gives the index number of the feature whose gradient is in sortedGrad)
    return positions

positions = []
for i in range(len(x_test)):
    positions.append(GetImpFactors(i, LinModel))

weights = dict([(i, 0) for i in range(len(positions[0]))])

stretch = 1

for lt in range(len(positions)):
    for i in range(len(positions[lt])):
        weights[positions[lt][i]] += stretch * 2 ** (i)

weights = dict(sorted(weights.items(), key=lambda item: item[1]))

print("The most important factors are as follows:")
for i in weights.keys():
    print(columns[i])