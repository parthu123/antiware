import pandas as pd
import numpy as np
import math


# convert class in the form of 0 and 1
def actual_list(data_actual,d_class):
    dlist = []

    for x in range(len(data_actual)):
        if(data_actual[x] == d_class):
            dlist.append(1)
        else:
            dlist.append(0)

    return dlist


# returns training set separates according to class
# and class probabilities(cp)
def separate_Class(dataset):
    separated = {}
    cp = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
            cp[vector[-1]] = 0
        separated[vector[-1]].append(vector[0:-1])
        cp[vector[-1]] += 1

    for k in cp:
        cp[k] /= float(len(dataset))
    return separated, cp


# calculate mean and standard deviation
def learn(separated):
    summary = {}
    for classValue, instances in separated.iteritems():
        summary[classValue] = zip(
            np.mean(instances, axis=0), np.std(instances, axis=0, ddof=1))
    return summary


# Gaussian probablity density function
def Gaussian_probablity(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# calculates class probablities for given feature vector
def class_probablity(vector, summary, cp):
    probablity = {}
    for classValue, classSummaries in summary.iteritems():
        probablity[classValue] = cp[classValue]
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = vector[i]
            probablity[classValue] *= Gaussian_probablity(x, mean, stdev)
    return probablity


# predict the class for given input attributes
def predict(summaries, inputVector, cp):
    array1 = []
    probabilities = class_probablity(inputVector, summaries, cp)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
            array1.append(bestLabel)
    #print array1
    return bestLabel


# get predictions for test set
def getPredictions(summaries, testSet, cp):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i], cp)
        predictions.append(result)
    return predictions


# function to calculate accuracy of the prediction
def accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def k_fold_cross_validation(X, K, randomise=False):
    ac = 0

    if randomise:
        from random import shuffle
        X = list(X)
        shuffle(X)
    for k in range(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        test = [x for i, x in enumerate(X) if i % K == k]
        # convert from list to matrix form
        training = np.vstack(training)
        test = np.vstack(test)
        # print(validation)
        separated, cp = separate_Class(training)   # cp is class probablity
        summaries = learn(separated)
        predictions = getPredictions(summaries, test, cp)
        ac += accuracy(test, predictions)
    # average accuracy of all the folds
    f_accuracy = float(ac) / float(K)
    print(f_accuracy)


def main():
    # set variables
    folds = 10

    # read and load file
    filename = 'for_meta.csv'
    d_class = 1
    df = pd.read_csv(filename)
    m, n = df.shape
    data_actual = df.ix[:, n - 1]
    actual_data = actual_list(data_actual,d_class)

    # feature matrix
    features = df.as_matrix(columns=df.columns[0:n - 1])
    X = np.c_[features, actual_data]

    k_fold_cross_validation(X, folds, False)


main()
