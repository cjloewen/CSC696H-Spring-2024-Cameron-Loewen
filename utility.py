'''
Utility.py holds most of the functions for getting various metrics and dealing with files
Most of the data is written to disk for the sake of reformatting later in case things changed, or new visuals were needed
'''
import numpy as np
import numpy.linalg as linalg
import scipy
from scipy.special import kl_div
from scipy.special import rel_entr
from scipy.stats import norm


def getKLStats(a, b):
    # go from list of function outputs to list of values at each point
    a = transpose(a)
    b = transpose(b)
    max: float = 0
    avg = 0
    for i in range(len(a)):
        afit = norm.fit(a[i])
        bfit = norm.fit(b[i])
        kl = getKLDiv(afit, bfit)
        avg += kl
        if(kl > max):
            max = kl
    return round(max, 3), round(avg / len(a), 3)

def getKLDiv(a, b):
    a_mean, a_std = a[0], a[1]
    b_mean, b_std = b[0], b[1]
    kl = np.log((b_std/a_std)) + (a_std**2 + (a_mean - b_mean)**2) / (2*b_std**2) - .5
    return kl

def getTrueStats(a, b):
    a = transpose(a)
    max_diff_mean = 0
    avg_diff = 0
    for i in range(len(a)):
        a_var = np.var(a[i], ddof=1)
        a_mean = np.mean(a[i])
        diff_mean = abs(a_mean - b[i])
        avg_diff += diff_mean
        max_diff_mean = max(max_diff_mean, diff_mean)
    size = len(a)
    return (round(max_diff_mean, 3), round(avg_diff / size, 3))


def getEmpiricalStats(a, b):
    a = transpose(a)
    b = transpose(b)
    max_diff_var: float = 0
    max_diff_mean = 0
    a_avg_var = 0
    b_avg_var = 0
    a_avg_mean = 0
    b_avg_mean = 0

    for i in range(len(a)):
        a_var = np.var(a[i], ddof=1)
        b_var = np.var(b[i], ddof=1)
        a_mean = np.mean(a[i])
        b_mean = np.mean(b[i])
        diff_var: float = abs(a_var - b_var)
        diff_mean = abs(a_mean - b_mean)
        a_avg_var += a_var
        b_avg_var += b_var
        a_avg_mean += a_mean
        b_avg_mean += b_mean
        max_diff_var = max(max_diff_var, diff_var)
        max_diff_mean = max(max_diff_mean, diff_mean)
    size = len(a)
    return ((
        round(max_diff_mean,3),
        round(a_avg_mean / size, 3),
        round(b_avg_mean / size), 3), (
        round(max_diff_var, 3),
        round(a_avg_var / size, 3), 
        round(b_avg_var / size, 3)
        ))

def getVarStats(a, b):
    a = transpose(a)
    b = transpose(b)
    max_diff_var: float = 0
    a_avg_var = 0
    b_avg_var = 0

    for i in range(len(a)):
        a_var = np.var(a[i], ddof=1)
        b_var = np.var(b[i], ddof=1)

        diff_var: float = abs(a_var - b_var)
        a_avg_var += a_var
        b_avg_var += b_var
        max_diff_var = max(max_diff_var, diff_var)
    size = len(a)
    return (
        round(max_diff_var, 3),
        round(a_avg_var / size, 3), 
        round(b_avg_var / size, 3)
        )

def getMeanStats(a, b):
    # go from list of function outputs to list of values at each point
    a = transpose(a)
    b = transpose(b)
    max = 0
    for i in range(len(a)):
        a_mean = np.mean(a[i])
        b_mean = np.mean(b[i])
        diff = abs(a_mean - b_mean)
        if(diff > max):
            max = diff
    return max

def meanY(x, y):
    y = transpose(y)
    return [np.mean(i) for i in y]

def transpose(x):
    return [list(i) for i in zip(*x)]

    
# code from the keras tutorials
def compute_predictions(model, examples, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(len(examples)):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
        )


def discretize(data, size = 101, a=-3, b=3):
    bins = np.linspace(a, b, size)
    binSize = bins[1] - bins[0]
    discretized = list(np.digitize(data, bins))
    denom = len(data)
    pdf = []
    for i in range(1, len(bins)):
        pdf.append(discretized.count(i) / denom)
    return pdf, binSize

def getKL(a, b) -> float:
    if(len(a) != len(b)):
        print("The PDFs given are not of the same support")
        quit()
    return sum(rel_entr(a, b))



def tempWrite(experimentResults, name="results.txt", titleLine="Default"):
    outfile = open(name, "w")
    outfile.write(f"{titleLine}\n")
    for x in experimentResults.keys():
        outfile.write(f"Config: {x}\n") 
        meanStats = experimentResults[x][0]
        varStats = experimentResults[x][1]
        auc = experimentResults[x][2]
        outfile.write(f"{meanStats[0]}, {meanStats[1]},\n")
        outfile.write(f"{varStats[0]}, {varStats[1]}, {varStats[2]}\n")
        outfile.write(f"{auc[0]}, {auc[1]}\n")
    outfile.close()

def writeFunction(x, y, name):
    file = open(name, "w")
    for i in x:
        file.write(f"{i[0]} ")
    file.write("\n")
    for i in y:
        for j in i:
            file.write(f"{j[0]} ")
        file.write("\n")
    file.close()

def readFunction(name):
    file = open(name, "r")
    x = [float(i) for i in file.readline().strip().split()]
    y_list = []
    for i in file.readlines():
        y_list.append([float(j) for j in i.split()])
    return x, y_list
    file.close()

def readIn(name):
    infile = open(name, "r")
    infile.readline()
    lines = infile.readlines()
    counter = 0
    results = {}
    while(counter * 4 < len(lines)):
        index = counter * 4
        key = lines[index].strip().split(": ")[1]
        key = key.split(",")
        key = (int(key[0][1:]), int(key[1][:-1]))
        meanStats = [float(x) for x in lines[index + 1].strip(", \n").split(", ")]
        varStats = [float(x) for x in lines[index + 2].strip().split(", ")]
        kl = [float(x) for x in lines[index + 3].strip().split(", ")]
        results[key] = (meanStats, varStats, kl)
        counter += 1
    infile.close()
    return results