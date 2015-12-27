import math
import random
import matplotlib.pyplot as plot

points = [(4, 2), (2, 1), (6, 4), (4, 2), (6, 9), (7, 2), (5, 6), (10, 8), (17, 9), (9, 8)]
plotPoints(points, 'green')
covarianceMatrix = covarianceMatrix(points)
eigenVector, eigenValue = eigen(covarianceMatrix)
plot.arrow(0, 0, eigenVector[0]*20, eigenVector[1]*20)
projectedPoints = project(points, eigenVector)
plotPoints(projectedPoints, 'blue')
plot.show()
plot.close()

def eigen(mat):
    n = len(mat)
    eigenMat = [0] * n
    eigenMat[0] = 1
    val = 0

    for l in range(100):
        tmp = [0] * n
        for i in range(n):
            for j in range(n):
                tmp[i] += mat[i][j] * eigenMat[j]
        val = magnitude(tmp)
        for k in range(n):
            eigenMat[k] = tmp[k] / val

    return eigenMat, val

def mean(pointList):
    x, y = zip(*pointList)
    return (sum(x)/len(x), sum(y)/len(y))

def covariance(x, y, means):
    s = 0
    for i in range(len(x)):
        s += (x[i] - means[0]) * (y[i] - means[1])
    return s / (len(x)-1)

def magnitude(v):
    sq = 0
    for k in range(len(v)):
        sq += v[k] * v[k]
    return math.sqrt(sq)

def covarianceMatrix(l):
    f = list(zip(*l))
    n = len(f)
    m = mean(l)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            v = covariance(f[i], f[j], m)
            row.append(v)
        matrix.append(row)
    return matrix

def project(l, v):
    n = len(v)
    res = []
    for p in l:
        dot = p[0]*v[0] + p[1]*v[1]
        res.append((dot * v[0], dot * v[1]))
    return res

def plotPoints(listValues, plotColor):
    x, y = zip(*listValues)
    plot.scatter(x, y, color=plotColor)
