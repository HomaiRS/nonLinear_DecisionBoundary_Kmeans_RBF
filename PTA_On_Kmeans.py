
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:19:31 2019

@author: Homai
"""

import sys
import random
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import linalg as LA
from myK_Means import Output_Centroids_points
from myK_Means import plotter
from myK_Means import CallPlotter
from numpy import genfromtxt

def merger(a, b, Column):
    return pd.merge(b, a, on=Column, how='inner')

def Dist_a_Vect_Scalar(Matrix, xvect):
    diffVect = (Matrix - np.vstack(xvect))
    return LA.norm(diffVect, axis=0)

def GetBeta(Xvecti, ci): ## Radial Basis Function network
    normDiff = Dist_a_Vect_Scalar(Xvecti, ci)
#    sigma = normDiff.mean()
#    if sigma == 0:
#    sigma = 0.1
#    Betai = 1/(2 * sigma**2)
    Betai = 0.1
    return Betai
  
def GetBetaPerCluster(aMatrix, Clustr_labels):   
    for i in Clustr_labels:
        cluster = aMatrix[aMatrix['K'] == Clustr_labels[i]]
        XY = cluster[['X','Y']]; XY = XY.values
        Centroids = cluster.loc[cluster.index[0],['Cx','Cy']]
        beta = GetBeta(XY.T, Centroids); aMatrix = aMatrix.values
        Idx = np.argwhere(aMatrix[:,2] == Clustr_labels[i])
        aMatrix[Idx, 5:6] = beta
        aMatrix = pd.DataFrame(aMatrix); aMatrix.columns = ['X','Y','K', 'Cx','Cy','Beta']  
    return aMatrix

def RBF(anyX, beta, centroid):  
    normDiff = Dist_a_Vect_Scalar(centroid, anyX)
    Phi = np.exp(-1 * beta * (normDiff)**2)
    return np.append([1], Phi) ## 21*1 dimensional vector

def intializeWeights(Ncentroids):
    return np.random.uniform(0,1,Ncentroids)

def SignActivationFunc(v):
    if v>=0:
        return 1
    else:
        return -1

def PerceptronTA(Mx, Centroid, Betas, eta, MaxEpoch):
    W = intializeWeights(len(Betas)+1) ## the 1 is for bias
    epoch = 0; MisClassN = 10; accTol = 98
    EpochAcc = np.ones([MaxEpoch+1])
    while (EpochAcc[epoch] < accTol) & (epoch < MaxEpoch): # & (MisClassN > 0) 
        correctClassify = 0
        for i in range(len(Mx)):
            xy = Mx[i,0:2]; 
            phiVect = RBF(xy.T, Betas, Centroid.T)
            di = Mx[i,2]
            Penalize = (eta * phiVect) * (di - SignActivationFunc(np.dot(W.T, phiVect)))
            W = W + Penalize  
        for j in range(len(Mx)):   
            xx = Mx[j, 0:2]
            PhiPerx = RBF(xx.T, Betas, Centroid.T)
#            error[j] = abs(Mx[j, 2] - SignActivationFunc(np.dot(W.T, PhiPerx)))
            if SignActivationFunc(np.dot(W.T, PhiPerx)) == Mx[j, 2]:
                correctClassify = correctClassify + 1
        EpochAcc[epoch] = (correctClassify/len(Mx))*100 ## Percentage accuracy on the training data
#        MisClassN = np.sum(error)
        print('Epoch = ', epoch, ', Accuracy: %', EpochAcc[epoch]); print('-'*30)
        epoch = epoch + 1; 
    return W #--- firt element of W is bias

def PlotContourOfBoundary(X, Y, Z):  
    levels = 30
    cs = plt.contour(X, Y, Z.T, levels)
    plt.clabel(cs,inline=2,fontsize=9)
    
## Not done here yet
def DecisionBoundary(Betas, Centroid, weight):
    XYz = []; pts = np.zeros(2)
    X1 = np.arange(0,1,0.01)
    X2 = np.arange(0,1,0.01)
    gMx = np.zeros((len(X1),len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            pts[0]= X1[i]; pts[1]= X2[j]  
            PhiVect = RBF(pts.T, Betas, Centroid.T)
            XYz.append([pts[0], pts[1], np.dot(weight, PhiVect)])
            gMx[i,j] =  np.dot(weight, PhiVect)
    return X1, X2 , gMx        




eta = 0.5; Num_centroids = 20; leg=[]
NumberInput = 100;  KMeans_maxEpoch=10

[X, C1, C2, d, K1, K2, Clusters_C1, New_Centroids1, Clusters_C2, New_Centroids2] = Output_Centroids_points(NumberInput, KMeans_maxEpoch)

DataStruc = np.c_[X.T,d]; DataStruc = pd.DataFrame(DataStruc); DataStruc.columns = ['X','Y','d']
AllClusters = np.c_[Clusters_C1,Clusters_C2]; AllClusters = AllClusters.T
All_Centroids = np.c_[New_Centroids1, New_Centroids2]; All_Centroids=All_Centroids.T

Matrix = np.zeros((len(AllClusters),6))
## Xx, Xy, Cluster, CentroidX, CentroidY 
Matrix[:,0:3] = AllClusters
for i in range(len(AllClusters)):
    Matrix[i,3:5] = All_Centroids[int(AllClusters[i,2]),:]

dfMatrix = pd.DataFrame(Matrix); dfMatrix.columns = ['X','Y','K', 'Cx','Cy','Beta']  

K = np.append(K1,K2)
dfMatrix2 = GetBetaPerCluster(dfMatrix, K)
Df = merger(dfMatrix2, DataStruc, ['X','Y'])

ClustersPhi_info = Df.groupby(['K'], as_index=False).agg({"Cx":"first","Cy":"first", "Beta":"first"})

#del dfMatrix, dfMatrix2
#del dfMatrix, Matrix, All_Centroids, AllClusters, DataStruc
#Df.to_csv('an.csv', sep=',', index=False, header= None)
#my_data = genfromtxt('myData.csv', delimiter=',')
Df2 = np.asarray(Df); PTA_Maxepoch = 500
Weight = PerceptronTA(Df2, ClustersPhi_info[['Cx','Cy']], ClustersPhi_info['Beta'], eta ,PTA_Maxepoch)


[X, Y, G_XY] = DecisionBoundary(ClustersPhi_info['Beta'], ClustersPhi_info[['Cx','Cy']],Weight)
# fig = plt.figure(figsize=(7,7))
fig = plt.figure(figsize = (8,6)); Ax = plt.subplot(111)
PlotContourOfBoundary(X, Y, G_XY)

#==================================== K-means visualization
# fig = plt.figure(figsize = (8,6)); Ax = plt.subplot(111)
plotter(C1, C2, New_Centroids1, NumberInput)
plotter(C1, C2, New_Centroids2, NumberInput)
leg.append('Centroids C1'); leg.append('Centroids C2');
K1 = np.arange(10); K2 = np.arange(10,20,1) 
CallPlotter(C1, C2, NumberInput, Clusters_C1, K1, Ax, 'o')
CallPlotter(C1, C2, NumberInput, Clusters_C2, K2, Ax, 'D')
#plt.savefig('2Kmeans_BothClasses.pdf', dpi=200)











