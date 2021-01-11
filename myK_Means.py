#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:11:04 2019

@author: Homai
"""
## This script is the implementation of K-means algorithm for clustering a dataset (Unsupervised learning)
import sys
import random
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import linalg as LA


def DesiredOutput(xx): # input of this function is the input pattern
    if (xx[1] < (1/5)*np.sin(10*xx[0])+0.3) | (((xx[1]-0.8)**2 + (xx[0]-0.5)**2) < (0.15)**2):
        return 1
    else:
        return -1

def GetDesiredOutput(inpuT):
    d_i = np.zeros(len(inpuT[0]))
    C_1 =[]; C_2=[]
    for i in range(len(inpuT[0])):
       d_i[i] = DesiredOutput(inpuT[:,i])
       if d_i[i] ==1:
           C_1.append(inpuT[:,i])
       elif d_i[i] ==-1:
           C_2.append(inpuT[:,i])
    return d_i, np.asarray(C_1), np.asarray(C_2)
    
def Inpute(Num):
    X = np.random.uniform(0,1,Num)
    Y = np.random.uniform(0,1,Num)
    Z = np.asanyarray(np.c_[X,Y])
    return np.transpose(Z)

def GetListCol(List, iCol):
    return [x[iCol] for x in List]

def GetAClusterPoints(ClassClusterS, label):
    df = pd.DataFrame(ClassClusterS.T)
    return df[df[2]==label]

def GetSeparatelyClusterPoints(ClassClusters, K):
    ListOfAllClusters =[]
    for i in K:
        dfClassClusters = GetAClusterPoints(ClassClusters, i)
        ListOfAllClusters.append(dfClassClusters)
    return ListOfAllClusters

def plotter(C1, C2, Centrids, Input_num): ## C1 and C2 are two different classes of the data
    numpy_c1 = np.asanyarray(C1); X_c1= numpy_c1[:,0]; Y_c1 = numpy_c1[:,1]
    numpy_c2 = np.asanyarray(C2); X_c2= numpy_c2[:,0]; Y_c2 = numpy_c2[:,1]
#    plt.scatter(X_c1, Y_c1, marker='D',c='K')
#    plt.scatter(X_c2, Y_c2, marker= 'D', c='red')
    plt.scatter(Centrids[0,:], Centrids[1,:], marker="x", c='K')
    plt.xlabel('X coordinate', fontweight='bold',fontsize=12)
    plt.ylabel('Y coordinate', fontweight='bold',fontsize=12)
    plt.title('input data visualization both Classes --'+str(Input_num)+' points', fontweight='bold',fontsize=14)    

def ClusterPlotter(dfClassClusterS,col, Markr):
    numpY = np.asanyarray(dfClassClusterS); 
    x= numpY[:,0]; y= numpY[:,1]
    plt.scatter(x, y,c= col, marker = Markr, alpha=0.6)
    
def CallPlotter(C_1, C_2, N_input, Clusters_Ci, K, ax, Markr):    
    colors = ['red', 'green','blue','teal','darkorchid','orange','magenta','aqua','lime','saddlebrown',
                    'darkgoldenrod', 'g', 'deepskyblue', 'rosybrown', 'coral', 'yellow', 'dimgray','violet', 'darkkhaki', 'firebrick']
    for i in K:
        dfClusters_C1i = GetAClusterPoints(Clusters_Ci, i)
        ClusterPlotter(dfClusters_C1i, colors[i], Markr) # Random colur generation = np.random.uniform(0,1,3,)
        leg.append('Cluster'+str(i))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    # Put a legend below current axis
    ax.legend(leg, loc='upper center', bbox_to_anchor=(0.5, -0.11),
              fancybox=True, shadow=True, ncol=5)

## This function updates the centroids of a cluster by getting the mean points that are in that cluster (pts_InCluster)
def UpdateCentroids(Allpts_InCluster): 
    Mean = Allpts_InCluster.mean() 
    return Mean[:-1]

def Dist_a_Vect_Scalar(Matrix, xvect):
    diffVect = (Matrix - np.vstack(xvect))
    return LA.norm(diffVect, axis=0)

# Randomly sample k points of data as centroids -- step 1
def initializeK_Means(xx, k):
    N = len(xx[0])
    centroid_Idx = random.sample(range(N), k)
    return xx[:,centroid_Idx]
    
def RunK_Means(xx, k, Centroids, epoch, MaxEpoch): # gets data(xx) as the input and the "number" of clusters on the data(k)
    epoch = epoch + 1
    Clusters_pts = np.zeros((3,len(xx[0])))
    for i in range(len(xx[0])):
        DistVect = Dist_a_Vect_Scalar(Centroids, xx[:,i]) # Distance of each point to all obtained centroids -- step 2
        ArgMinDist = DistVect.argmin() # Index of the cluster
        Clusters_pts[0:2,i] = xx[:,i]
        Clusters_pts[2,i] = k[ArgMinDist]
    listClusters = GetSeparatelyClusterPoints(Clusters_pts, k) # list of lists
    counter=-1; OldCentroids = Centroids
    NewCentroids = np.zeros((2, len(Centroids[0])))
    for lbl in range(len(k)): ## lbl is the label
        counter=counter+1
        pointsInThisCluster = listClusters[lbl]
        NewCentroid_lbl = UpdateCentroids(pointsInThisCluster) # element of lbl of the listClusters
        NewCentroids[:, counter] = NewCentroid_lbl
    Centroids = NewCentroids[:,~np.all(np.isnan(NewCentroids), axis=0)]
    print(LA.norm(NewCentroids - OldCentroids))
    if (epoch < MaxEpoch) & (LA.norm(NewCentroids - OldCentroids) > 1e-6):
        RunK_Means(xx, k, Centroids, epoch, MaxEpoch)
    return Clusters_pts, Centroids        


def Output_Centroids_points(NumberInput, maxEpoch):
    X = Inpute(NumberInput)
    [d, C1, C2] = GetDesiredOutput(X)
    K1 = np.arange(10); K2 = np.arange(10,20,1) 
    #Clusters_C2 = K_Means(C2.T, 10)
    
    Centroids1 = initializeK_Means(C1.T, len(K1))
    [Clusters_C1, New_Centroids1] = RunK_Means(C1.T, K1, Centroids1, -1, maxEpoch)
    
    
    Centroids2 = initializeK_Means(C2.T, len(K2))
    [Clusters_C2, New_Centroids2] = RunK_Means(C2.T, K2, Centroids2, -1, maxEpoch)
    return X, C1, C2, d, K1, K2, Clusters_C1, New_Centroids1, Clusters_C2, New_Centroids2

NumberInput = 100; leg=[]; maxEpoch=1000
#[X, C1, C2, d, K1, K2, Clusters_C1, New_Centroids1, Clusters_C2, New_Centroids2] = Output_Centroids_points(NumberInput, maxEpoch)

