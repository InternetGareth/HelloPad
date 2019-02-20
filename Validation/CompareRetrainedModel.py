#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:55:41 2019

@author: garethwalker
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def get_centroid(cluster):
  #strictly speaking, DBSCAN does not have a 'centroid', but for the purposes of a geographic coordinates, this is a good aproximation
  
  cluster_ary = np.asarray(cluster)
  centroid = cluster_ary.mean(axis = 0)
  return centroid

def cluster(DataF):
    Model = DataF.Model.iloc[0]
    LatLongdf = DataF[['Latitude','Longitude']]
    PadArray = DataF[['Latitude','Longitude']].values
    rad = 0.0001
    db = DBSCAN(eps=rad, min_samples=5).fit(PadArray)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    cdbsc_clusters = pd.Series([LatLongdf[labels==n] for n in range(n_clusters_)])
    centroids = cdbsc_clusters.map(get_centroid)
    CentDF = pd.DataFrame(columns=['Latitude','Longitude','Model','Centroid'])
    for i in range(len(centroids)):
        CentDF.loc[i,'Latitude']= centroids[i][0]
        CentDF.loc[i,'Longitude']= centroids[i][1]
        CentDF.loc[i,'Model']= Model
        CentDF.loc[i,'Centroid']= 'Y'
    
    return CentDF

    
    
    
    
original = pd.read_csv('HelloPads.csv')
original = original[original['PadFound']=='Y']
original['Model']='ResNet50 - equal negative cases'
clusters = cluster(original)
originalWC= pd.concat([original,clusters],ignore_index=True, sort=False)


new = pd.read_csv('HelloPads_retrained.csv')
new = new[new['PadFound']=='Y']
new['Model'] = 'ResNet50 - 150% negative cases'
clusters = cluster(new)
newWC= pd.concat([new,clusters],ignore_index=True, sort=False)

trainingEG= pd.concat([originalWC,newWC],ignore_index=True, sort=False)
trainingEG.reset_index()
trainingEG.to_csv('TraingEgs.csv')






