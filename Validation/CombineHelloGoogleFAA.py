#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:07:27 2019

@author: garethwalker
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


def get_centroid(cluster):
  #strictly speaking, DBSCAN does not have a 'centroid', but for the purposes of a geographic coordinates, this is a good aproximation
  
  cluster_ary = np.asarray(cluster)
  centroid = np.median(cluster_ary, axis = 0) # luster_ary.mean(axis = 0)
  return centroid

def cluster(DataF):
    
    LatLongdf = DataF[['Latitude','Longitude']]
    PadArray = DataF[['Latitude','Longitude']].values
    db = DBSCAN(eps=rad, min_samples=threshold).fit(PadArray)
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
        CentDF.loc[i,'Centroid']= 'Y'
        CentDF.loc[i,'Cluster Radius'] = rad
        CentDF.loc[i,'Cluster Threshold'] = threshold
        CentDF.loc[i,'Data Source'] = 'HelloPad - Cluster Analysis'
    
    return CentDF





def generate_FAA():
    FAADF = pd.read_csv('PadCoordinates/USHP.csv',low_memory=False)
    heliports = FAADF[FAADF['Type']=='HELIPORT']
    
    LatDeg = heliports['ARPLatitude'].str.slice(start = 0, stop = 2)
    LatDeg = LatDeg.astype(float)
    LatMin = heliports['ARPLatitude'].str.slice(start = 3, stop = 5)
    LatMin = LatMin.astype(float)
    LatSec = heliports['ARPLatitude'].str.slice(start = 6, stop = 13)
    LatSec = LatSec.astype(float)
    
    LongDeg = heliports['ARPLongitude'].str.slice(start = 0, stop = 3)
    LongDeg = LongDeg.astype(float)
    LongMin = heliports['ARPLongitude'].str.slice(start = 4, stop = 6)
    LongMin = LongMin.astype(float)
    LongSec = heliports['ARPLongitude'].str.slice(start = 7, stop = 14)
    LongSec = LongSec.astype(float)
    
    
    LatDec = LatDeg + (LatMin /60) + (LatSec / 3600)
    LongDec = -(LongDeg + (LongMin /60) + (LongSec / 3600))
    
    FAADF = pd.DataFrame()
    
    FAADF['Latitude']=LatDec
    FAADF['Longitude']=LongDec
    FAADF['Data Source'] = 'FAA'  
    
    return FAADF

def generate_Google():
    googleDF = pd.read_csv('PadCoordinates/GoogleResults.csv')
    googleDF = googleDF[['Latitude', 'Longitude']]
    googleDF['Data Source'] = 'Google Maps'
    return googleDF



def generate_HelloPad():
    global threshold
    global rad
    print('Reading HelloPads')
    PadsDF = pd.read_csv('PadCoordinates/HelloPads-LA_FC_0_260.csv')
    PadsDF =  PadsDF[PadsDF['PadFound']=='Y']
    PadsDF['Data Source'] = 'HelloPad'
    rad = 0.0001
    threshold = 20
    ClusterDF = cluster(PadsDF)
    PadsDF = pd.concat([PadsDF,ClusterDF],ignore_index=True, sort=False)
    return PadsDF

# this code is for parameter tuning and sits within generate_HelloPad
    
#  ThresholdRange = 10
#   RadRange = 5
#   RadItter = (1*(10**-4))/10
 
#    for i in range(RadRange):
#        rad = (1*10**-4) + (RadItter*i)
#        print('Rad :' +str(rad))
#        for ii in range(ThresholdRange):
#            ClusterDF = cluster(PadsDF)
#            threshold = (ii*5)
#            print('Threshold: ' + str(threshold))
#            PadsDF = pd.concat([PadsDF,ClusterDF],ignore_index=True, sort=False)
#    return PadsDF


baseDF=pd.read_csv('PadCoordinates/BaseCase.csv')
baseDF['Data Source'] = 'Manual ID'
baseDF['Latitude'] = baseDF['Y']
baseDF['Longitude'] = baseDF['X']
baseDF= baseDF[['Data Source','Latitude','Longitude']]




PadsDF = generate_HelloPad()
FAADF =  generate_FAA()
GoogleDF = generate_Google()


FinalDB= pd.concat([PadsDF,FAADF,GoogleDF,baseDF],ignore_index=True, sort=False)
FinalDB = FinalDB.reset_index(drop=True)
FinalDB.to_csv('PadCoordinates/HelloPad_LA-FC_Cluster.csv')

