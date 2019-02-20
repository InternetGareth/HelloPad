#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 19:28:15 2019

@author: garethwalker
"""
#### code to test final pad predictions against baseline of manually identified pads in test area (LA financial district)


import pandas as pd

def radius (x1,y1,x2,y2):
    r = ((x1-x2)**2+(y1-y2)**2)**0.5
    return r



PredDF = pd.read_csv('../PadCoordinates/HelloPad_LA-FC_SingleCluster2.csv')
PredDF = PredDF[PredDF['Data Source']=='HelloCopter - Cluster Analysis'].reset_index()[['Latitude', 'Longitude']]

BaseDF = pd.read_csv('../PadCoordinates/BaseCase 2.csv')
BaseDF[['Latitude','Longitude']]=BaseDF[['Y','X']]
BaseDF = BaseDF[['Latitude','Longitude']]

MatrixDF = pd.DataFrame()

#calibrate radius based on input
#this one is 100meters

Cal1 = [34.056128, -118.251834]
Cal2 = [34.055506, -118.251041]

r = radius(Cal1[0],Cal1[1],Cal2[0],Cal2[1])

#work through predictions for true and false positives

for i in range(len(PredDF)):
    for ii in range(len(BaseDF)):
        testRad = radius(PredDF['Latitude'][i],PredDF['Longitude'][i],BaseDF['Latitude'][ii],BaseDF['Longitude'][ii])
        if testRad <= r:
            PredDF.loc[i,'Confusion']= 'True Positive'
            break
        elif testRad > r:
            PredDF.loc[i,'Confusion'] = 'False Positive'
    
#work through basecase for any missed predictions (false negatives)

for i in range(len(BaseDF)):
    for ii in range(len(PredDF)):
        testRad=radius(BaseDF['Latitude'][i],BaseDF['Longitude'][i],PredDF['Latitude'][ii],PredDF['Longitude'][ii])
        if testRad <= r:
            BaseDF.loc[i,'Confusion']='True Positive'
            break
        elif testRad > r:
            BaseDF.loc[i,'Confusion']='False Negative'
    

TP = len(PredDF[PredDF['Confusion']=='True Positive'])
FP = len(PredDF[PredDF['Confusion']=='False Positive'])
FN = len(BaseDF[BaseDF['Confusion']=='False Negative'])

Precision = round(TP / (TP + FP),3)
Recall = round(TP / (TP + FN),3)

print('True Positives :' + str(TP))
print('False Positives :' + str(FP))
print('False Negatives :' + str(FN))



print('Precision: ' + str(Precision*100) + '%' )
print('Recall: ' + str(Recall*100) + '%' )


    
    