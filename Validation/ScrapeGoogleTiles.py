#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:08:16 2019

@author: garethwalker
"""

import requests
from PIL import Image
import math
import os
import pandas as pd

h=640
w=640
GoogleAPI = '&scale=2&size='+str(w)+'x'+str(h)+'&maptype=satellite&key=AIzaSyCecCmspXSwh2oHMNNqlW5ur4Yyuq6KOCQ'
URL = 'https://maps.googleapis.com/maps/api/staticmap?center='
zoomarg = '&zoom='
zoomval = 20


MasterTileFolder = 'ScanData/MasterTiles/'
SubTileFolder = 'ScanData/SubTiles/Pad/'
data_dir = 'ScanData/SubTiles/'
detected_folder = 'ScanData/Detected/'

#this test pad and lat lng values only used to precalculate the window width
TestPad =  '34.076289, -118.380785'
lat=float(TestPad.split(sep=',')[0])
lng=float(TestPad.split(sep=',')[1])

#Focussed LA Example With Pads
Start = [34.076503, -118.382241]
Finish = [34.073202, -118.376504]



def get_image(PadLat,PadLong):
    global file
    PadCoords = str(PadLat)+', '+str(PadLong)
    r = requests.get(URL+PadCoords+zoomarg+str(zoomval)+GoogleAPI)

    if not os.path.exists(MasterTileFolder):
        os.makedirs(MasterTileFolder)
    f = open(MasterTileFolder+'MasterTile'+str(samplecount)+'.png', 'wb')
    f.write(r.content)
    f.close()
    samplewindows.Filename[samplecount]= MasterTileFolder+'MasterTile'+str(samplecount)+'.png'

def getPointLatLng(x, y):

    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoomval + 8)
    degreesPerPixelY = 360 / math.pow(2, zoomval + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * ( y - h / 2)
    pointLng = lng + degreesPerPixelX * ( x  - w / 2)
    return (pointLat, pointLng) 

    
#get_image(TestPad)

def scrape_windows():
    global samplecount
    global samplewindows
    global LatWindows
    global LongWindows
    LatLen = Finish[0] - Start[0]
    LongLen= Finish[1] - Start[1]
    LatWindowLength = abs(getPointLatLng(0,w)[0]-getPointLatLng(w,0)[0])
    LongWindowLength= abs(getPointLatLng(0,w)[1]-getPointLatLng(w,0)[1])
    
    LongWindows = math.ceil(LongLen/LongWindowLength)
    LatWindows = math.ceil(abs((LatLen/LatWindowLength)))
    
    samplewindows = pd.DataFrame(index = range(LatWindows*LongWindows), columns=['Lat','Long','Filename'])
    
    
    samplecount = 0
    
    for i in range(LatWindows):
        for ii in range(abs(LongWindows)):
            print('Getting Master Tile ' + str(samplecount) + ' of ' + str(LatWindows*LongWindows))
            samplewindows.Long[samplecount] = Start[1] + LongWindowLength*ii
            samplewindows.Lat[samplecount]= Start[0] - LatWindowLength*i
            get_image(samplewindows.Lat[samplecount],samplewindows.Long[samplecount])
            samplecount = samplecount + 1
            samplewindows.to_csv(MasterTileFolder+'SampleWindows.csv')







