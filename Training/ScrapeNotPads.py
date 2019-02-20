# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:50:46 2019

@author: Gareth
"""

import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain
import glob
import random
from PIL import Image

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
        

URL = 'https://maps.googleapis.com/maps/api/staticmap?center='
API = '&zoom=20&scale=2&size=640x640&maptype=satellite&key=AIzaSyCecCmspXSwh2oHMNNqlW5ur4Yyuq6KOCQ'

NPads = len(glob.glob('Data\Pad\*.*'))
N = 

LA = (34.309786, -118.575607, 33.669074, -117.795862)

LALatRange = int((LA[0]-LA[2])*10**6)
LALongRange = int((LA[1]-LA[3])*10**6)

Lats = []
Longs = []
LatLongs = []


def tile_that_pad(file,width):
    #image = 'png filename and location'
    #width = number of tiles in a row, should be around 4
    global BoxInRow
    im = Image.open(file)
    mastersize = im.size[1]
    BoxInRow= width
    crop_size= mastersize/BoxInRow
    
    y1=0
    y2=crop_size
    tilen=0
    for col in range(BoxInRow):
        x1=0
        x2=crop_size
        for row in range(BoxInRow):
            tilen=tilen+1
            tile=im.crop((x1,y1,x2,y2))
            tilePixelCoord=[x1,y1,x2,y2]
            tile.save(scrape_dir+'Pad'+str(TileNumber)+'tile'+str(tilen)+'.png')
            x1=x1+crop_size
            x2=x2+crop_size
        y1=y1+crop_size
        y2=y2+crop_size
        
        
for pad in range(NPads):
    LatSample = LA[2]+(random.randint(0,LALatRange)*10**-6)
    LongSample =LA[3]+(random.randint(0,LALatRange)*10**-6)
    Lats.append(LatSample)
    Longs.append(LongSample)
    LatLongs.append(str(LatSample)+","+str(LongSample))



fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None, llcrnrlat=20, urcrnrlat=40, llcrnrlon=-120, urcrnrlon=-100, )
m.scatter(Longs,Lats, s=1,c='red')
draw_map(m)

for i in range(NPads):
    try:
        location = LatLongs[i]
        r = requests.get(URL+str(location)+API)
        f = open('not_helipad'+str(i)+'.png', 'wb')
        f.write(r.content)
        f.close()
        r = Image.open('not_helipad'+str(i)+'.png')
        r=r.crop((0,0,420,420))
        r.save('not_helipad'+str(i)+'_cropped.png')
        print('got image '+str(i))
    except Exception as e:
        print('had to skip '+str(i)+" : "+str(e))
        continue