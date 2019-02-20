#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:58:07 2019

@author: garethwalker
"""
import requests
from PIL import Image
import pandas as pd
import os
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from torch.autograd import Variable
import re
import glob


planetApi = '6b7bf00d2c054fa787f9c3beaee46b9b'
GoogleAPI = '&scale=2&size=640x640&maptype=satellite&key=AIzaSyCecCmspXSwh2oHMNNqlW5ur4Yyuq6KOCQ'
URL = 'https://maps.googleapis.com/maps/api/staticmap?center='
zoomarg = '&zoom='
zoomval = 20
h=640
w=640
TestPad =  '34.076289, -118.380785'
MasterTileFolder = 'ScanData/MasterTiles/'
SubTileFolder = 'ScanData/SubTiles/Pad/'
data_dir = 'ScanData/SubTiles/'
detected_folder = 'ScanData/Detected/'

lat=float(TestPad.split(sep=',')[0])
lng=float(TestPad.split(sep=',')[1])


def get_image(coords):
    global file

    r = requests.get(URL+TestPad+zoomarg+str(zoomval)+GoogleAPI)

    if not os.path.exists(MasterTileFolder):
        os.makedirs(MasterTileFolder)
    f = open(MasterTileFolder+'Zoom'+str(zoomval)+'.png', 'wb')
    f.write(r.content)
    f.close()
    file = MasterTileFolder+'Zoom'+str(zoomval)+'.png'


#https://stackoverflow.com/questions/47106276/converting-pixels-to-latlng-coordinates-from-google-static-image
def getPointLatLng(x, y):


    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoomval + 8)
    degreesPerPixelY = 360 / math.pow(2, zoomval + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * ( y - h / 2)
    pointLng = lng + degreesPerPixelX * ( x  - w / 2)
    return (pointLat, pointLng) 


def sliding_window(window_ratio,step_ratio):
    global SweepBoxes
    global window
    global tilecount
    print('generating sliding window samples')

    im = Image.open(file)
    mastersize = im.size[1]
    window=mastersize*window_ratio
    shift=window*step_ratio
    totalsteps = int(im.size[1]/shift)
    
    y1=0
    x1=0
    x2=int(window)
    y2=int(window)
 
    
    
    
    for step in range(totalsteps):
        
        x1=0
        for step in range(totalsteps):
            
            SweepCentroidPixel = [(x1+x2)/2,(y1+y2)/2]
            SweepCentroidLatLng= getPointLatLng(SweepCentroidPixel[0]/2,SweepCentroidPixel[1]/2)
            SweepBoxes.loc[tilecount,'Boxes']       = [x1,y1,x2,y2]
            SweepBoxes.loc[tilecount,'PixelX']      = SweepCentroidPixel[0]
            SweepBoxes.loc[tilecount,'PixelY']      = SweepCentroidPixel[1]
            SweepBoxes.loc[tilecount,'Latitude']    = SweepCentroidLatLng[0]
            SweepBoxes.loc[tilecount,'Longitude']   = SweepCentroidLatLng[1]
            SweepBoxes.loc[tilecount,'MasterTile']  = file
            tile=im.crop((x1,y1,x2,y2))
            tile.save(SubTileFolder+'tile'+str(tilecount)+'.png')
            
                    

            x1=x1+shift
            x2=x1+window
            tilecount=tilecount+1
            
            
        y1=y1+shift
        y2=y1+window
            
            
       

def tile_that_pad(file,width):
    #image = 'png filename and location'
    #width = number of tiles in a row
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
        
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index  

def get_test_images():
    global classes
    #data = datasets.ImageFolder(data_dir, transform=test_transforms)
    data = ImageFolderWithPaths(data_dir, transform=test_transforms)
    classes = data.classes
   # indices = list(range(len(data)))
    #np.random.shuffle(indices)
    #idx = indices[:num]
    #from torch.utils.data.sampler import SubsetRandomSampler
    #sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, batch_size=tilecount)
    dataiter = iter(loader)
    images, labels, paths = dataiter.next()
    return images, labels, paths

class ImageFolderWithPaths(datasets.ImageFolder):
    #credit to : https://gist.githubusercontent.com/andrewjong/
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
def detect_pads ():
    global device
    global test_transforms
    global found_tile_index
    test_transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('aerialmodel.pth')
    model.eval()
    model

    
    
    padcount=0
    to_pil = transforms.ToPILImage()
    images, labels, paths = get_test_images()
    

    
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image)
        print(str(ii) + ' of ' + str(len(images)))
        if str(classes[index])=='Pad' :
            
            print ('Pad found')
            print (str(paths[ii]))
            padcount = padcount+1
            image.save(detected_folder+str(ii)+'.png')
            found_tile_index = (int((re.findall('\d+', paths[ii] )[0])))
            SweepBoxes.PadFound[found_tile_index] = 'Y'
        else:
            print ('Pad not found')
    CleanUp =  glob.glob(SubTileFolder+'*')
    for f in CleanUp:
        os.remove(f)
        
def plot_pads_found():
    for i in range(len(SweepBoxes.Latitude[SweepBoxes.PadFound.notnull()])):
        x = SweepBoxes.Latitude[SweepBoxes.PadFound.notnull()].iloc[i]
        y = SweepBoxes.Longitude[SweepBoxes.PadFound.notnull()].iloc[i]
        plt.ylim(SweepBoxes.Longitude[SweepBoxes.PadFound.notnull()].min(),SweepBoxes.Longitude[SweepBoxes.PadFound.notnull()].max())
        plt.xlim(SweepBoxes.Latitude[SweepBoxes.PadFound.notnull()].min(), SweepBoxes.Latitude[SweepBoxes.PadFound.notnull()].max())
        plt.scatter(x,y)
    
    plt.show()
    

def scan_master_tiles():
    global MasterTileCount
    global file
    global lat
    global lng
    global SweepBoxes
    global tilecount
    MasterTileDF = pd.read_csv(MasterTileFolder+'SampleWindows.csv')
    MasterTileCount = len(MasterTileDF.Filename)
    tilecount = 0
    SweepBoxes = pd.DataFrame(columns=['MasterTile','Boxes','PixelX','PixelY','Latitude','Longitude','PadFound'])
    for i in range(MasterTileCount):
        print ('Scanning Master Tile ' + str(i) + 'of' + str(MasterTileCount))
        file =  MasterTileDF.Filename[i]
        lat =  MasterTileDF.Lat[i]
        lng =  MasterTileDF.Long[i]
        sliding_window(0.33,0.1)
        detect_pads()
        SweepBoxes.to_csv('HelloPads.csv')
##from https://github.com/hrldcpr/mercator.py/blob/master/mercator.py
        
#for single image clasifcation on pretrained: https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411
#get_image(TestPad)
#sliding_window(0.33,0.1)  #super slow but shows detection
#detect_pads()
#plot_pads_found()


        


        