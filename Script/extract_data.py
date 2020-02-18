# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:45:23 2019

@author: Ilyas
"""

import numpy as np
import pandas as pd
from skimage import io
from scipy import misc
import os
from skimage import color
import xml.etree.ElementTree as ET
import sys

def extract_images(trainPATH,devPATH):

    ''' Input : - trainPATH : path of the training images
                - devPATH : path of validation images 
        Output: - Xtrain : training matrix of resized training images
                - Xdev : validation matrix of resized validation images
                - trainfilename : name of all images in train folder
                - devfilename : name of all images in dev folder
        '''
    trainfilename = []
    devfilename = []
    Xtrain = np.zeros(((5483,500,250)))
    Xdev = np.zeros(((1110,500,250)))
    
    for i, element in enumerate (os.listdir(trainPATH)):
        print(i)
        filename_w_ext = os.path.basename(element)
        filename, file_extension = os.path.splitext(filename_w_ext)
        trainfilename.append(filename[0:6])
        
        img = color.rgb2gray(io.imread(trainPATH + '/' + element))

        resize_image = misc.imresize(img,(500,250))
        Xtrain[i] = resize_image
        
    for i, element in enumerate (os.listdir(devPATH)):
        print(i)
        filename_w_ext = os.path.basename(element)
        filename, file_extension = os.path.splitext(filename_w_ext)
        devfilename.append(filename[0:6])
                
        img = color.rgb2gray(io.imread(devPATH + '/' + element))
        
        resize_image = misc.imresize(img,(500,250))
        Xdev[i] = resize_image
    
    return(Xtrain,Xdev ,trainfilename , devfilename)

def get_class(trainfilename, devfilename, dftrain, dfdev):
    
    
    ''' Input : - trainfilename : name of all images in train folder
                - devfilename : name of all images in dev folder
                - dftrain : dataframe of train csv file 
                - dfdev : dataframe of dev csv file  
        Output: - ytrain_final : vector of training class labels 
                - ydev_final : vector of dev class labels 
        '''
        
    filetrain = dftrain['Filename']
    ytraincsv = dftrain['Class']
    ytrain = []
    
    for i in range(len(trainfilename)):
        print(i)
        for j in range(len(filetrain)):   
            if trainfilename[i] in filetrain[j]:
                indtrain = np.where(filetrain == '../Data/train2/txt/' + trainfilename[i] + '.txt')[0][0]
                ytrain.append(ytraincsv[indtrain])
        
    filedev = dfdev['Filename']
    ydevcsv = dfdev['Class']
    ydev = []
    
    for i in range(len(devfilename)):
        print(i)
        for j in range(len(filedev)):   
            if devfilename[i] in filedev[j]:
                inddev = np.where(filedev == '../Data/dev2/txt/' + devfilename[i] + '.txt')[0][0]
                ydev.append(ydevcsv[inddev])
    
    classes = ['C1','C2','C3','C4','C5']
    
    ytrain_final = np.zeros(len(ytrain))
    ydev_final = np.zeros(len(ydev))
    
    for i in range(len(ytrain)):
        ytrain_final[i] = int(classes.index(ytrain[i]))
        
    for i in range(len(ydev)):
        ydev_final[i] = int(classes.index(ydev[i]))
    
    return(ytrain_final,ydev_final)
    
def get_language(trainfilename, devfilename, dftrain, dfdev):
    
    ''' Input : - trainfilename : name of all images in train folder
                - devfilename : name of all images in dev folder
                - dftrain : dataframe of train csv file 
                - dfdev : dataframe of dev csv file  
        Output: - ytrain_final : vector of training language labels 
                - ydev_final : vector of dev language labels 
        '''
        
    filetrain = dftrain['Filename']
    ytraincsv = dftrain['Language']
    ytrain = []
    
    for i in range(len(trainfilename)):
        print(i)
        for j in range(len(filetrain)):   
            if trainfilename[i] in filetrain[j]:
                indtrain = np.where(filetrain == '../Data/train2/txt/' + trainfilename[i] + '.txt')[0][0]
                ytrain.append(ytraincsv[indtrain])
        
    filedev = dfdev['Filename']
    ydevcsv = dfdev['Language']
    ydev = []
    
    for i in range(len(devfilename)):
        print(i)
        for j in range(len(filedev)):   
            if devfilename[i] in filedev[j]:
                inddev = np.where(filedev == '../Data/dev2/txt/' + devfilename[i] + '.txt')[0][0]
                ydev.append(ydevcsv[inddev])
    
    classes = ['fr','ar','en']
    
    ytrain_final = np.zeros(len(ytrain))
    ydev_final = np.zeros(len(ydev))
    
    for i in range(len(ytrain)):
        ytrain_final[i] = int(classes.index(ytrain[i]))
        
    for i in range(len(ydev)):
        ydev_final[i] = int(classes.index(ydev[i]))
    
    return(ytrain_final,ydev_final)

def get_type(trainPATH,devPATH,trainfilename,devfilename):

    ''' Input : - trainfilename : name of all images in train folder
                - devfilename : name of all images in dev folder
                - dftrain : dataframe of train csv file 
                - dfdev : dataframe of dev csv file  
        Output: - ytrain_final : vector of training language labels 
                - ydev_final : vector of dev language labels 
        '''
        
    ## Train data
    documents_type = []
    filenames_train = []
    for i, element in enumerate (os.listdir(trainPATH)):
        print(i)
        filename_w_ext = os.path.basename(element)
        filename, file_extension = os.path.splitext(filename_w_ext)
        filenames_train.append(filename)
        path = trainPATH + '/' + element
        
        tree = ET.parse(path)
        root = tree.getroot()
        ns = {'xmlns': 'http://lamp.cfar.umd.edu/media/projects/GEDI/'}

        for DL_DOCUMENT in root.findall('xmlns:DL_DOCUMENT', ns):
            DL_PAGE = DL_DOCUMENT.find('xmlns:DL_PAGE', ns)
            script = []
            for DL_ZONE in DL_PAGE.findall('xmlns:DL_ZONE', ns):
                if DL_ZONE.get("script") != None:
                    script.append(DL_ZONE.get("script"))
            if len(np.unique(script).tolist()) == 1:
                documents_type.append(np.unique(script).tolist()[0])
            else:
                documents_type.append('mixte')
    
    typedoc = ['typed','hand','mixte']
    ytrain = np.zeros(len(trainfilename))
    for i in range(len(trainfilename)):
        ytrain[i] = typedoc.index(documents_type[filenames_train.index(trainfilename[i])])
    
    ## Dev data
    documents_type = []
    filenames_dev = []
    for i, element in enumerate (os.listdir(devPATH)):
        print(i)
        filename_w_ext = os.path.basename(element)
        filename, file_extension = os.path.splitext(filename_w_ext)
        filenames_dev.append(filename)
        path = devPATH  + '/' + element
        
        tree = ET.parse(path)
        root = tree.getroot()
        ns = {'xmlns': 'http://lamp.cfar.umd.edu/media/projects/GEDI/'}

        for DL_DOCUMENT in root.findall('xmlns:DL_DOCUMENT', ns):
            DL_PAGE = DL_DOCUMENT.find('xmlns:DL_PAGE', ns)
            script = []
            for DL_ZONE in DL_PAGE.findall('xmlns:DL_ZONE', ns):
                if DL_ZONE.get("script") != None:
                    script.append(DL_ZONE.get("script"))
            if len(np.unique(script).tolist()) == 1:
                documents_type.append(np.unique(script).tolist()[0])
            else:
                documents_type.append('mixte')
    
    ydev = np.zeros(len(devfilename))
    for i in range(len(devfilename)):
        ydev[i] = typedoc.index(documents_type[filenames_dev.index(devfilename[i])])
    
    return(ytrain,ydev)

    
#%% 

# For instance, we can pass arguments through the command line :
# python extract_data.py ../Data/classes.csv  ../Data/classes_dev.csv  ../Data/train2/train_png  ../Data/dev2/dev_png  ../Data/train2/xml  ../Data/dev2/xml
    
if __name__ == "__main__":
    # load dataframe from CSV file
    dftrain = pd.read_csv(sys.argv[1] + '/classes.csv', sep="\t")
    dfdev = pd.read_csv(sys.argv[1] + '/classes_dev.csv', sep="\t")
    
    # Path of images
    imgtrainPATH = sys.argv[2]
    imgdevPATH = sys.argv[3]
    
    # Path of xmls
    xmltrainPATH = sys.argv[4]
    xmldevPATH = sys.argv[5]
    
    # extract images
    Xtrain, Xdev, trainfilename , devfilename = extract_images(imgtrainPATH,imgdevPATH)
    
    # get class
    ytrain_class, ydev_class = get_class(trainfilename, devfilename, dftrain, dfdev)
    
    # get_lLanguage
    ytrain_language, ydev_language = get_language(trainfilename, devfilename, dftrain, dfdev)
    
    # get_type
    ytrain_type, ydev_type = get_type(xmltrainPATH,xmldevPATH,trainfilename,devfilename)
    
    # normalization
    Xtrain=(Xtrain-np.mean(Xtrain, axis=0))/np.std(Xtrain, axis=0)
    Xdev=(Xdev-np.mean(Xdev, axis=0))/np.std(Xdev, axis=0)
    
    # Saving data into npz file
    filenpz = sys.argv[1] + '/data'
    np.savez(filenpz,Xtrain=Xtrain,ytrain_class=ytrain_class,Xdev=Xdev,ydev_class=ydev_class, ytrain_language=ytrain_language,ydev_language=ydev_language,ytrain_type=ytrain_type,ydev_type=ydev_type)
