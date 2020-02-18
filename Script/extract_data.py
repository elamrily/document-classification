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
import time

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
        filename_w_ext = os.path.basename(element)
        filename, file_extension = os.path.splitext(filename_w_ext)
        trainfilename.append(filename[0:6])
        
        img = color.rgb2gray(io.imread(trainPATH + '/' + element))

        resize_image = misc.imresize(img,(500,250))
        Xtrain[i] = resize_image
        
    for i, element in enumerate (os.listdir(devPATH)):
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
    filedev = dfdev['Filename']
    ydevcsv = dfdev['Class']
    
    ytrain = []
    for i in range(len(trainfilename)):
        for j in range(len(filetrain)):   
            if trainfilename[i] in filetrain[j]:
                ytrain.append(ytraincsv[j])
    
    
    fichier = open(sys.argv[1] + "/missed_data.txt", "a")
    fichier.write("missed data in train png dataset \n")
    count = 0
    for i in range(len(filetrain)):
        if filetrain[i][:6] not in trainfilename:
            count+=1
            fichier.write(filetrain[i][:6]+'\n')
    fichier.write('number of element: ' + str(count) + '\n')
    count = 0
    fichier.write("\nmissed data in dev png dataset \n")
    for i in range(len(filedev)):
        if filedev[i][:6] not in devfilename:
            count+=1
            fichier.write(filedev[i][:6]+'\n')
    fichier.write('number of element: ' + str(count) + '\n')
    fichier.close() 
        
    
    ydev = []
    for i in range(len(devfilename)):
        for j in range(len(filedev)):   
            if devfilename[i] in filedev[j]:
                ydev.append(ydevcsv[j])
    
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
        for j in range(len(filetrain)):   
            if trainfilename[i] in filetrain[j]:
                ytrain.append(ytraincsv[j])
        
    filedev = dfdev['Filename']
    ydevcsv = dfdev['Language']
    ydev = []
    
    for i in range(len(devfilename)):
        for j in range(len(filedev)):   
            if devfilename[i] in filedev[j]:
                ydev.append(ydevcsv[j])
    
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
# python extract_data.py ../Data  ../Data/train2/train_png  ../Data/dev2/dev_png  ../Data/train2/xml  ../Data/dev2/xml
    
if __name__ == "__main__":
    # load dataframe from CSV file
    dftrain = pd.read_csv(sys.argv[1] + '/classes.csv', sep=",")
    dfdev = pd.read_csv(sys.argv[1] + '/classes_dev.csv', sep=",")
    # Path of images
    imgtrainPATH = sys.argv[2]
    imgdevPATH = sys.argv[3]
    # Path of xmls
    xmltrainPATH = sys.argv[4]
    xmldevPATH = sys.argv[5]
    # extract images
    print('------------------------------------------------------------------')
    print('Extracting images from train and dev folders')
    print('Used folders:\t', imgtrainPATH, '\t', imgdevPATH)
    print('!! This will take few minutes !!')
    t = time.time()
    Xtrain, Xdev, trainfilename, devfilename = extract_images(imgtrainPATH,
                                                               imgdevPATH)
    print('Execution time:', '{:^10.2f}'.format((time.time()-t)/60), ' min')
    # get class
    print('------------------------------------------------------------------')
    print('Getting class of images')
    print('Used folders:\t', sys.argv[1])
    print('!! This will take few minutes !!')
    t = time.time()
    ytrain_class, ydev_class = get_class(trainfilename,
                                         devfilename,
                                         dftrain,
                                         dfdev)
    print('missed data are saved in :\t', sys.argv[1])
    print('Execution time:', '{:^10.2f}'.format((time.time()-t)/60), ' min')
    # get_Language
    print('------------------------------------------------------------------')
    print('Getting language of images')
    print('Used folders:\t', sys.argv[1])
    print('!! This will take few minutes !!')
    t = time.time()
    ytrain_language, ydev_language = get_language(trainfilename,
                                                  devfilename,
                                                  dftrain,
                                                  dfdev)
    print('Execution time:', '{:^10.2f}'.format((time.time()-t)/60), ' min')
    # get_type
    print('------------------------------------------------------------------')
    print('Getting type of images')
    print('Used folders:\t', xmltrainPATH, '\t', xmldevPATH)
    print('!! This will take few minutes !!')
    t = time.time()
    ytrain_type, ydev_type = get_type(xmltrainPATH,
                                      xmldevPATH,
                                      trainfilename,
                                      devfilename)
    print('Execution time:', '{:^10.2f}'.format((time.time()-t)/60), ' min')
    # normalization
    Xtrain = (Xtrain-np.mean(Xtrain, axis=0))/np.std(Xtrain, axis=0)
    Xdev = (Xdev-np.mean(Xdev, axis=0))/np.std(Xdev, axis=0)
    # Saving data into npz file
    print('------------------------------------------------------------------')
    print('Saving data into npz file in:\t', sys.argv[1])
    filenpz = sys.argv[1] + '/data'
    t = time.time()
    np.savez(filenpz, Xtrain=Xtrain, ytrain_class=ytrain_class, Xdev=Xdev,
             ydev_class=ydev_class, ytrain_language=ytrain_language,
             ydev_language=ydev_language, ytrain_type=ytrain_type,
             ydev_type=ydev_type)