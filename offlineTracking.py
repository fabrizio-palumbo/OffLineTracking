
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:31:46 2021
@author: fabrizp
"""

import cv2 as cv
import h5py as h5
import numpy as np
import os as os
from os import listdir
from os.path import isfile, join
from numpy import swapaxes 
#from SavePositionFiles import *'

#open(cv.version)
#cv.__version__

#pathVideo = str("C:\\Users\\Bonnie\\Documents\\Personal\\CodeRefinery\\video\\") 
# Set a relative path video
folder = os.listdir(os.getcwd());
for P in folder: 
     Path=os.path.isdir(P)
     if Path :
         if P == 'video':
             pathVideo = [os.getcwd() + os.path.sep + P];
         

def GetRawFrame(pathVideo):
    # check packages
    # we now get all the files in the directory we are interested in
    videoFiles = [f for f in listdir(pathVideo) if isfile(join(pathVideo, f))]
    # check if it is h5
    Files = [join(pathVideo, name) for name in videoFiles]
    rawVideoInfo = h5.File(Files[0], 'r')
    rawVideo = (rawVideoInfo['frames'])
    rawFrame = np.array(rawVideo[:, :, :])
    rawFrame = rawFrame.swapaxes(2, 0).swapaxes(1, 2)[:, :, :]
    return rawFrame

def getImageObject(image):
    #check if image is binary
    elementErode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    elementDilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 3))  # averages over pixels
    fgMask = cv.erode(image, elementErode, iterations=1)
    fgMask = cv.dilate(fgMask, elementDilate, iterations=1)
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)       
    return contours

def calculateXY(cnts,cx,cy):   
    m = cv.moments(cnts) 
    if m["m00"] > 0:  # gets x and y of center of mass
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    return cx, cy
def getReasonableArea(contours,areaDetectedMean,areaDetectedTotal,counterFrame):
    areas = np.array([cv.contourArea(c) for c in contours])
    if len(areas) == 1:  # if there is only one area, we assume it is the fish
        areaDetectedTotal += areas
        areaDetectedMean = areaDetectedTotal/(counterFrame+1)
        detectedIndex = 0
    elif len(areas) > 1:  # if there is more than one area
        # takes the difference of all detected areas and the mean
        differences = np.abs(areas - areaDetectedMean)
        # takes index of the area which is closer to the mean area (fish)
        detectedIndex = differences.argmin()
        areaDetectedTotal += areas[detectedIndex];
        areaDetectedMean = areaDetectedTotal/(counterFrame+1)
        detectedIndex = 0
    return detectedIndex,areaDetectedTotal,areaDetectedMean

def ExtractPosition(rawFrame,history,shadow):
# substracts over 500 frames, threshold of 16, detect shadows is false
    backSub = cv.createBackgroundSubtractorMOG2(history, 16, shadow)
    areaDetectedMean = 0
    areaDetectedTotal = 0
    counterFrame = 0
    xPos = np.array([0])
    yPos = np.array([0])
    for Frames in rawFrame:
        # rearranges data to be in C mode, usable for cvopen functions
        frame = np.ascontiguousarray(Frames)
        # applies it on a frame, but fram is an object, so history is saved there
        fgMask = backSub.apply(frame)
        # integration of pixel, amoutn of change between pixels, makes edges more pronounced
        contours= getImageObject(fgMask)
        cx=xPos[-1];
        cy=yPos[-1];
        if contours: # lists all areas which have a countour
            detectedIndex,areaDetectedTotal,areaDetectedMean= getReasonableArea(contours,areaDetectedMean,areaDetectedTotal,counterFrame)
                 # if conoturs is non NAN,
            cnts = cv.drawContours(fgMask, contours[detectedIndex], -1, (0, 255, 0), 1)
            cx, cy = calculateXY(cnts, cx, cy)      
        xPos = np.append(xPos, cx)  # array with x position of fish
        yPos = np.append(yPos, cy)  # array with y position of fish
        counterFrame += 1
    return xPos, yPos


def SavePositionFiles(xPos, yPos,path):
    PosFile = open(path+'output_position.txt', "w")
    print(path)
    videoFps = 10
    timeLapsFrames = int(1000/videoFps)  # in milliseconds
    for i in range(xPos.shape[0]):
        PosFile.write("deltaT: "+str(timeLapsFrames)+" x: " +
                    str(xPos[i]) + " y: "+str(yPos[i]) + "\n")
    PosFile.close()
    return

def saveVideo(path, Video):
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    t, x, y=Video.shape
    out = cv.VideoWriter(path+'output_detected.avi', fourcc, 50.0, (y, x), False)
    out.write(Video.astype("uint8"))
    out.release()
    return


Video=GetRawFrame(pathVideo);
x,y=ExtractPosition(Video,500,False);
SavePositionFiles(x, y,pathVideo)

