import cv2 as cv
import h5py as h5
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy import swapaxes 

# we now get all the files in the directory we are interested in 
path= np.str("C:\\Users\\fabri\\OneDrive\\Desktop\\offline_tracking\\example\\") #here you get the path of the folder in which your video is
videoFiles = [f for f in listdir(path) if isfile(join(path, f))]
Files=[join (path, name) for name in videoFiles ]
rawVideoInfo=h5.File(Files[0],'r')
videoFps=10;
timeLapsFrames=int(1000/videoFps);#in milliseconds
videoFiles = [f for f in listdir(path) if isfile(join(path, f))]
Files=[join (path, name) for name in videoFiles ]
rawVideoInfo=h5.File(Files[0],'r')
path=Files[0].replace('.h5','');
rawVideo=(rawVideoInfo['frames']);
rawFrame=np.array(rawVideo[:,:,:])
rawFrame=rawFrame.swapaxes(2,0).swapaxes(1,2)[:,:,:]
del rawVideo
backSub=cv.createBackgroundSubtractorMOG2(500, 16, False)
t, x, y=rawFrame.shape
framesMasked=[] 
areaTot= []
areaDetectedMean=0;
counterFrame=0
fourcc = cv.VideoWriter_fourcc( 'M','J','P','G')
out = cv.VideoWriter(path+'output_detected.avi', fourcc , 50.0, (y,x),False)
out_original = cv.VideoWriter(path+'raw_video.avi', fourcc , 50.0, (y,x),False)
xPos=np.array([0]); yPos=np.array([0]);
for Frames in rawFrame:
    frame =np.ascontiguousarray(Frames)
    fgMask = backSub.apply(frame)
    elementErode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    elementDilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (6,3))
    fgMask=cv.erode( fgMask, elementErode, iterations = 1)
    fgMask=cv.dilate( fgMask, elementDilate, iterations = 1)
    framesMasked.insert(counterFrame, fgMask)
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    areas=np.array([cv.contourArea(c) for c in contours])
    areaTot.insert(counterFrame, areas)
    
    if len(areas) == 1:
        areaDetectedMean+=areas
        areaDetectedMean=areaDetectedMean/(counterFrame+1)
        detectedIndex=0;
    elif len(areas)>1:
        differences= np.abs(areas - areaDetectedMean)
        detectedIndex = differences.argmin()
        cv.waitKey()
    if contours:         
        cnts = cv.drawContours(fgMask, contours[detectedIndex], -1, (0, 255, 0), 1)     
        m = cv.moments(cnts)
        if m["m00"] > 0: 
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
    xPos=np.append(xPos,cx);
    yPos=np.append(yPos,cy);
    fgMask=cv.circle(fgMask, (cx, cy), 4, (255), 1) 
    frame=cv.circle(frame, (cx, cy), 4, (255), 1)   
    if counterFrame>1000:
        for n in range((counterFrame-1000),counterFrame):
            fgMask=cv.circle(fgMask, (xPos[n], yPos[n]), 0, (255), 1)
            frame=cv.circle(frame, (xPos[n], yPos[n]), 0, (255), 1)
    else :
        for n in range(counterFrame):
            fgMask=cv.circle(fgMask, (xPos[n], yPos[n]), 0, (255), 1)
            frame=cv.circle(frame, (xPos[n], yPos[n]), 0, (255), 1)
    out.write(fgMask.astype("uint8"))
    out_original.write(frame.astype("uint8"))
    counterFrame +=1
PosFile = open(path+'output_position.txt', "w")
for i in range(xPos.shape[0]):
    PosFile.write("deltaT: "+str(timeLapsFrames)+"x: "+str(xPos[i])+ " y: "+str(yPos[i])+ "\n")
PosFile.close()
out.release()

## Christa was here