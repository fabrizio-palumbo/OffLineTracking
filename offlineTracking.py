import numpy as np
import cv2 as cv
import h5py as h5
from os import listdir
from os.path import isfile, join
#from SavePositionFiles import *

pathVideo = str("/home/annianni/Documents/ExampleCourseHackathon/videos")

def GetRawFrame(pathVideo):
    # check packages
    # we now get all the files in the directory we are interested in
    videoFiles = [f for f in listdir(pathVideo) if isfile(join(pathVideo, f))]
    # check if it is h5
    Files = [join(pathVideo, name) for name in videoFiles]
    rawVideoInfo = h5.File(Files[0], 'r')
    path = Files[0].replace('.h5', '')
    rawVideo = (rawVideoInfo['frames'])
    rawFrame = np.array(rawVideo[:, :, :])
    rawFrame = rawFrame.swapaxes(2, 0).swapaxes(1, 2)[:, :, :]
    return rawFrame, path

def GetVideo(rawFrame, path):
# substracts over 500 frames, threshold of 16, detect shadows is false
    backSub = cv.createBackgroundSubtractorMOG2(500, 16, False)
    t, x, y = rawFrame.shape
    framesMasked = []
    areaTot = []
    areaDetectedMean = 0
    areaDetectedTotal = 0
    counterFrame = 0
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv.VideoWriter(path+'output_detected.avi', fourcc, 50.0, (y, x), False)
    out_original = cv.VideoWriter(
        path+'raw_video.avi', fourcc, 50.0, (y, x), False)
    xPos = np.array([0])
    yPos = np.array([0])
    for Frames in rawFrame:
        # rearranges data to be in C mode, usable for cvopen functions
        frame = np.ascontiguousarray(Frames)
        # applies it on a frame, but fram is an object, so history is saved there
        fgMask = backSub.apply(frame)
        # integration of pixel, amoutn of change between pixels, makes edges more pronounced
        elementErode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        elementDilate = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (6, 3))  # averages over pixels
        fgMask = cv.erode(fgMask, elementErode, iterations=1)
        fgMask = cv.dilate(fgMask, elementDilate, iterations=1)
        framesMasked.insert(counterFrame, fgMask)

        contours, hierarchy = cv.findContours(
            fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # lists all areas which have a countour
        areas = np.array([cv.contourArea(c) for c in contours])
        areaTot.insert(counterFrame, areas)

        if len(areas) == 1:  # if there is only one area, we assume it is the fish
            areaDetectedTotal += areas
            areaDetectedMean = areaDetectedTotal/(counterFrame+1)
            detectedIndex = 0
        elif len(areas) > 1:  # if there is more than one area
            # takes the difference of all detected areas and the mean
            differences = np.abs(areas - areaDetectedMean)
            # takes index of the area which is closer to the mean area (fish)
            detectedIndex = differences.argmin()
            cv.waitKey()  # to check what is happening

        if contours:         # if conoturs is non NAN,
            cnts = cv.drawContours(fgMask, contours[detectedIndex], -1, (0, 255, 0), 1)
            m = cv.moments(cnts) 
            if m["m00"] > 0:  # gets x and y of center of mass
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

        xPos = np.append(xPos, cx)  # array with x position of fish
        yPos = np.append(yPos, cy)  # array with y position of fish
        # plots a circle on the mask
        fgMask = cv.circle(fgMask, (cx, cy), 4, (255), 1)
        # plots a circle on the frame
        frame = cv.circle(frame, (cx, cy), 4, (255), 1)

        if counterFrame > 1000:  # path history is plotted, 1000 frames prior then disappears
            for n in range((counterFrame-1000), counterFrame):
                fgMask = cv.circle(fgMask, (xPos[n], yPos[n]), 0, (255), 1)
                frame = cv.circle(frame, (xPos[n], yPos[n]), 0, (255), 1)
        else:
            # if there are less than 1000 frames yet, beginnign of video
            for n in range(counterFrame):
                fgMask = cv.circle(fgMask, (xPos[n], yPos[n]), 0, (255), 1)
                frame = cv.circle(frame, (xPos[n], yPos[n]), 0, (255), 1)

        out.write(fgMask.astype("uint8"))
        out_original.write(frame.astype("uint8"))
        counterFrame += 1

    return xPos, yPos, out


def SavePositionFiles(xPos, yPos, out,path):
    PosFile = open(path+'output_position.txt', "w")
    print(path)
    videoFps = 10
    timeLapsFrames = int(1000/videoFps)  # in milliseconds
    for i in range(xPos.shape[0]):
        PosFile.write("deltaT: "+str(timeLapsFrames)+"x: " +
                    str(xPos[i]) + " y: "+str(yPos[i]) + "\n")
    PosFile.close()
    out.release()

(rawFrame, path) = GetRawFrame(path)
(xPos, yPos, out) = GetVideo(rawFrame, path)
SavePositionFiles(xPos, yPos, out,path)

