# System imports
from os import path
import math

# Third-Party Imports
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from common import *

## Helper function to do butterworth filtering
def butterWorthFiltering(lowpass1, lowpass2, pyr_prev, framePyramids, wl, wh, samplingRate, order=1):
    pyr_levels = len(framePyramids)
    normalization = samplingRate / 2
    wlNormalized = wl / normalization
    whNormalized = wh / normalization 
    low_a, low_b = signal.butter(order, wlNormalized, btype='lowpass', analog=False)
    high_a, high_b = signal.butter(order, whNormalized, btype='lowpass', analog=False)
    new_lowpass1, new_lowpass2, new_pyr_prev , filtered = [] , [] , [] , []
    for level in range(pyr_levels):
        filtered.append(np.zeros((framePyramids[level].shape), dtype="float32"))
        new_lowpass1.append(np.zeros((framePyramids[level].shape), dtype="float32"))
        new_lowpass2.append(np.zeros((framePyramids[level].shape), dtype="float32"))
        new_pyr_prev.append(np.zeros((framePyramids[level].shape), dtype="float32"))
    for level in range(pyr_levels):
        new_lowpass1[level][:,:,:] = ((-1*high_b[1])*lowpass1[level][:,:,:]) + (high_a[0]*framePyramids[level][:,:,:]) + (high_a[1]*pyr_prev[level][:,:,:] / high_b[0])
                                            
        new_lowpass2[level][:,:,:] =   ((-1*low_b[1])*lowpass2[level][:,:,:]) + (low_a[0]*framePyramids[level][:,:,:]) + (low_a[1]*pyr_prev[level][:,:,:] / low_b[0])
        
        filtered[level][:,:,:] = new_lowpass1[level][:,:,:] - new_lowpass2[level][:,:,:]
    new_pyr_prev = framePyramids
    return new_lowpass1, new_lowpass2, new_pyr_prev, filtered

def idealBandPassFiltering(framePyramids, sizeGuidance, wl, wh, samplingRate):
    # Assumed that framePyramids is for all frames at once
    pyramidLvls = len(framePyramids)
    frameCount = framePyramids[0].shape[0]
    totalPyramidSize = 0
    for level in range(pyramidLvls):
        totalPyramidSize += (sizeGuidance[level][0] * sizeGuidance[level][1])
    framePyramidsNp = np.zeros((frameCount,totalPyramidSize,3), dtype="float32")
    prevSize = 0
    for level in range(pyramidLvls):
        currSize = sizeGuidance[level][0]*sizeGuidance[level][1]
        framePyramidsNp[:,prevSize:prevSize+currSize,:] = np.reshape(framePyramids[level], (frameCount, sizeGuidance[level][0]*sizeGuidance[level][1], 3))
        prevSize += currSize

    n = framePyramidsNp.shape[0]
    Freq = np.arange(n, dtype="float32")
    Freq = (Freq/n)*samplingRate
    tempmask = np.where(Freq > wl , Freq, 0)
    tempmask = np.where(Freq < wh, tempmask , 0)
    tempmask = np.where(tempmask == 0 , tempmask, 1)
    tempmask = np.reshape(tempmask,(n,1))
    tempmask = np.tile(tempmask, (1,framePyramidsNp.shape[1]))
    mask = np.zeros((framePyramidsNp.shape[0], framePyramidsNp.shape[1], framePyramidsNp.shape[2]), dtype="float32")
    for i in range(mask.shape[2]):
        mask[:,:,i] = tempmask
    F = np.fft.fft(framePyramidsNp, axis=0)
    F = np.where(mask != 0, F, 0)
    filtered = np.real(np.fft.ifft(F, axis=0))
    finalFiltered = []
    prevSize = 0
    for level in range(pyramidLvls):
        currSize = sizeGuidance[level][0]*sizeGuidance[level][1]
        reshapeSize = (frameCount, sizeGuidance[level][0], sizeGuidance[level][1], sizeGuidance[level][2])
        finalFiltered.append(np.reshape(filtered[:,prevSize:prevSize+currSize,:], reshapeSize))
        prevSize += currSize
    return finalFiltered

## Helper function to magnify a filtered frame    
def pixelMagnification(filteredInput, alpha, lambda_c):
    ## page 4, figure 6 of the paper
    delta = lambda_c / 8 / (1+alpha)
    exaggeration_factor = 2
    vidHeight = filteredInput[0].shape[0]
    vidWidth = filteredInput[0].shape[1]
    pyr_levels = len(filteredInput)
    magnified = [0] * pyr_levels
    for level in range(pyr_levels):
        magnified[level] = np.copy(filteredInput[level])
    calculated_lambda = (((vidHeight ** 2) + (vidWidth ** 2))**0.5)/3
    for level in range(pyr_levels-1, -1,-1):
        currAlpha = ((calculated_lambda/delta)/8) - 1
        currAlpha = currAlpha * exaggeration_factor
        if level == 0 or level == pyr_levels-1:
            magnified[level][:,:,:] = 0
        elif currAlpha > alpha:
            magnified[level][:,:,:] = alpha * filteredInput[level][:,:,:]
        else:
            magnified[level][:,:,:] = currAlpha * filteredInput[level][:,:,:]
        calculated_lambda = calculated_lambda / 2
    return magnified


def videoMagnificationButterWorthFilter(fileToProcess, alpha, lambda_c, samplingRate, chromeAttenuation, lowFreq, highFreq):

    datadir = os.getcwd()
    samplesFn = datadir + "/samples"
    resultsFn = datadir + "/results/eulerian"
    vidFn = fileToProcess.split(".")[0]
    resParentFn = resultsFn + "/" + vidFn
    vidFramesFn = resParentFn + "/videoFramesOrigin"
    reconstructedFn = resParentFn + "/magnifiedFrames"

    ## create results folder according to processed file
    if os.path.exists(resParentFn) == False:
        os.makedirs(resParentFn)
    if os.path.exists(vidFramesFn) == False:
        os.makedirs(vidFramesFn)
    if os.path.exists(reconstructedFn) == False:
        os.makedirs(reconstructedFn)

    ## load video file into frames np array
    ## save frames to disk as well
    videoInputFn    = cv2.VideoCapture(samplesFn + "/" + fileToProcess)
    frameCount      = int(videoInputFn.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight     = int(videoInputFn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth      = int(videoInputFn.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameChannels   = 3
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(resultsFn + "/" + vidFn + "_Eulerian_magnified_alpha" + str(alpha)+ ".mp4", four_cc, 30, (frameWidth, frameHeight), 1)

    frames = np.zeros((frameCount, frameHeight, frameWidth, frameChannels), dtype="float32")
    # framesReconstructedRGB = np.zeros((frameCount, frameHeight, frameWidth, frameChannels), dtype="float32")
    success, image = videoInputFn.read()
    frameIndex = 1
    print("Processing", fileToProcess , "...")
    print("Extracting original video frames...")
    while success:
        frames[frameIndex-1,:,:,:] = np.float32(image/255.0)
        cv2.imwrite(vidFramesFn + "/frame%04d.png" %(frameIndex), image)
        success, image = videoInputFn.read()
        frameIndex += 1
    print("Extracted %d frames" %(frameIndex-1))

    pyramidLvls = 7
    guidanceFrame = cv2.imread(vidFramesFn+"/frame0001.png")
    _, sizeGuidance = buildLaplacian(guidanceFrame,levels=pyramidLvls)

    print("Started filtering & magnification...")
    # progress is used for cosmetic display of how much of the video is done
    progress = np.linspace(0,frameCount-1, 11, dtype="int32")
    for frameIdx in range(frameCount):
        frameYIQ = convertBGR2YIQ(frames[frameIdx])
        framePyramids = []
        for level in range(pyramidLvls):
            lvlHeight = sizeGuidance[level][0]
            lvlWidth = sizeGuidance[level][1]
            framePyramids.append(np.zeros((lvlHeight, lvlWidth, frameChannels), dtype="float32"))
        currPyramid,_ = buildLaplacian(frameYIQ,levels=pyramidLvls)
        for level in range(pyramidLvls):
            framePyramids[level][:,:,:] = currPyramid[level]
        if frameIdx == 0:
            lowpass1 = framePyramids
            lowpass2 = framePyramids
            pyr_prev = framePyramids
            saveVideo(frames[frameIdx,:,:,:], reconstructedFn, writer, frameIdx)
        else:
            lowpass1, lowpass2, pyr_prev, filteredLaplacian = butterWorthFiltering(lowpass1, lowpass2, pyr_prev, framePyramids, lowFreq,highFreq, samplingRate)
            pixelsMagnified = pixelMagnification(filteredLaplacian, alpha, lambda_c)
            reconstructedFrame = reconstructVideoFrames(pixelsMagnified, chromeAttenuation=chromeAttenuation)
            reconstructedFrame += frameYIQ
            frameReconstructedRGB = convertYIQ2RGB(reconstructedFrame[:,:,:])
            saveVideo(frameReconstructedRGB, reconstructedFn, writer, frameIdx)
            if frameIdx in progress:
                print("Completed" , np.where(progress==frameIdx)[0][0] * 10 ,"%" , "of the video")
    writer.release()
    print("Done...")

def videoMagnificationIdealFilter(fileToProcess, alpha, lambda_c, samplingRate, chromeAttenuation, lowFreq, highFreq):

    datadir = os.getcwd()
    samplesFn = datadir + "/samples"
    resultsFn = datadir + "/results/eulerian"
    vidFn = fileToProcess.split(".")[0]
    resParentFn = resultsFn + "/" + vidFn
    vidFramesFn = resParentFn + "/videoFramesOrigin"
    reconstructedFn = resParentFn + "/magnifiedFrames"
    
    ## create results folder according to processed file
    if os.path.exists(resParentFn) == False:
        os.makedirs(resParentFn)
    if os.path.exists(vidFramesFn) == False:
        os.makedirs(vidFramesFn)
    if os.path.exists(reconstructedFn) == False:
        os.makedirs(reconstructedFn)

    ## load video file into frames np array
    ## save frames to disk as well
    videoInputFn    = cv2.VideoCapture(samplesFn + "/" + fileToProcess)
    frameCount      = int(videoInputFn.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight     = int(videoInputFn.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth      = int(videoInputFn.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameChannels   = 3
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(resultsFn + "/" + vidFn + "_Eulerian_magnified_alpha" + str(alpha) + "_" +str(lowFreq) + "to" + str(highFreq) +".mp4", four_cc, 30, (frameWidth, frameHeight), 1)

    frames = np.zeros((frameCount, frameHeight, frameWidth, frameChannels), dtype="float32")
    success, image = videoInputFn.read()
    frameIndex = 1
    print("Processing", fileToProcess , "...")
    print("Extracting original video frames...")
    while success:
        frames[frameIndex-1,:,:,:] = np.float32(image/255.0)
        cv2.imwrite(vidFramesFn + "/frame%04d.png" %(frameIndex), image)
        success, image = videoInputFn.read()
        frameIndex += 1
    print("Extracted %d frames" %(frameIndex-1))

    framesYIQ = np.zeros((frameCount, frameHeight, frameWidth, frameChannels), dtype="float32")
    for frame in range(frameCount):
        framesYIQ[frame,:,:,:] = convertBGR2YIQ(frames[frame,:,:,:]) 

    pyramidLvls = 7
    testFrame = cv2.imread(vidFramesFn+"/frame0001.png")
    _, sizeGuidance = buildLaplacian(testFrame,levels=pyramidLvls)
    framePyramids = []
    for level in range(pyramidLvls):
        lvlHeight = sizeGuidance[level][0]
        lvlWidth = sizeGuidance[level][1]
        framePyramids.append(np.zeros((frameCount, lvlHeight, lvlWidth, frameChannels), dtype="float32"))

    print("Started filtering...")
    for frame in range(frameCount):
        currFrame = framesYIQ[frame]
        currPyramid,_ = buildLaplacian(currFrame,levels=pyramidLvls)
        for level in range(pyramidLvls):
            framePyramids[level][frame,:,:,:] = currPyramid[level]
    
    filtered = idealBandPassFiltering(framePyramids, sizeGuidance, lowFreq, highFreq, samplingRate)

    print("Started magnification...")
    for frameIdx in range(frameCount):
        filteredLevels = []
        for level in range(pyramidLvls):
            filteredLevels.append(filtered[level][frameIdx,:,:,:])
        pixelsMagnified = pixelMagnification(filteredLevels, alpha, lambda_c)
        reconstructedFrame = reconstructVideoFrames(pixelsMagnified, chromeAttenuation=chromeAttenuation)
        reconstructedFrame += framesYIQ[frameIdx,:,:,:]
        frameReconstructedRGB = convertYIQ2RGB(reconstructedFrame[:,:,:])
        saveVideo(frameReconstructedRGB, reconstructedFn , writer, frameIdx)
    writer.release()
    print("Done...")