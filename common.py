# System imports
from os import path
import math

# Third-Party Imports
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

## helper function to convert from BGR to YIQ
def convertBGR2YIQ(input_frame):
    input_frame = input_frame[:,:,[2,1,0]]
    transform = np.array([[0.299,      0.587,        0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617, 0.31119955]])
    height, width, channels = input_frame.shape[0], input_frame.shape[1], input_frame.shape[2]
    outputFrame = np.zeros((height, width, channels), dtype="float32")
    outputFrame = np.dot(input_frame, transform.T.copy())
    return outputFrame

## helper function to convert from YIQ to RGB 
def convertYIQ2RGB(input_frame):
    transform = np.array([[1.0,      0.956,        0.619], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]])
    height, width, channels = input_frame.shape[0], input_frame.shape[1], input_frame.shape[2]
    outputFrame = np.zeros((height, width, channels), dtype="float32")
    outputFrame = np.dot(input_frame, transform.T.copy())
    return outputFrame[:,:,[2,1,0]]

## Helper function to build a Gaussian pyramid of a frame
def buildGaussian(input_img,levels=3, ksize=0, size=7):  
    pyramid = [cv2.GaussianBlur(input_img, (size,size),ksize)]
    for i in range(1,levels):
        gaussian = cv2.GaussianBlur(pyramid[i-1], (size,size),ksize)
        pyramid.append(gaussian[::2,::2])  
    return pyramid

## Helper function to build a Laplacian pyramid of a frame
def buildLaplacian(input_img, levels=3, ksize=0, size=7):
    gaussian = buildGaussian(input_img, levels = levels, ksize=ksize, size=size)
    sizes = []
    fi = [input_img]
    li = []
    for i in range(1,levels):
        l = cv2.GaussianBlur(fi[i-1],(size,size),ksize)
        li.append(l)
        f = l[::2,::2]
        fi.append(f)
    pyramid = [input_img - cv2.GaussianBlur(input_img, (size,size), ksize)]
    for i in range(1,levels-1):
        h = fi[i] - li[i]
        pyramid.append(h)
    pyramid.append(gaussian[-1])
    for level in pyramid:
        sizes.append(level.shape)
    return (pyramid,sizes)


## Helper function to reconstruct a frame from magnified laplacian pyramid
def reconstructVideoFrames(magnified, chromeAttenuation=0):
    if len(magnified[0].shape) == 3:
        multiChannel = True
    else:
        multiChannel = False
    pyr_levels = len(magnified)
    frameHeight, frameWidth = magnified[0].shape[0], magnified[0].shape[1] 
    if multiChannel:
        frameChannels = magnified[0].shape[2]
        finalFrame = np.zeros((frameHeight, frameWidth, frameChannels), dtype="float32")
    else:
        frameChannels = 1
        finalFrame = np.zeros((frameHeight, frameWidth), dtype="float32")
    temp = magnified[-1]
    for level in range(pyr_levels-2, -1, -1):
        currSize = (magnified[level].shape[1], magnified[level].shape[0])
        temp = cv2.pyrUp(temp, dstsize=currSize)
        temp = temp + magnified[level]
    finalFrame = temp
    if multiChannel:
        finalFrame[:,:,1] = finalFrame[:,:,1] * chromeAttenuation
        finalFrame[:,:,2] = finalFrame[:,:,2] * chromeAttenuation
    # finalFrame[:,:,:] += original[:,:,:]
    return finalFrame

## Helper function to save frames to disk & combine frames into a video file
def saveVideo(reconstFrames, reconstructedFn, writer, frameIdx):
    cv2.imwrite(reconstructedFn + "/frame%04d.png" %(frameIdx+1), 255*reconstFrames[:,:,:])
    writer.write(cv2.convertScaleAbs(255*reconstFrames[:,:,:]))