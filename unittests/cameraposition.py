#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""A unittest to check whether the camera position is consistently found"""
# Input: directory locations with images (.tif)
# Output: data structure with particle positions in real space

# Imports
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.optimize as optimization
from scipy.signal import find_peaks
from scipy import ndimage
import os
import fnmatch
import json

from images2positions_functions import *

############################################################################


def main():

    # Reading settings file
    f = open(args.f, 'r')
    settings = json.loads(f.read())

    # Directories
    basedir = settings['basedir']
    calibrationdir = basedir + settings['calibrationdir']

    # refraction indices
    nair   = settings['nair']
    nwater = settings['nwater']
    n      = nair/nwater

    # Line spacing (real world) [m]
    linespacing = settings['linespacing']
    linespacingpx = settings['linespacingpx']

    # Channel width [m]
    channelwidth = settings['channelwidth']

    # Cropping bounds (top, bottom, left, right)
    bounds = settings['bounds']

    # Central pixels to exclude
    centerpx = settings['centerpx']

    # Fitting order (C*x^order+...)
    warpingorder = settings['warpingorder']

    #####################################


    # Filelist
    # Allocation
    flist = np.empty(0,dtype=int)
    Hlist = np.empty(0,dtype=int)

    for (dirpath, dirnames, filenames) in os.walk(calibrationdir):
        if np.size(fnmatch.filter([dirpath],'*H*'))==1:
            f = np.asarray(fnmatch.filter(filenames,'*.tif*'))
            flist = np.append(flist,f)

            if np.size(f) > 0:
                H = dirpath[dirpath.find('H')+1:]
                for i in range(0,np.size(f),1):
                    Hlist = np.asarray(np.append(Hlist,int(H)))

    # Warping list
    calAlist = [calibrationdir + 'H0/' + i for i in flist[Hlist==0]]

    # Camera position list
    calBlist = [calibrationdir + 'H' + str(Hlist[i]) + '/' + flist[i] for i in range(0,np.size(flist))]

    #############################################

    # Camera warping
    # Linking the pixels to the real world coordinates, because the camera warps the image. This script gives the transformation from pixels to real world coordinates for a water height of 0.

    #Obtain line positions
    xpix, Hreal = pixHlist(calAlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)

    xreal, Nlines = clusterlines(xpix,linespacing)

    # Fit the warping function through the data
    pix2realx = pixrealfit(xpix, xreal, warpingorder)

    # Now a similar thing for the y-coordinate, across the width of the channel
    imagesize = np.shape(readcropimage(calAlist[0],bounds=bounds))
    pix2realy = pixrealfit([0,imagesize[0]],[0,channelwidth],1)

    ###########################

    # Camera position
    # Finding the camera position by using known flat water heights.

    xpix, H = pixHlist(calBlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)
    xprojected = pix2realx(xpix)
    xreal, Nlines = clusterlines(xpix,Nlines=Nlines,linespacing=linespacing)
 
    xc, Hc = cameraposition(xprojected,xreal,H,n)

    #################################


if __name__ == '__main__':
    unittest.main()
