#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Investigate the accuracy of the surface shape method by skipping lines in the analysis and reconstructing their positions."""
# Input: directory locations with images (.tif)
# Output: data structure with lines in real space

# Imports
import numpy as np
from matplotlib import pyplot as plt
import os
import fnmatch
import json

from images2positions_functions import *

############################################################################


def main():
    
    verbose = True

    datadir = '../data/'
    settingsfile = datadir+'settings_lineaccuracy.txt'
    
    # Reading settings file
    f = open(settingsfile, 'r')
    settings = json.loads(f.read())
    f.close()

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

    # Thresholding values
    thresholdvalue = settings['thresholdvalue']

    # Fitting order (C*x^order+...)
    warpingorder = settings['warpingorder']
    surfaceshapeorder = settings['surfaceshapeorder']

    plots   = settings['plots']

    verbose = settings['verbose']

    if verbose: print('Constants read')

    #####################################

    # Warping
    calAlist = datadir+'H0.tif'

    # Camera position
    calBlist = [datadir+'H0.tif',datadir+'H7.tif']

    # Measurement list
    calClist = [datadir+'H7_wave.tif',datadir+'H7.tif']

    Hlist = np.asarray([0,7])
    
    if verbose: print('User inputs read')

    #############################################

    # Camera warping
    # Linking the pixels to the real world coordinates, because the camera warps the image. This script gives the transformation from pixels to real world coordinates for a water height of 0.

    #Obtain line positions
    xpix, Hreal = pixHlist(calAlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)

    xreal, Nlines = clusterlines(xpix,linespacing,Nlines=17)

    # Fit the warping function through the data
    pix2realx = pixrealfit(xpix, xreal, warpingorder)

    if verbose:
        print('Camera warping mapped')
        print('Along channel (x):\n',    pix2realx)

    ###########################

    # Camera position
    # Finding the camera position by using known flat water heights.

    xpix, H = pixHlist(calBlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)
    xprojected = pix2realx(xpix)
    xreal, Nlines = clusterlines(xpix,Nlines=Nlines,linespacing=linespacing)
 
    xc, Hc = cameraposition(xprojected,xreal,H,n)

    if verbose:
        print('Camera position')
        print('xc = ',format(xc[0],'.3f'), '+-',format(xc[1],'.3f')+' m')
        print('Hc = ',format(Hc[0],'.3f'), '+-',format(Hc[1],'.3f')+' m')

    #################################

    # Surface shape

    # Open image
    for file in calClist:
        if verbose: print('File',file)

        image = readcropimage(file,bitdepth=16,bounds=bounds)

        # Find lines in pixel values
        lines = findlines(np.uint8(image/16),linespacingpx,centerpx)

        if verbose: 
            print('Line positions found')
            print(lines)
    
        # Obtain water surface shape
        xprojected = pix2realx(lines)
        xreal, Nlines = clusterlines(lines,linespacing,Nlines=17)

        H  = Hpolynomial(xreal,xprojected,xc[0],Hc[0],n)
        Hp = np.polyder(H)

if __name__ == "__main__":
    main()
