#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Investigate the accuracy of the surface shape method by skipping lines in the analysis and reconstructing their positions."""
# Input: directory locations with images (.tif)
# Output: data structure with lines in real space

# Imports
import numpy as np
import json
import os
import fnmatch
from images2positions_functions import *


############################################################################
def find_lineaccuracy():
     
    # Reading settings file
    settingsfile = os.path.abspath(os.getcwd())+'/data/settings_lineaccuracy.txt'
    f = open(settingsfile, 'r')
    settings = json.loads(f.read())

    # Directories
    basedir = settings['basedir']
    calibrationdir = basedir + settings['calibrationdir']
    measurementdir = basedir + settings['measurementdir']

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

    plots   = True#settings['plots']

    verbose = settings['verbose']

    if verbose: print('Constants read')

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

    # Measurement list
    calClist = [measurementdir + i for i in os.listdir(measurementdir) if i.endswith(".tif")]

    if verbose: print('User inputs read')

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

    if plots:
        plt.figure(figsize=(12,8))
        plt.scatter(xpix,xreal,marker='x',color='red',label='Data')

        x = np.linspace(0,imagesize[1],500)
        plt.plot(x,pix2realx(x),label='Best polynomial fit')

        plt.xlabel('Image position [px]')
        plt.ylabel('Real position [m]')
        plt.grid()
        plt.legend()
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

    if verbose:
        print('Camera warping mapped')
        print('Along channel (x):\n',    pix2realx)
        print('Across channel (y):',    pix2realy)

    ###########################

    # Camera position
    # Finding the camera position by using known flat water heights.

    xpix, H = pixHlist(calBlist,Hlist,bounds=bounds,centerpx=centerpx,linespacingpx=linespacingpx)
    xprojected = pix2realx(xpix)
    xreal, Nlines = clusterlines(xpix,Nlines=Nlines,linespacing=linespacing)
 
    xc, Hc = cameraposition(xprojected,xreal,H,n)

    if plots:
        xin = np.vstack((xprojected,H,n*np.ones(np.shape(H))))
        popt, pcov = optimization.curve_fit(func_flatsurf, xin, xreal)
        # perr = np.sqrt(np.diag(pcov))

        plt.figure(figsize=(12,8))
        plt.scatter(xprojected,xreal,label='Data')
        plt.plot(xin[0],func_flatsurf(xin,xc[0],Hc[0]),'r-',label='Best fit')
        plt.xlabel('Projected x')
        plt.ylabel('Real world x')
        plt.legend()
        plt.grid()
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()

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

        if verbose: print('Line positions found')

        # Obtain water surface shape
        xprojected = pix2realx(lines)
        xreal, Nlines = clusterlines(lines,linespacing,Nlines=17)

        H  = Hpolynomial(xreal,xprojected,xc[0],Hc[0],n)
        Hp = np.polyder(H)

        # Look only at the even lines (0,2,4...)
        linesoddreal = xreal[0::2]
        linesoddproj = xprojected[0::2]
        
        print(linesoddreal,linesoddproj)

        H  = Hpolynomial(linesoddreal,linesoddproj,xc[0],Hc[0],n)
        Hp = np.polyder(H)


        

        if plots:
            plt.figure(figsize=(12,8))
            plt.imshow(image,origin='low')
            for i in lines[0::2]:
                plt.axvline(i,linewidth=1,color='red')
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            #plt.figure(figsize=(12,8))
            #plt.scatter(xprojected,0*xprojected,color='blue')
            #plt.scatter(xreal,0*xreal,color='red')
            #x = np.linspace(0,0.4,100)
            #plt.fill_between(x,0,H(x),color='blue',alpha=0.1)
            #plt.draw()
            #plt.waitforbuttonpress(0)
            #plt.close()

if __name__ == "__main__":
    find_lineaccuracy()

