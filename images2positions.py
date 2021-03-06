#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""A combination of experimental calibration and particle location correction"""
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

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('-f', help="settings file")
    args = parser.parse_args()

    verbose = False
    if args.verbose:
        verbose = True
        print("verbosity turned on")

    # Reading settings file
    f = open(args.f, 'r')
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

    # Mean water height [m]
    Hmean = settings['Hmean']

    # Cropping bounds (top, bottom, left, right)
    bounds = settings['bounds']

    # Central pixels to exclude
    centerpx = settings['centerpx']

    # Thresholding values
    thresholdvalue = settings['thresholdvalue']

    # Fitting order (C*x^order+...)
    warpingorder = settings['warpingorder']
    surfaceshapeorder = settings['surfaceshapeorder']

    reconstruction = settings['reconstruction']

    plots   = settings['plots']

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

    if np.size(calClist) == 0:
        print('No measurement files specified, quitting.')
        quit()
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
    xreal, NlinesB = clusterlines(xpix,Nlines=Nlines,linespacing=linespacing)

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

        # Find markers where particles are present
        markers = findparticles(image,thresholdvalue)

        # Remove particles from image
        image_noparticles = removeparticles(image,markers,method=3)

        # Find lines in pixel values
        lines = findlines(np.uint8(image_noparticles/16),linespacingpx,centerpx,plot=False)

        if verbose: print('Line positions found')

        # Correct for the particles that are not separated on the first try
        imagecorrected, markerscorrected = correctmarkers(image,markers,thresholdvalue)

        # Obtain the positions of the particles in pixels
        positionspix = particlepositions(imagecorrected,markerscorrected)
        positions    = np.asarray([pix2realx(positionspix[:,0]),pix2realy(positionspix[:,1])])

        if verbose: print('Particle positions found [px] and projected [m]')

        #Convert line positions (px -> m projected)
        xprojected = pix2realx(lines)
        #Use Number of lines that we known that are there, from calibration B
        xreal, Nlines = clusterlines(lines,linespacing,Nlines=NlinesB)

        if reconstruction == True:
            #######################################
            # Surface shape reconstruction method #
            #######################################

            H  = fitHpolynomial(xreal,xprojected,xc[0],Hc[0],n,order=surfaceshapeorder,Hmean=Hmean)
            Hp = np.polyder(H)

            # Convert to real meters (correction only for x coordinate)
            positionsreal = np.asarray([projected2real(positions[0,:],H,Hp,xc[0],Hc[0],n),positions[1,:]])
        elif reconstruction == False:
            #############################
            # Line interpolation method #
            #############################

            # Approximate the particle positions by linear interpolation (li) of neighbouring lines
            # The order of the interpolation is given by surfaceshapeorder

            # Array to overwrite
            positionsreal = np.asarray([positions[1,:],positions[1,:]])

            # Loop over particles (projected) x positions
            for i in range(0,np.size(positions[0,:]),1):
                x = positions[0,i]
                # Find indices of the poldegree closest lines
                idx = np.argsort(abs(x-xreal))[0:surfaceshapeorder+1]
                # Now use idx inter/extrapolate using a polynomial fit
                # x = xprojected[idx] (projected positions lines), y = xreal[idx] (real positions lines)
                fit = np.polyfit(xprojected[idx],xreal[idx],np.size(xreal[idx])-1)
                p = np.poly1d(fit)

                # Obtain the real x position
                positionsreal[0,i] = p(x)


        savename = file[0:-3]+'dat'
        np.savetxt(savename,positionsreal)
        if verbose: print('Particle positions [m] stored in',savename)

        if plots:
            plotimage = image*1
    #         plotimage[markerscorrected == -1] = 5000

            plt.figure(figsize=(20,20))
            plt.imshow(plotimage,origin='low')
            for i in lines:
                plt.axvline(i,linewidth=1,color='red')
            plt.scatter(positionspix[:,0],positionspix[:,1],color='red',marker='x',s=5)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            plt.figure(figsize=(12,8))
            plt.scatter(xprojected,0*xprojected,color='blue',label='projected positions')
            plt.scatter(xreal,0*xreal,color='red',label='real positions')
            x = np.linspace(np.min((np.min(xprojected),np.min(xreal))),np.max((np.max(xprojected),np.max(xreal))),100)

            if reconstruction:
                plt.fill_between(x,0,H(x),color='blue',alpha=0.1)
                
                data = (xprojected,H,xc[0],Hc[0])
                xw = optimization.root(intersection,xprojected,args=data)
                xw = xw.x
                for i in range(0,np.size(xprojected),1):
                    plt.plot([xprojected[i],xw[i]],[0,H(xw[i])],'k--')
                    plt.plot([xreal[i],xw[i]],[0,H(xw[i])],'k')
                    plt.plot([xw[i],xc[0]],[H(xw[i]),Hc[0]],'k')
                plt.scatter(xw,H(xw),marker='x')
                plt.ylim(-0.01,0.1)
                plt.legend()
                plt.draw()
                plt.waitforbuttonpress(0)
                plt.close()

            # Compare images
            plt.figure(figsize=(12,8))
            plt.scatter(positions[0,:],positions[1,:],color='blue',label='projected positions')
            plt.scatter(positionsreal[0,:],positionsreal[1,:],color='red',label='real positions',marker='x')
            plt.xlabel('Distance along channel [m]')
            plt.ylabel('Distance across channel [m]')
            plt.fill_between([0,np.max(positions[0,:])],0.1,0.11,color='gray')
            plt.fill_between([0,np.max(positions[0,:])],0,-0.01,color='gray')
    #         plt.axes().set_aspect('equal')
            plt.legend()
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()


if __name__ == "__main__":
    main()
