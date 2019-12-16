#!/usr/bin/env python
# coding: utf-8

# Timo van Overveld, 2019

"""Investigate the accuracy of the surface shape method by comparing the particle positions of the method with interpolated positions."""
# Input: directory locations with images (.tif)
# Output: data structure with particle positions in real space

# Imports
import numpy as np
import json
import os
import argparse
import fnmatch
from images2positions_functions import *


############################################################################
def find_particleaccuracy(iterate_order=False,poldegree=1,verbose=False,data=False):
     
    # Reading settings file
    settingsfile = os.path.abspath(os.getcwd())+'/data/settings_particleaccuracy.txt'
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

    # Mean water height during measurements [m]
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

    plots   = settings['plots']

    if verbose == False: verbose = settings['verbose']

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
    if data:
        dataarray = np.array(np.empty((4),dtype=float))

    # Open image
    for file in calClist:
        if verbose: print('File',file)

        image = readcropimage(file,bitdepth=16,bounds=bounds)

        # Find markers where particles are present
        markers = findparticles(image,thresholdvalue)
 
        # Remove particles from image
        image_noparticles = removeparticles(image,markers,method=3)

        # Find lines in pixel values
        lines = findlines(np.uint8(image/16),linespacingpx,centerpx)

        if verbose: print('Line positions found')
        
        # Correct for the particles that are not separated on the first try
        imagecorrected, markerscorrected = correctmarkers(image,markers,thresholdvalue)

        # Obtain the positions of the particles in pixels
        positionspix = particlepositions(imagecorrected,markerscorrected)

        #TODO: contintue here
    
        # Obtain all information on the lines
        xprojected = pix2realx(lines)
        xreal, Nlines = clusterlines(lines,linespacing,Nlines=17)
    
        H  = fitHpolynomial(xreal,xprojected,xc[0],Hc[0],n,surfaceshapeorder)
        Hp = np.polyder(H)

        # Approximate the odd line positions by linear interpolation (li) of neighbouring lines
        #lines_li = (linesevenproj[0:-1]+linesevenproj[1:])/2
        # This was faulty, because it assumed the other line to be in between the other lines. The infromation stored in xprojected is in fact available from the image, so predict the line in the center.
        # Index to which line it belongs
        lines_li = np.empty(0,dtype=float) 
        for i in range(0,np.size(xreal),1):
            if xreal[i] in linesevenreal:
                # Line is known
                pass
            else:
                # Line unknown, find which 2 known lines are closest
                #id1, id2 = np.argsort(abs(xreal[i]-linesevenreal))[0:2] # Known lines
                #id1 = np.argwhere(linesevenreal[id1]==xreal)[0][0] # Convert to all idx of all lines
                #id2 = np.argwhere(linesevenreal[id2]==xreal)[0][0]
                # Now use id1 and id2 to inter/extrapolate to i
                #y = xreal, x = xprojected. 3 xprojected are known
                #lineinter = (xreal[id2]-xreal[id1])/(xprojected[id2]-xprojected[id1])*(xprojected[i]-xprojected[id1])+xreal[id1]
                #lines_li = np.append(lines_li,lineinter)
                
                idx_er = np.argsort(abs(xreal[i]-linesevenreal))[0:poldegree] # Known lines
                _, _, idx = np.intersect1d(linesevenreal[idx_er],xreal,return_indices=True) # Convert to all idx of all lines
                # Now use id1 and id2 to inter/extrapolate to i
                #y = xreal, x = xprojected. 3 xprojected are known
                fit = np.polyfit(xprojected[idx],xreal[idx],np.size(xreal[idx])-1)
                p = np.poly1d(fit)
                
                lines_li = np.append(lines_li,p(xprojected[i]))
        
        # Errors reconstructed lines
        abserror_rs = abs(linesreconstructed-linesoddreal)
        relerror_rs = abs(linesreconstructed-linesoddreal)/linespacing
        
        # Errors projected lines without reconstruction (projected2real)
        abserror_pr = abs(linesoddproj-linesoddreal)
        relerror_pr = abs(linesoddproj-linesoddreal)/linespacing

        # Errors linearly interpolated lines
        abserror_li = abs(lines_li-linesoddreal)
        relerror_li = abs(lines_li-linesoddreal)/linespacing

        if data:
            dataarray = np.vstack((dataarray,np.transpose(np.array([linesoddreal,linesreconstructed,linesoddproj,lines_li]))))

        # Optional, print errors
        if verbose:
            #print('Real line position\n',linesoddreal)
            #print('Projected line position\n',linesoddproj) 
            #print('Reconstructed lines position\n',linesreconstructed)
            #print('Interpolated lines position\n',lines_li)
            #print('Reconstructed, projected, linearly interpolated') 
            print(np.mean(relerror_rs),np.mean(relerror_pr),np.mean(relerror_li))
             
            #print('Relative error\n',relerror_rs)
            #print('Maximum error=\n',max(abserror_rs),'(abs)',max(relerror_rs),'(rel)')

        if plots:
            plt.figure(figsize=(12,8))
            plt.xlabel('Line number')
            plt.ylabel('Relative error [-]')
            plt.ylim(1e-5,1.0)
            plt.legend()
            plt.grid(axis='y')
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            #plt.semilogy(linesreconstructed,relerror_rs,'o',label='reconstructed')
            #plt.semilogy(linesoddproj      ,relerror_pr,'^',label='projected')
            #plt.semilogy(lines_li          ,relerror_li,'s',label='interpolated')
            #plt.xlabel('Position along channel [m]')
            #plt.ylabel('Relative error [-]')
            #plt.legend()
            #plt.draw()
            #plt.waitforbuttonpress(0)
            #plt.close()
    
    if data:
        # Pop the first row, contains empty
        dataarray = np.delete(dataarray,0,0) 
        filename = 'data/particleaccuracy_'+str(iterate_order)+'_'+str(poldegree)
        np.save(filename,dataarray)
        print('Data saved in ',filename)

if __name__ == "__main__":  
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default=1, help='polynomial degree for interpolation')
    parser.add_argument('-o', action='store_true', help='Enable the iterative fitting order')
    parser.add_argument('-v', action='store_true', help='Set verbosity')
    parser.add_argument('-d', action='store_true', help='Print data')
    args = parser.parse_args()

    find_particleaccuracy(iterate_order=args.o, poldegree=args.p, verbose=args.v, data=args.d)


